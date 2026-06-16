import base64
import io
import os
import shutil
import time
import asyncio
import cv2
import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI(title="GLM 视频帧实时分析系统")

# 临时视频存放目录
UPLOAD_DIR = "temp_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# GLM API 配置
GLM_API_URL = "http://192.168.21.128:8001/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-admin-ford-glm-5b-123"
}

# 💡 帧过滤阈值配置：MSE 均方误差低于该值则判定为“画面无显著变化”
# 高速公路行驶一般设在 10 ~ 30 之间比较灵敏；堵车静止时 MSE 通常会掉到 5 以下
FRAME_DIFF_THRESHOLD = 1500.0


def cv2_to_pil(cv2_img) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def frame_to_base64(cv2_frame) -> str | None:
    """将视频帧转换为缩略图 Base64 供前端展示和大模型输入"""
    try:
        pil_img = cv2_to_pil(cv2_frame)
        pil_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=55, optimize=True)
        b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_data}"
    except Exception as e:
        print(f"❌ 帧 Base64 转换失败: {e}")
        return None


def request_model(prompt: str, base64_url: str) -> str:
    """请求 GLM 模型"""
    payload = {
        "model": "glm-4v",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": base64_url}},
                ],
            }
        ],
        "temperature": 0.2,
        "max_tokens": 150
    }
    try:
        res = requests.post(GLM_API_URL, headers=HEADERS, json=payload, timeout=12)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"大模型请求失败: {str(e)}"


def frame_diff_roi(img1, img2):
    """计算两帧画面特定 ROI 区域的均方误差 (MSE)"""
    h, w, _ = img1.shape
    start_y = int(h * 0.2)
    end_y = int(h * 0.7)
    roi1 = img1[start_y:end_y, :]
    roi2 = img2[start_y:end_y, :]
    g1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
    return np.mean((g1.astype(float) - g2.astype(float)) ** 2)


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """处理视频文件上传，返回临时保存路径"""
    file_path = os.path.join(UPLOAD_DIR, f"{int(time.time())}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"success": True, "video_path": file_path}


@app.websocket("/ws/analyze")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 核心控制链路，实现实时抽帧与推送"""
    await websocket.accept()
    try:
        # 1. 接收前端传来的配置参数
        config = await websocket.receive_json()
        video_path = config.get("video_path")
        prompt = config.get("prompt", "请分析图片中是否包含动物、植物或建筑物。")
        interval_sec = float(config.get("interval_sec", 5.0))

        if not video_path or not os.path.exists(video_path):
            await websocket.send_json({"type": "error", "message": "视频文件不存在"})
            return

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps == 0 or np.isnan(fps):
            await websocket.send_json({"type": "error", "message": "无法读取视频或 FPS 为 0"})
            cap.release()
            return

        frame_interval = int(fps * interval_sec)
        frame_id = 0

        # 💡 新增：用于缓存上一次成功交由模型处理的视频帧
        last_processed_frame = None

        await websocket.send_json({"type": "status", "message": f"🎬 视频加载成功 (FPS: {fps:.2f})，开始分析..."})

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % frame_interval == 0:
                current_time_sec = frame_id / fps

                # 1. 优先生成当前帧的 Base64（无论跳不跳过，前端都要看图）
                base64_data = frame_to_base64(frame)

                if base64_data:
                    # 2. 画面相似度甄别过滤
                    if last_processed_frame is not None:
                        diff_score = frame_diff_roi(last_processed_frame, frame)
                        print(f"⏱️ 时间点 {current_time_sec:.1f}s | 帧差异度(MSE): {diff_score:.2f}")

                        if diff_score < FRAME_DIFF_THRESHOLD:
                            # 💡 关键改动：将 base64_data 作为 image 参数一起传给前端
                            await websocket.send_json({
                                "type": "frame_skip",
                                "time": f"{current_time_sec:.1f}s",
                                "image": base64_data,
                                "reason": f"与前一采样帧过于相似 (MSE: {diff_score:.1f} < {FRAME_DIFF_THRESHOLD})"
                            })
                            frame_id += 1
                            continue

                    # 3. 未被跳过的帧，正常交给大模型处理
                    last_processed_frame = frame.copy()

                    # 先发给前端展示“加载中”
                    await websocket.send_json({
                        "type": "frame_start",
                        "time": f"{current_time_sec:.1f}s",
                        "image": base64_data
                    })

                    # 异步执行大模型请求
                    loop = asyncio.get_event_loop()
                    start_time = time.time()
                    result = await loop.run_in_executor(None, request_model, prompt, base64_data)
                    cost_time = time.time() - start_time

                    # 推送大模型分析结果
                    await websocket.send_json({
                        "type": "frame_result",
                        "time": f"{current_time_sec:.1f}s",
                        "result": result,
                        "cost": f"{cost_time:.2f}s"
                    })

                await asyncio.sleep(0.01)

            frame_id += 1

        cap.release()
        await websocket.send_json({"type": "done", "message": "🎉 视频全部分析完成！"})

        if os.path.exists(video_path):
            os.remove(video_path)

    except WebSocketDisconnect:
        print("💡 前端连接已断开")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": f"运行异常: {str(e)}"})


@app.get("/")
async def get():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
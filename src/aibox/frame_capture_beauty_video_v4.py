import asyncio
import base64
import os
import torch
import numpy as np
import cv2
import clip
import json
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware

# ---------- 模型加载（保持单例） ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
use_fp16 = True if device == "cuda" else False

clip_model, clip_preprocess = clip.load("ViT-L/14", device=device, jit=False)
if use_fp16:
    clip_model = clip_model.half()


class AestheticPredictorV2(torch.nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


model_aesthetic = AestheticPredictorV2(768).to(device)
if os.path.exists("ava+logos-l14-linearMSE.pth"):
    model_aesthetic.load_state_dict(torch.load("ava+logos-l14-linearMSE.pth", map_location=device))
model_aesthetic.eval()
if use_fp16:
    model_aesthetic = model_aesthetic.half()


def get_laion_v2_score(cv2_frame):
    """ 计算 LAION V2 美学打分 """
    rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    image_tensor = clip_preprocess(pil_img).unsqueeze(0).to(device)

    if use_fp16:
        image_tensor = image_tensor.half()

    with torch.inference_mode():
        image_features = clip_model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        prediction = model_aesthetic(image_features)

    return float(prediction.item())


# ===================== 图像质量算法 =====================

def frame_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def frame_diff_roi(img1, img2):
    h, w, _ = img1.shape
    start_y, end_y = int(h * 0.2), int(h * 0.7)
    g1 = cv2.cvtColor(img1[start_y:end_y, :], cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2[start_y:end_y, :], cv2.COLOR_BGR2GRAY)
    return np.mean((g1.astype(float) - g2.astype(float)) ** 2)


def estimate_dynamic_range(img_bgr, percentile_low=1, percentile_high=99) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    p_low = np.percentile(gray, percentile_low)
    p_high = np.percentile(gray, percentile_high)
    return float(p_high - p_low)


def estimate_noise_std(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    edge_strength = np.abs(lap)

    smooth_mask = edge_strength < np.percentile(edge_strength, 30)
    if np.sum(smooth_mask) < 100:
        return float(np.std(gray))

    noise_map = cv2.GaussianBlur(gray, (3, 3), 0)
    noise_residual = gray - noise_map
    return float(np.std(noise_residual[smooth_mask]))


def check_frame_quality(frame, prev_frame=None):
    metrics = {
        "sharpness": 0.0,
        "brightness": 0.0,
        "dynamic_range": 0.0,
        "noise_std": 0.0,
        "mse_diff": 0.0,
        "aesthetic_score": 0.0,
        "remark": "通过验证"
    }

    sharp = frame_sharpness(frame)
    metrics["sharpness"] = round(sharp, 2)
    if sharp < 120:
        metrics["remark"] = f"未通过: 清晰度过低 ({sharp:.1f} < 120)"
        return False, metrics

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    metrics["brightness"] = round(brightness, 2)
    if brightness < 40 or brightness > 220:
        metrics["remark"] = f"未通过: 曝光异常 (亮度: {brightness:.1f})"
        return False, metrics

    metrics["dynamic_range"] = round(estimate_dynamic_range(frame), 2)
    metrics["noise_std"] = round(estimate_noise_std(frame), 2)

    if prev_frame is not None:
        diff = frame_diff_roi(prev_frame, frame)
        metrics["mse_diff"] = round(diff, 2)
        if diff < 1500:
            metrics["remark"] = f"未通过: 与前图高度相似 (MSE: {diff:.1f})"
            return False, metrics

    aesthetic_score = get_laion_v2_score(frame)
    metrics["aesthetic_score"] = round(aesthetic_score, 3)

    if aesthetic_score >= 7.5:
        metrics["remark"] = "卓越/艺术级"
    elif aesthetic_score >= 6.0:
        metrics["remark"] = "优秀/专业级"
    else:
        metrics["remark"] = "常规高质量"

    return True, metrics


# ===================== FastAPI 核心路由 =====================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/analyze/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()
    last_saved_frame = None
    results = []

    try:
        while True:
            # 1. 持续接收前端流式抛上来的帧 JSON 数据
            data = await websocket.receive_text()
            msg = json.loads(data)

            # 如果前端发来结束信号，退出循环
            if msg.get("type") == "finish":
                break

            if msg.get("type") == "analyze_frame":
                frame_id = msg["frame_id"]
                timestamp = msg["timestamp"]
                b64_img_raw = msg["image_base64"]

                # 2. 解析 Base64 图片
                if "," in b64_img_raw:
                    b64_img_raw = b64_img_raw.split(",")[1]

                img_bytes = base64.b64decode(b64_img_raw)
                np_array = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                # 3. 核心质量与美学跑分
                is_passed, metrics = check_frame_quality(frame, prev_frame=last_saved_frame)

                if is_passed:
                    last_saved_frame = frame.copy()
                    results.append({
                        "frame_id": frame_id,
                        "timestamp_sec": timestamp,
                        **metrics
                    })

                # 4. 算完一帧，立刻把结果弹射回前端渲染
                await websocket.send_json({
                    "type": "frame_result",
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "is_passed": is_passed,
                    "metrics": metrics,
                    "image_base64": msg["image_base64"]  # 原样带回前端展示（由于前端抽帧本来就是高清1280，这里无需二次resize）
                })

        # 5. 前端发送完成信号后，生成 Excel 报表
        if results:
            output_excel = f"reports/report_{task_id}.xlsx"
            os.makedirs("reports", exist_ok=True)
            # todo: 如果需要，可在此处恢复你 save_to_styled_excel(results, output_excel) 的逻辑
            pass

    except WebSocketDisconnect:
        print(f"任务 {task_id} 客户端断开连接")
    except Exception as e:
        print(f"任务 {task_id} 处理发生异常: {str(e)}")
    finally:
        pass


if __name__ == '__main__':
    import uvicorn
    # 保持单进程模式，防止多进程环境下 GPU 模型加载冲突
    uvicorn.run(app, host="0.0.0.0", port=8000)
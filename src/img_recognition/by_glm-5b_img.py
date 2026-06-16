import base64
import io
import os
import time
import cv2
import numpy as np
import requests
from PIL import Image

# ==================== 配置区域 ====================
VIDEO_PATH = r"C:\pablo\05_self_code\yolo-sam-lama\data\video\高速行车记录仪视频.mp4"
OUTPUT_DIR = "frames_test"
SAVE_LOCAL_FRAME = False  # 是否将抽取的帧保存到本地硬盘
# GLM API 配置（请根据你的部署环境修改 URL 和 Headers）
API_URL = "http://192.168.21.128:8001/v1/chat/completions"  # 替换为你的本地或云端 GLM API 地址
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-admin-ford-glm-5b-123"  # 如果不需要 Key 可以留空或删除
}
# ==================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def cv2_to_pil(cv2_img) -> Image.Image:
    """将 OpenCV 的 BGR 图像转换为 PIL 的 RGB 图像"""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def frame_to_base64(cv2_frame) -> str | None:
    """直接将内存中的视频帧转换为 Base64 字符串"""
    try:
        pil_img = cv2_to_pil(cv2_frame)
        # 适当缩小分辨率，降低传输带宽和多模态大模型的处理延迟
        pil_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        # 压缩质量设为 55%，进一步减少数据量
        pil_img.save(buffer, format="JPEG", quality=55, optimize=True)
        b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_data}"
    except Exception as e:
        print(f"❌ 帧 Base64 转换失败: {e}")
        return None


def request_model(url: str, heads: dict[str, str], base64_url: str) -> str:
    """请求 GLM 模型进行视觉分析"""
    payload = {
        "model": "glm-4v",  # 根据你实际使用的本地/线上模型名称填写，例如 glm-4v 或 glm-5b-q4
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请分析图片中是否包含动物、植物或建筑物。如果存在，请简单介绍；如果不存在，请回答'未检测到目标'。"
                    },
                    {"type": "image_url", "image_url": {"url": base64_url}},
                ],
            }
        ],
        "temperature": 0.2,
        "max_tokens": 150
    }
    try:
        # 适当延长超时到 10s，多模态模型推理通常比纯文本慢
        res = requests.post(url, headers=heads, json=payload, timeout=10)
        res.raise_for_status()

        # 兼容标准 OpenAI 格式的返回
        res_json = res.json()
        return res_json["choices"][0]["message"]["content"]
    except Exception as e:
        return f"大模型请求失败: {str(e)}"


def process_video_analysis(interval_sec: float):
    """
    核心主流程：每隔 interval_sec 秒抽取一帧，并提交给 GLM 模型分析打印
    """
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ 错误：找不到视频文件 -> {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0 or np.isnan(fps):
        print("❌ 错误：无法读取视频或 FPS 为 0")
        cap.release()
        return

    print(f"🎬 视频加载成功 | FPS: {fps:.2f}")

    # 计算跳帧步长
    frame_interval = int(fps * interval_sec)
    frame_id = 0
    analysis_count = 0

    print(f"🚀 开始分析视频，每 {interval_sec} 秒分析一帧...\n" + "=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 判断是否到达抽帧点
        if frame_id % frame_interval == 0:
            # 计算当前帧在视频中的时间点
            current_time_sec = frame_id / fps
            analysis_count += 1

            print(f"\n[帧索引: {frame_id} | 视频时间点: {current_time_sec:.1f}s]")

            # 可选：是否保存到本地
            if SAVE_LOCAL_FRAME:
                frame_name = f"frame_{analysis_count:04d}_{current_time_sec:.1f}s.jpg"
                cv2.imwrite(os.path.join(OUTPUT_DIR, frame_name), frame)

            # 1. 将当前帧转换为 Base64
            base64_data = frame_to_base64(frame)

            if base64_data:
                # 2. 调用大模型
                print("🤖 正在请求 GLM 模型分析...")
                start_time = time.time()
                result = request_model(API_URL, HEADERS, base64_data)
                end_time = time.time()

                # 3. 打印展示结果
                print(f"⏱️ 耗时: {end_time - start_time:.2f} 秒")
                print("📝 分析结果:")
                print(f"{result}")
                print("-" * 50)
            else:
                print("⚠️ 该帧转换 Base64 失败，跳过。")

        frame_id += 1

    cap.release()
    print("\n🎉 视频全部分析完成！")


if __name__ == "__main__":
    # 设定每 5 秒抽一帧进行分析
    process_video_analysis(interval_sec=5.0)
import base64
import io

import cv2
import time
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from img_recognition.by_yolo26 import request_model

GLM_5B_Q4_URL = "http://192.168.21.128:8001/v1/chat/completions"
GLM_Q4_HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer sk-admin-ford-glm-5b-123"}
# ======================
# 配置区（按需改）
# ======================

VIDEO_INPUT = r"C:\Users\pablozhao\Documents\WXWork\1688857975789108\Cache\File\2026-06\AIGC素材_运动模糊模拟不同车速_车外动物\车速测试素材\演示视频\010kph_演示.mp4"
VIDEO_OUTPUT = "data/output/video_result.mp4"

MODEL_PATH = "models/yolo26n.pt"

FRAME_SKIP = 30          # 每 30 帧抽 1 帧请求大模型
CONF_THRES = 0.1
IOU_THRES = 0.4
IMG_SIZE = 1280

# ======================
# 工具函数
# ======================

def frame_to_base64(frame, max_size=(512, 512)) -> str:
    """OpenCV BGR → base64 data url"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    buf = np.asarray(bytearray(), dtype="uint8")

    with io.BytesIO() as buffer:
        img.save(buffer, format="JPEG", quality=55, optimize=True)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{b64}"


def render_text_on_frame(frame_bgr: np.ndarray, text: str) -> np.ndarray:
    """
    把大模型描述渲染到视频帧底部
    """
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    font_size = max(16, int(w * 0.022))
    font = ImageFont.load_default()

    # 自动换行
    lines = []
    line = ""
    for ch in text:
        if draw.textbbox((0, 0), line + ch, font=font)[2] > w * 0.92:
            lines.append(line)
            line = ch
        else:
            line += ch
    lines.append(line)

    line_h = font_size + 10
    box_h = len(lines) * line_h + 20

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle([0, h - box_h, w, h], fill=(0, 0, 0, 180))
    img = Image.alpha_composite(img, overlay)

    draw = ImageDraw.Draw(img)
    y = h - box_h + 10
    for l in lines:
        draw.text((20, y), l, font=font, fill=(255, 255, 255, 255))
        y += line_h

    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


# ======================
# 主流程
# ======================

def video_detection(video_path: str, output_path: str):
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    frame_id = 0
    last_description = "正在识别中..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---- YOLO 检测 ----
        results = model.predict(
            frame,
            conf=CONF_THRES,
            iou=IOU_THRES,
            imgsz=IMG_SIZE,
        )[0]

        # 画框
        if len(results) > 0:
            frame = results.plot()

        # ---- 抽帧请求大模型 ----
        if frame_id % FRAME_SKIP == 0:
            try:
                b64 = frame_to_base64(frame)
                last_description = request_model(
                    GLM_5B_Q4_URL, GLM_Q4_HEADERS, b64
                )
            except Exception as e:
                last_description = f"模型请求失败: {e}"

        # ---- 渲染文字 ----
        frame = render_text_on_frame(frame, last_description)

        writer.write(frame)
        frame_id += 1

        if frame_id % 100 == 0:
            print(f"✅ 已处理 {frame_id} 帧")

    cap.release()
    writer.release()
    print(f"🎉 视频处理完成: {output_path}")


# ======================
# CLI
# ======================

if __name__ == "__main__":
    Path("data/output").mkdir(parents=True, exist_ok=True)
    video_detection(VIDEO_INPUT, VIDEO_OUTPUT)
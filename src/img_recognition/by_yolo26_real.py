import base64
import io
import os
from pathlib import Path
import cv2  # 实时视频处理核心库
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from pydantic import BaseModel
import pandas as pd
import requests
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class DetectResult(BaseModel):
    class_name: str
    confidence: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def cv2_to_pil(cv2_img) -> Image.Image:
    """将 OpenCV 的 BGR 图像转换为 PIL 的 RGB 图像"""
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(pil_img) -> np.ndarray:
    """将 PIL 的 RGB 图像转换为 OpenCV 的 BGR 图像"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def frame_base64(cv2_frame) -> str | None:
    """直接将内存中的视频帧转换为 Base64，无需保存到硬盘"""
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


def request_model(url: str, heads: dict[str, str], base64_url: str) -> str:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "简要介绍一下图片中的动物（20字以内）"},  # 实时显示，字数不宜过多
                    {"type": "image_url", "image_url": {"url": base64_url}},
                ],
            }
        ],
        "cache_prompt": True
    }
    try:
        res = requests.post(url, headers=heads, json=payload, timeout=5)  # 缩短超时时间避免卡死
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"大模型请求失败: {str(e)}"


def render_text_on_frame(cv2_frame, text: str) -> np.ndarray:
    """在视频帧下方或安全区域渲染中文文本"""
    try:
        img = cv2_to_pil(cv2_frame).convert("RGBA")
        img_w, img_h = img.size

        # 自适应字体大小
        font_size = max(16, int(img_w * 0.025))
        font = None
        for font_path in ["msyh.ttc", "simsun.ttc", "SourceHanSansSC-Regular.otf", "Arial Unicode.ttf",
                          "NotoSansCJK-Regular.ttc"]:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except IOError:
                continue
        if font is None:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(img)
        sample_bbox = font.getbbox("测试Text")
        line_height = int((sample_bbox[3] - sample_bbox[1]) * 1.5)

        # 文本自动换行
        lines = []
        for paragraph in text.splitlines():
            if not paragraph.strip():
                continue
            current_line = ""
            for char in paragraph:
                test_line = current_line + char
                bbox = draw.textbbox((0, 0), test_line, font=font)
                if (bbox[2] - bbox[0]) > (img_w * 0.9):
                    lines.append(current_line)
                    current_line = char
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line)

        # 绘制半透明背景遮罩（防止白字在浅色背景下看不清）
        padding = 20
        total_text_height = len(lines) * line_height + padding
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([(0, img_h - total_text_height), (img_w, img_h)], fill=(0, 0, 0, 160))
        img = Image.alpha_composite(img, overlay)

        # 写字
        draw = ImageDraw.Draw(img)
        y_text = img_h - total_text_height + (padding // 2)
        for line in lines:
            draw.text((20, y_text), line, font=font, fill=(255, 255, 255, 255))
            y_text += line_height

        return pil_to_cv2(img.convert("RGB"))
    except Exception as e:
        print(f"❌ 渲染文本失败: {e}")
        return cv2_frame


def process_video(video_source: str | int):
    """
    核心视频处理函数
    :param video_source: 可以是视频文件路径（如 "car_animal.mp4"），也可以是摄像头索引（如 0）
    """
    # 1. 初始化模型与配置
    GLM_5B_Q4_URL = "http://192.168.21.128:8001/v1/chat/completions"
    GLM_Q4_HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer sk-admin-ford-glm-5b-123"}

    YOLO_26 = str(BASE_DIR / "models" / "yolo26n.pt")
    print("正在加载 YOLO 模型...")
    model = YOLO(YOLO_26)

    ANIMAL_CLASSES = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}

    # 2. 打开视频流
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"❌ 无法打开视频源: {video_source}")
        return

    print(f"🎬 成功打开视频源: {video_source}，正在实时检测...（按 'Q' 键退出）")

    frame_count = 0
    model_description = "等待检测动物..."

    # 频率控制：每 30 帧（约1秒）允许请求一次大模型，避免卡顿
    LLM_FRAME_INTERVAL = 30
    last_llm_frame = -LLM_FRAME_INTERVAL

    video_records = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("🏁 视频播放结束或无法获取当前帧。")
                break

            frame_count += 1

            # --- 步骤 1: YOLO 逐帧检测 ---
            # 直接传入 frame 矩阵，不进行磁盘 IO
            results = model.predict(frame, conf=0.1, iou=0.4, imgsz=640, verbose=False)
            result = results[0]

            has_animal = False
            current_frame_animals = []

            if len(result) > 0:
                names_dict = model.names
                # 过滤出动物目标并直接在原帧上画框（利用 YOLO 自带的 plot 功能）
                keep_indices = []
                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0].item())
                    class_name = names_dict[cls_id]

                    if class_name in ANIMAL_CLASSES:
                        has_animal = True
                        keep_indices.append(i)

                        # 记录数据供后续导出 Excel
                        coords = box.xyxy[0].tolist()
                        current_frame_animals.append({
                            "帧号": frame_count,
                            "动物类型": class_name,
                            "置信度": round(float(box.conf[0].item()), 4),
                            "坐标": [round(c, 2) for c in coords]
                        })

                if keep_indices:
                    # 将画好目标框的图像覆盖回 frame
                    frame = result[keep_indices].plot()

            # --- 步骤 2: 条件触发大语言模型 ---
            if has_animal:
                # 满足帧间隔才调用大模型，防止由于大模型延迟导致视频卡顿
                if frame_count - last_llm_frame >= LLM_FRAME_INTERVAL:
                    print(f"⏰ 第 {frame_count} 帧触发大模型分析...")
                    base64_url = frame_base64(frame)
                    if base64_url:
                        # 注意：此处为同步请求，视频会稍微停顿。如果需要极度流畅，需要换成多线程异步请求
                        model_description = request_model(GLM_5B_Q4_URL, GLM_Q4_HEADERS, base64_url)
                        last_llm_frame = frame_count
            else:
                model_description = "未检测到动物"

            # --- 步骤 3: 渲染大模型文本到视频帧 ---
            frame = render_text_on_frame(frame, model_description)

            # --- 步骤 4: 实时显示视频 ---
            cv2.imshow("YOLO + LLM Real-time Detection", frame)

            # 存储历史记录
            if current_frame_animals:
                for animal in current_frame_animals:
                    animal["大模型描述"] = model_description
                    video_records.append(animal)
            else:
                video_records.append({
                    "帧号": frame_count,
                    "动物类型": "无",
                    "置信度": 0.0,
                    "坐标": [],
                    "大模型描述": model_description
                })

            # 按 'q' 键手动退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 用户手动中止视频播放。")
                break

    finally:
        # 5. 释放资源与生成报表
        cap.release()
        cv2.destroyAllWindows()

        if video_records:
            df = pd.DataFrame(video_records)
            excel_path = Path(os.getcwd()) / "视频检测报表.xlsx"
            df.to_excel(excel_path, index=False)
            print(f"📊 视频处理完成。报表已保存至: {excel_path}")


if __name__ == "__main__":
    # 可以传入本地视频文件路径
    video_path = r"C:\Users\pablozhao\Documents\WXWork\1688857975789108\Cache\File\2026-06\AIGC素材_运动模糊模拟不同车速_车外动物\车速测试素材\演示视频\010kph_演示.mp4"

    # 如果想调用电脑自带摄像头实时检测，可以把下面这行取消注释：
    # video_path = 0

    process_video(video_path)
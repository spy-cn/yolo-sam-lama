import base64
import io
import os
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw
from pydantic import BaseModel
import pandas as pd
import requests
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_ROOT_DIR = BASE_DIR / "data/frame_car"


class DetectResult(BaseModel):
    class_name: str
    confidence: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def general_detection(img_path: str, model: YOLO) -> list[DetectResult]:
    """核心检测：传入已经实例化好的 model 避免重复加载"""
    results = model.predict(img_path, conf=0.1, iou=0.4, imgsz=1280, verbose=False)

    result = results[0]
    if len(result) == 0:
        return []

    names_dict = model.names
    ANIMAL_CLASSES = {
        "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe"
    }
    keep_indices = []
    class_max_conf = {}

    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0].item())
        class_name = names_dict[cls_id]

        if class_name not in ANIMAL_CLASSES:
            continue

        conf_val = float(box.conf[0].item())

        if cls_id not in class_max_conf:
            class_max_conf[cls_id] = conf_val
            keep_indices.append(i)
        else:
            if abs(conf_val - class_max_conf[cls_id]) < 1e-5:
                keep_indices.append(i)

    dr_list: list[DetectResult] = []
    for idx in keep_indices:
        box = result.boxes[idx]
        cls_id = int(box.cls[0].item())
        class_name = names_dict[cls_id]
        conf_val = float(box.conf[0].item())

        coords = box.xyxy[0].tolist()
        xmin, ymin, xmax, ymax = [round(c, 2) for c in coords]
        detect_obj = DetectResult(
            class_name=class_name,
            confidence=conf_val,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax
        )
        dr_list.append(detect_obj)

    # 保存画框图片
    if keep_indices:
        filtered_result = result[keep_indices]
        p = Path(img_path)

        # 确保 output 子文件夹存在
        output_dir = p.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = str(output_dir / f"{p.stem}_output{p.suffix}")
        filtered_result.save(filename=save_path)

    return dr_list


def crop_detected_objects(img_path: str, detect_list: list[DetectResult], save_dir: str):
    if not detect_list:
        return

    img = Image.open(img_path)
    img_w, img_h = img.size

    output_path = Path(save_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, detect in enumerate(detect_list):
        xmin = max(0, min(int(detect.xmin), img_w))
        ymin = max(0, min(int(detect.ymin), img_h))
        xmax = max(0, min(int(detect.xmax), img_w))
        ymax = max(0, min(int(detect.ymax), img_h))

        if xmax <= xmin or ymax <= ymin:
            continue

        cropped_img = img.crop((xmin, ymin, xmax, ymax))
        file_name = f"{Path(img_path).stem}_{i}_{detect.class_name}_{detect.confidence:.2f}.jpg"
        save_to = output_path / file_name
        cropped_img.save(save_to)


def img_base64(img_path: str) -> str | None:
    try:
        with Image.open(img_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=55, optimize=True)
            img_bytes = buffer.getvalue()
        b64_data = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_data}"
    except Exception as e:
        print(f"❌ 读取图片 Base64 失败 [{img_path}]: {e}")
        return None


def request_model(url: str, heads: dict[str, str], base64_url: str) -> str:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "介绍一下图片中的动物"},
                    {"type": "image_url", "image_url": {"url": base64_url}},
                ],
            }
        ],
        "cache_prompt": True
    }
    try:
        res = requests.post(url, headers=heads, json=payload, timeout=60)
        res.raise_for_status()
        result = res.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"大模型请求失败: {str(e)}"


def render_text_on_image(target_img_path: str, text: str, output_path: str):
    try:
        img = Image.open(target_img_path).convert("RGBA")
        img_w, img_h = img.size

        font_size = max(16, int(img_w * 0.025))
        font = None
        for font_path in ["msyh.ttc", "simsun.ttc", "SourceHanSansSC-Regular.otf", "Arial Unicode.ttf"]:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except IOError:
                continue
        if font is None:
            font = ImageFont.load_default()

        draw = ImageDraw.Draw(img)
        sample_bbox = font.getbbox("测试Text")
        single_line_height = sample_bbox[3] - sample_bbox[1]
        line_height = int(single_line_height * 1.5)

        lines = []
        raw_paragraphs = text.splitlines()

        for paragraph in raw_paragraphs:
            if not paragraph.strip():
                lines.append("")
                continue
            current_line = ""
            for char in paragraph:
                test_line = current_line + char
                bbox = draw.textbbox((0, 0), test_line, font=font)
                line_w = bbox[2] - bbox[0]
                if line_w > (img_w * 0.9):
                    lines.append(current_line)
                    current_line = char
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line)

        padding = 30
        total_text_height = int(len(lines) * line_height) + padding

        draw = ImageDraw.Draw(img)
        y_text = img_h - total_text_height + (padding // 2)
        for line in lines:
            if line:
                draw.text((20, y_text), line, font=font, fill=(255, 255, 255, 255))
            y_text += line_height

        # 确保输出目录存在后再保存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        final_img = img.convert("RGB")
        final_img.save(output_path)
    except Exception as e:
        print(f"❌ 渲染文本到图片失败: {e}")


def batch_process_folders(root_dir: str | Path):
    """遍历根目录下的子文件夹，对每个文件夹进行批量图片处理，并各自生成 Excel"""
    GLM_5B_Q4_URL = "http://192.168.21.128:8001/v1/chat/completions"
    GLM_Q4_HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer sk-admin-ford-glm-5b-123"}

    YOLO_26 = str(BASE_DIR / "models" / "yolo26n.pt")
    print("正在加载 YOLO 模型...")
    model = YOLO(YOLO_26)

    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"错误: 根目录 {root_path} 不存在！")
        return

    valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    for folder in root_path.iterdir():
        if not folder.is_dir():
            continue

        # 过滤出当前文件夹下的原始图片文件
        img_paths = [
            p for p in folder.iterdir()
            if p.suffix.lower() in valid_extensions
               and "_output" not in p.stem
               and "_final_result" not in p.stem
        ]

        if not img_paths:
            print(f"📁 文件夹 [{folder.name}] 中未发现有效图片，跳过。")
            continue

        print(f"\n📂 开始处理文件夹: {folder.name} (共 {len(img_paths)} 张图片)")

        folder_records = []

        for img_path in img_paths:
            str_img_path = str(img_path)
            print(f"  📸 正在处理: {img_path.name}")

            # 1. YOLO 检测
            detect_list = general_detection(str_img_path, model)
            is_detected = "是" if len(detect_list) > 0 else "否"

            # 2. 裁剪目标
            crop_dir = folder / "crops"
            if detect_list:
                crop_detected_objects(str_img_path, detect_list, save_dir=str(crop_dir))

            # 3. 【核心优化点】：条件触发大语言模型请求
            if detect_list:
                # 只有 YOLO 识别到动物，才进行 Base64 转换并请求大模型
                base64_url = img_base64(str_img_path)
                if base64_url:
                    model_description = request_model(GLM_5B_Q4_URL, GLM_Q4_HEADERS, base64_url)
                else:
                    model_description = "图片 Base64 转换失败，未发起请求"
            else:
                # 未识别到动物，直接赋予默认信息，跳过请求节省开销
                model_description = "未检测到动物，跳过大模型请求"

            # 4. 融合文本渲染
            yolo_output_img = str(folder / "output" / f"{img_path.stem}_output{img_path.suffix}")
            final_render_img = str(folder / "final" / f"{img_path.stem}_final_result{img_path.suffix}")

            # 确定渲染底图（如果有 YOLO 圈框图用圈框图，否则用原图）
            target_render_base = yolo_output_img if Path(yolo_output_img).exists() else str_img_path
            render_text_on_image(target_render_base, model_description, final_render_img)

            # 5. 整合数据记录
            if detect_list:
                for detect in detect_list:
                    coords_str = f"[{detect.xmin}, {detect.ymin}, {detect.xmax}, {detect.ymax}]"
                    folder_records.append({
                        "文件名": img_path.name,
                        "是否识别到动物": is_detected,
                        "动物类型": detect.class_name,
                        "置信度": round(detect.confidence, 4),
                        "动物坐标(xyxy)": coords_str,
                        "大模型文字描述": model_description
                    })
            else:
                folder_records.append({
                    "文件名": img_path.name,
                    "是否识别到动物": is_detected,
                    "动物类型": "无",
                    "置信度": 0.0,
                    "动物坐标(xyxy)": "[]",
                    "大模型文字描述": model_description
                })

        # 6. 生成 Excel 报告
        if folder_records:
            df = pd.DataFrame(folder_records)
            excel_path = folder / f"{folder.name}_识别报表.xlsx"
            df.to_excel(excel_path, index=False)
            print(f"📊 成功为文件夹 [{folder.name}] 生成 Excel: {excel_path}")


if __name__ == "__main__":
    batch_process_folders(INPUT_ROOT_DIR)
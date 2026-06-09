import base64
from pathlib import Path
import io
import requests
from PIL import Image, ImageFont, ImageDraw
from pydantic import BaseModel
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_IMG = str(BASE_DIR / "data/test" / "img.png")


class DetectResult(BaseModel):
    class_name: str
    confidence: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def general_detection(img_path: str) -> list:
    YOLO_26 = str(BASE_DIR / "models" / "yolo26n.pt")
    model = YOLO(YOLO_26)

    results = model.predict(img_path, conf=0.1, iou=0.4, imgsz=1280)

    result = results[0]
    if len(result) == 0:
        print("未检测到任何目标")
        return

    # 获取模型的类别名称映射字典 (例如 {0: 'person', 1: 'bicycle', ...})
    names_dict = model.names
    ANIMAL_CLASSES = {
        "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe"
    }
    keep_indices = []
    class_max_conf = {}

    # 1. 筛选出每一类置信度最高的目标
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0].item())
        class_name = names_dict[cls_id]  # 提前获取类名

        # 【核心修改 1】如果当前识别到的物体不在动物白名单里，直接跳过
        if class_name not in ANIMAL_CLASSES:
            continue

        conf_val = float(box.conf[0].item())

        if cls_id not in class_max_conf:
            class_max_conf[cls_id] = conf_val
            keep_indices.append(i)
        else:
            if abs(conf_val - class_max_conf[cls_id]) < 1e-5:
                keep_indices.append(i)

    # 2. 针对筛选后的目标，打印它们的类别名称和坐标
    print("\n" + "=" * 50)
    print("【每一类置信度最高的目标坐标信息】")
    print("=" * 50)
    dr_list: list[DetectResult] = []
    for idx in keep_indices:
        box = result.boxes[idx]
        cls_id = int(box.cls[0].item())
        class_name = names_dict[cls_id]  # 获取可读的类别名称
        conf_val = float(box.conf[0].item())

        # 获取左上角和右下角的像素坐标 [xmin, ymin, xmax, ymax]
        coords = box.xyxy[0].tolist()
        xmin, ymin, xmax, ymax = [round(c, 2) for c in coords]  # 保留两位小数
        detect_obj = DetectResult(
            class_name=class_name,
            confidence=conf_val,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax
        )
        dr_list.append(detect_obj)
        print(f"类别: {class_name:<12} | 置信度: {conf_val:.4f} | 坐标(xyxy): [{xmin}, {ymin}, {xmax}, {ymax}]")

    print("=" * 50 + "\n")

    # 3. 弹出窗口展示图片
    if keep_indices:
        filtered_result = result[keep_indices]
        filtered_result.show()
        p = Path(img_path)
        save_path = str(p.parent / f"{p.stem}_output{p.suffix}")


        filtered_result.save(filename=save_path)
        print(f"已成功保存画框图片至: {save_path}")
    return dr_list


def plant_detection(img_path: str):
    YOLO_26 = str(BASE_DIR / "models/plant-leaf-detection-and-classification" / "best.pt")
    model = YOLO(YOLO_26)
    results = model.predict(img_path, conf=0.1, iou=0.4, imgsz=1280)
    results[0].show()


def building_detection():
    """建筑物识别"""
    YOLO_SEG_PATH = str(BASE_DIR / "models/keremberke/yolov8m-building-segmentation" / "best.pt")
    model = YOLO(YOLO_SEG_PATH)
    results = model.predict(INPUT_IMG, conf=0.1, iou=0.4, imgsz=1280)
    results[0].show()
    if len(results[0]) > 0:
        # results[0] 内部的数据默认按 conf 从大到小排序，[:1] 表示只取第一个（即最高置信度）
        top1_result = results[0][:1]
        # 打印这唯一一个框的信息
        print(top1_result)
        # 只展示这一个框
        top1_result.show()
    else:
        print("没有检测到任何满足阈值的目标。")


def crop_detected_objects(img_path: str, detect_list: list[DetectResult], save_dir: str = "output/crops"):
    """
    根据 YOLO 检测到的坐标列表，从原图中裁剪出对应的物体并保存。

    :param img_path: 原图路径
    :param detect_list: 包含坐标的 DetectResult 对象列表
    :param save_dir: 裁剪后的图片保存目录，默认为 output/crops
    """
    if not detect_list:
        print("没有检测结果，跳过裁剪。")
        return

    # 1. 打开原图并创建保存目录
    img = Image.open(img_path)
    img_w, img_h = img.size  # 获取原图的实际宽高，用于安全越界检查

    output_path = Path(save_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"开始裁剪图片: {img_path}，共有 {len(detect_list)} 个目标...")

    # 2. 遍历检测结果进行裁剪
    for i, detect in enumerate(detect_list):
        # 核心防错：将坐标强转为整数，并限制在图片实际宽高范围内，防止越界报错
        xmin = max(0, min(int(detect.xmin), img_w))
        ymin = max(0, min(int(detect.ymin), img_h))
        xmax = max(0, min(int(detect.xmax), img_w))
        ymax = max(0, min(int(detect.ymax), img_h))

        # 检查避免裁剪出宽高为 0 的无效图片
        if xmax <= xmin or ymax <= ymin:
            print(f"目标 {i} [{detect.class_name}] 坐标无效，跳过裁剪。")
            continue

        # 3. 执行裁剪 (格式: 左, 上, 右, 下)
        cropped_img = img.crop((xmin, ymin, xmax, ymax))

        # 4. 生成保存文件名 (例如: 0_dog_0.94.jpg)
        file_name = f"{i}_{detect.class_name}_{detect.confidence:.2f}.jpg"
        save_to = output_path / file_name

        # 5. 保存裁剪后的图片
        cropped_img.save(save_to)
        print(f"成功保存裁剪图: {save_to}")

    print("所有目标裁剪完成！\n")

def img_base64(img_path: str) -> str | None:
    """读取图片并将其转换为 base64 编码的 Data URL"""
    try:
        with Image.open(img_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            # 缩放并压缩以提高传输效率
            img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=55, optimize=True)
            img_bytes = buffer.getvalue()
        b64_data = base64.b64encode(img_bytes).decode("utf-8")
        print(b64_data)
        return f"data:image/jpeg;base64,{b64_data}"
    except Exception as e:
        print(f"❌ 读取并重构图片失败 [{img_path}]: {e}")
        return None


def request_model(url: str, heads: dict[str, str], base64_url: str) -> str:
    """构建 payload 并请求大模型，返回文本结果"""
    # 这样模型能直接返回正常的纯文本描述，避免了复杂的 JSON 解析或模型因不兼容而报错
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
        # 设置 timeout=60 防止单个请求死锁导致整个批量任务挂起
        res = requests.post(url, headers=heads, json=payload, timeout=60)
        res.raise_for_status()
        result = res.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"请求失败: {str(e)}"


def render_text_on_image(target_img_path: str, text: str, output_path: str):
    """
    将文本自动换行并渲染到图片的底部，带有一个半透明黑色背景遮罩。
    彻底修复了因原始文本自带换行符(\n)导致的文字重叠Bug。
    """
    try:
        img = Image.open(target_img_path).convert("RGBA")
        img_w, img_h = img.size

        # 1. 动态计算字号
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

        # 2. 精准计算单行文本高度与行距
        sample_bbox = font.getbbox("测试Text-j-g-Q")
        single_line_height = sample_bbox[3] - sample_bbox[1]
        line_height = int(single_line_height * 1.5)  # 保持 1.5 倍的舒适行距

        # 3. 【核心修复】文本自动换行逻辑：先处理文本内部自带的原始换行
        lines = []
        raw_paragraphs = text.splitlines()  # 先把大模型返回的文本按 "\n" 切割成独立段落

        for paragraph in raw_paragraphs:
            if not paragraph.strip():  # 如果是个空行，直接保留一个空位置
                lines.append("")
                continue

            current_line = ""
            for char in paragraph:
                test_line = current_line + char
                bbox = draw.textbbox((0, 0), test_line, font=font)
                line_w = bbox[2] - bbox[0]

                # 如果当前段落长度超过了图片宽度的 90%，触发自动换行
                if line_w > (img_w * 0.9):
                    lines.append(current_line)
                    current_line = char
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line)

        # 4. 重新动态计算文本区域所需的总高度
        padding = 30
        total_text_height = int(len(lines) * line_height) + padding

        # 5. 绘制半透明底层背景框
        # overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        # overlay_draw = ImageDraw.Draw(overlay)
        # overlay_draw.rectangle([(0, img_h - total_text_height), (img_w, img_h)], fill=(0, 0, 0, 160))
        # img = Image.alpha_composite(img, overlay)

        # 6. 把文字写到遮罩上方
        draw = ImageDraw.Draw(img)
        y_text = img_h - total_text_height + (padding // 2)
        for line in lines:
            if line:  # 非空行才绘制
                draw.text((20, y_text), line, font=font, fill=(255, 255, 255, 255))
            y_text += line_height  # 每过一行，无条件平移固定的行高，绝对不会重叠

        # 7. 转回 RGB 并保存
        final_img = img.convert("RGB")
        final_img.save(output_path)
        print(f"🎉 最终大模型描述已完美渲染并保存至: {output_path}")
        final_img.show()

    except Exception as e:
        print(f"❌ 渲染文本到图片失败: {e}")
if __name__ == "__main__":
    GLM_5B_Q4_URL = "http://192.168.21.128:8001/v1/chat/completions"
    GLM_Q4_HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer sk-admin-ford-glm-5b-123"}

    img_path = str(BASE_DIR / "data/test" / "img_3.png")
    base64_url = img_base64(img_path)
    detect_list = general_detection(img_path)
    print(detect_list)
    for detect in detect_list:
        class_name = detect.class_name
        print(class_name)
    crop_detected_objects(img_path,detect_list,save_dir="./")
    model_description = request_model(GLM_5B_Q4_URL, GLM_Q4_HEADERS, base64_url)
    p = Path(img_path)
    yolo_output_img = str(p.parent / f"{p.stem}_output{p.suffix}")  # 这是 YOLO 刚存好的图
    final_render_img = str(p.parent / f"{p.stem}_final_result{p.suffix}")  # 最终融合图

    if Path(yolo_output_img).exists():
        render_text_on_image(yolo_output_img, model_description, final_render_img)
    else:
        # 如果 YOLO 没检测到目标没产生新图，就直接在原图上画大模型的字
        render_text_on_image(img_path, model_description, final_render_img)

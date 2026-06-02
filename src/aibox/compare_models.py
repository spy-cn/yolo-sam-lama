import base64
import io
import os
import time
import pandas as pd
import requests
from PIL import Image

# --- 配置参数 ---
YOUTU_4B_URL = "http://192.168.21.128:8001/v1/chat/completions"
YOUTU_HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer sk-admin-youtu8b123"}
GLM_5B_URL = "http://192.168.21.128:8000/v1/chat/completions"
GLM_HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer sk-admin-ford-glm-5b-123"}

PROMPT_TEXT = "用中文描述一下这张图片"

# 支持的图片格式拓展名
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


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
        return f"data:image/jpeg;base64,{b64_data}"
    except Exception as e:
        print(f"❌ 读取并重构图片失败 [{img_path}]: {e}")
        return None


def request_model(url: str, heads: dict[str, str], base64_url: str) -> str:
    """构建 payload 并请求大模型，返回文本结果"""
    # 💡 优化点：移除了 "response_format": {"type": "json_object"}
    # 这样模型能直接返回正常的纯文本描述，避免了复杂的 JSON 解析或模型因不兼容而报错
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_TEXT},
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


def scan_images_from_dirs(base_dirs: list) -> list:
    """从给定的多个文件夹（支持子文件夹深度遍历）中扫描所有图片"""
    img_files = []
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"⚠️ 警告: 文件夹不存在，已跳过 -> {base_dir}")
            continue

        # os.walk 会自动递归遍历当前文件夹以及它下面的所有子文件夹
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    full_path = os.path.join(root, file)
                    img_files.append(full_path)
    return img_files


def batch_process_multiple_dirs(target_dirs: list, output_excel: str = "multi_dir_results.xlsx"):
    """批量处理多个文件夹内的图片，并将结果保存到 Excel"""

    # 1. 扫描所有图片
    img_files = scan_images_from_dirs(target_dirs)

    if not img_files:
        print("📁 未在指定的文件夹中找到任何有效图片。")
        return

    print(f"🚀 开始处理，在所有文件夹中共找到 {len(img_files)} 张图片...")

    # 用于存放最终 Excel 数据的列表
    data_records = []

    # 定义要跑的模型任务
    models_to_run = [
        {"name": "YOUTU_4B", "url": YOUTU_4B_URL, "headers": YOUTU_HEADERS},
        {"name": "GLM_5B", "url": GLM_5B_URL, "headers": GLM_HEADERS},
    ]

    # 2. 循环处理每一张图片
    for index, img_path in enumerate(img_files, start=1):
        filename = os.path.basename(img_path)
        # 获取图片直接所属的文件夹名字
        folder_name = os.path.basename(os.path.dirname(img_path))

        print(f"\n进度 [{index}/{len(img_files)}] 正在处理: {folder_name} -> {filename}")

        # 转化 base64
        base64_url = img_base64(img_path)
        if not base64_url:
            continue

        # 分别请求不同的模型
        for model in models_to_run:
            print(f" └─ {model['name']} 请求中...")

            start_time = time.time() * 1000
            response_content = request_model(model["url"], model['headers'], base64_url)
            end_time = time.time() * 1000

            elapsed_time = round(end_time - start_time, 2)

            # 💡 优化点：防御性截断，防止极端情况下（如长 HTML 报错）单格字符超限导致 Excel 写入失败
            if len(response_content) > 30000:
                response_content = response_content[:30000] + "\n...[内容过长已被截断]..."

            # 记录数据
            data_records.append({
                "所属文件夹": folder_name,
                "图片名称": filename,
                "测试模型": model["name"],
                "耗时 (ms)": elapsed_time,
                "识别结果": response_content,
                "绝对路径": os.path.abspath(img_path)
            })

    # --- 3. 统一写入 Excel ---
    if not data_records:
        print("❌ 没有产生任何有效数据，取消生成 Excel。")
        return

    print("\n📊 正在生成 Excel 报表...")
    df = pd.DataFrame(data_records)

    # 调整列顺序
    columns_order = ["所属文件夹", "图片名称", "测试模型", "耗时 (ms)", "识别结果", "绝对路径"]
    df = df[columns_order]

    try:
        df.to_excel(output_excel, index=False, engine="openpyxl")
        print(f"✨ 全部处理完成！结果已成功保存至: {os.path.abspath(output_excel)}")
    except Exception as e:
        print(f"❌ 写入 Excel 失败 (请检查文件是否被占用): {e}")


if __name__ == "__main__":
    # 使用 r"" 原始字符串避免 Windows 路径反斜杠转义问题，配置正确
    TARGET_FOLDERS = [
        r"C:\pablo\05_self_code\yolo-sam-lama\data\person",
        r"C:\pablo\05_self_code\yolo-sam-lama\data\trash_cans",
        r"C:\pablo\05_self_code\yolo-sam-lama\data\weather_road",
        r"C:\pablo\05_self_code\yolo-sam-lama\data\wires_road",
    ]

    # 运行批量处理
    batch_process_multiple_dirs(target_dirs=TARGET_FOLDERS, output_excel="multi_folder_outputs.xlsx")
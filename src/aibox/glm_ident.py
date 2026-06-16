
import base64
import io
import os
import time
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

GLM_5B_URL = "http://192.168.21.128:8000/v1/chat/completions"
GLM_HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer sk-admin-ford-glm-5b-123"}

PROMPT_TEXT = "给出图片中路障的检测框的坐标"

def img_base64(img_path: str) -> str | None:
    """读取图片并将其转换为 base64 编码的 Data URL"""
    try:
        with Image.open(img_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            # 缩放并压缩以提高传输效率
            # img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=75, optimize=True)
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


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    # 配置输入图片路径
    INPUT_IMG = str(BASE_DIR / "data/trash_cans" / "conical_barrel_01.jpg")
    img_path = img_base64(INPUT_IMG)
    if img_path:
        result = request_model(GLM_5B_URL, GLM_HEADERS, img_path)
        print(result)
    else:
        print("路径为空")
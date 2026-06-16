import base64
import time
import requests
import json
import io
from PIL import Image

image_path = "../../data/wires_road/wires_road_02.jpeg"
start_time = time.time() * 1000

try:
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=55, optimize=True)
        img_bytes = buffer.getvalue()

    b64_data = base64.b64encode(img_bytes).decode("utf-8")
    formatted_url = f"data:image/jpeg;base64,{b64_data}"

except Exception as e:
    print(f"❌ 读取并重构图片失败: {e}")
    exit()

prompt_text = (
    "你是一名车载视觉助手，负责分析前视摄像头画面中的路况天气等级。"
    "判断当前天气路况等级："
    "- 雨：L1(小雨/潮湿) / L2(中雨/积水) / L3(大雨/严重积水)"
    "- 雪：L1(轻雪) / L2(中雪) / L3(大雪)"
    "- 雾：L1(轻雾) / L2(中雾) / L3(大雾)"
    "- 若多种天气并存，以最高等级输出"
    "输出格式（严格遵循）："
    "{\"weather_type\": \"雨/雪/雾/无\", \"level\": \"L1/L2/L3\",\"road_condition\":\"泥泞路段/积水路段/正常路段\", \"confidence\": 0-1}"
)

payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Locate all the instances that matches the following description: wires,electric wire."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": formatted_url
                    }
                }
            ]
        }
    ],
    "cache_prompt": True,
    "response_format": {"type": "json_object"}
}

url = "http://192.168.21.128:8003/v1/chat/completions"
headers  = {"Content-Type": "application/json", "Authorization": "Bearer sk-admin-youtu8b123"}

try:
    res = requests.post(url, headers=headers, json=payload, timeout=60)
    result = res.json()

    if "error" in result:
        print("\n服务端返回了错误:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("\n成功！模型回复如下:")
        print(result["choices"][0]["message"]["content"])

    end_time = time.time() * 1000
    print("共花费:", end_time - start_time, "ms")
except Exception as e:
    print(f"请求发生异常: {e}")
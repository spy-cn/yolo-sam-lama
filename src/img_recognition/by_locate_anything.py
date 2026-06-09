import base64
import time
import requests
import json
import io
import re
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


class ImageDetectionError(Exception):
    """自定义异常类，用于处理图像检测和可视化相关错误"""
    pass


def process_image_for_detection(
        image_path: str,
        max_size: Tuple[int, int] = (1024, 1024),  # 增加尺寸以提高可视化清晰度
        quality: int = 80  # 增加质量
) -> Tuple[str, Image.Image]:
    """
    处理图像并转换为base64格式，同时返回PIL Image对象用于后续绘制

    Args:
        image_path: 图像文件路径
        max_size: 最大尺寸限制
        quality: JPEG压缩质量

    Returns:
        (base64编码的图像字符串, PIL Image对象)

    Raises:
        ImageDetectionError: 图像处理失败时抛出
    """
    try:
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        img = Image.open(image_path)
        # 记录原始尺寸用于后续坐标转换
        original_size = img.size

        # 转换颜色模式
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 调整尺寸用于API调用（可以比可视化的尺寸小）
        api_img = img.copy()
        api_img.thumbnail((512, 512), Image.Resampling.LANCZOS)

        # 保存到内存缓冲区用于base64编码
        buffer = io.BytesIO()
        api_img.save(buffer, format="JPEG", quality=55, optimize=True)
        img_bytes = buffer.getvalue()
        b64_data = base64.b64encode(img_bytes).decode("utf-8")
        formatted_url = f"data:image/jpeg;base64,{b64_data}"

        # 调整尺寸用于可视化展示
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

        return formatted_url, img

    except Exception as e:
        raise ImageDetectionError(f"图像处理失败: {str(e)}")


def detect_objects_in_image(
        image_path: str,
        prompt: str = "Locate: car, person, house",
        api_url: str = "http://192.168.21.128:8003/v1/chat/completions",
        api_key: str = "sk-admin-youtu8b123",
        timeout: int = 60,
        max_retries: int = 3
) -> Tuple[Dict[str, Any], Image.Image]:
    """
    调用API检测图像中的目标物体

    Returns:
        (API响应结果字典, 用于可视化的PIL Image对象)
    """
    try:
        # 处理图像，获取base64和用于可视化的Image对象
        formatted_url, visualization_img = process_image_for_detection(image_path)

        # 构建请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # 构建请求体 - 移除了强制 json_object 的限制
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": formatted_url}}
                    ]
                }
            ],
            "cache_prompt": True
        }

        # 发送请求（带重试机制）
        last_exception = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                response.raise_for_status()
                result = response.json()
                return result, visualization_img

            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⚠️ 请求失败，{wait_time}秒后重试... ({attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                continue

        raise ImageDetectionError(f"请求失败（已重试{max_retries}次）: {str(last_exception)}")

    except Exception as e:
        raise ImageDetectionError(f"检测过程发生异常: {str(e)}")


def extract_detection_results(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从API响应中提取并解析检测结果
    """
    detections = []
    try:
        if "choices" not in response or len(response["choices"]) == 0:
            return detections

        content = response["choices"][0]["message"]["content"]
        if not content or not isinstance(content, str):
            return detections

        # 匹配 <ref>标签</ref><box><ymin><xmin><ymax><xmax></box>
        pattern = r"<ref>(.*?)</ref>\s*<box><(\d+)><(\d+)><(\d+)><(\d+)></box>"
        matches = re.findall(pattern, content)

        for match in matches:
            label, ymin, xmin, ymax, xmax = match
            detections.append({
                "label": label,
                # 坐标为 [ymin, xmin, ymax, xmax]，归一化为 0-1000
                "box_2d": [int(ymin), int(xmin), int(ymax), int(xmax)]
            })

    except Exception as e:
        print(f"⚠️ 结果解析失败: {e}")

    return detections


def visualize_detections(image: Image.Image, detections: List[Dict[str, Any]], output_path: str = "output.jpg"):
    """
    在图片上绘制检测框和标签并保存

    Args:
        image: PIL Image对象
        detections: 解析后的检测结果列表
        output_path: 输出图片路径
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # 尝试加载字体，如果失败则使用默认字体
    try:
        # 需根据系统调整字体路径，例如 Windows: "arial.ttf", Linux: "/usr/share/fonts/..."
        font = ImageFont.truetype("arial.ttf", size=max(15, int(height / 40)))
    except IOError:
        font = ImageFont.load_default()
        print("⚠️ 未找到指定字体，使用默认字体。")

    # 定义颜色列表用于不同的检测框
    colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]

    for i, det in enumerate(detections):
        label = det["label"]
        ymin_norm, xmin_norm, ymax_norm, xmax_norm = det["box_2d"]

        # 将归一化坐标 (0-1000) 转换为实际像素坐标
        left = xmin_norm * width / 1000
        top = ymin_norm * height / 1000
        right = xmax_norm * width / 1000
        bottom = ymax_norm * height / 1000

        color = colors[i % len(colors)]

        # 绘制检测框，增加线条宽度
        line_width = max(3, int(width / 300))
        draw.rectangle([left, top, right, bottom], outline=color, width=line_width)

        # 绘制标签背景和文本
        # 获取文本大小以绘制背景矩形
        try:
            # Pillow >= 9.2.0
            text_bbox = draw.textbbox((left, top), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            # 旧版本 Pillow
            text_width, text_height = draw.textsize(label, font=font)

        # 确保标签在图片范围内
        text_bottom = top if top - text_height > 0 else top + text_height + line_width

        # 绘制文本背景矩形
        draw.rectangle([left, top - text_height - line_width, left + text_width + line_width * 2, top], fill=color)
        # 绘制文本，使用白色
        draw.text((left + line_width, top - text_height - line_width), label, fill="white", font=font)

    # 保存图片
    image.save(output_path, quality=95)
    print(f"\n✅ 可视化结果已保存至: {output_path}")
    # 在受支持的环境中尝试显示图片（例如 Jupyter Notebook 或本地图片查看器）
    try:
        image.show()
    except Exception:
        print("无法自动打开图片预览。")


# 使用示例
if __name__ == "__main__":
    # 请替换为你的实际图片路径
    image_path = "../../data/trash_cans/conical_barrel_01.jpg"
    output_visualization_path = "detected_wires.jpg"

    try:
        print(f"正在检测图片: {image_path} ...")
        # 执行检测，获取结果和用于可视化的图片
        result, viz_img = detect_objects_in_image(image_path)

        # 提取结果
        detections = extract_detection_results(result)

        print("\n✅ API 调用成功！")

        # 显示耗时
        if "timings" in result:
            total_time = result["timings"].get("prompt_ms", 0) + result["timings"].get("predicted_ms", 0)
            print(f"耗时: {total_time:.2f} ms")

        if detections:
            print(f"\n📊 检测到 {len(detections)} 个目标:")
            print(json.dumps(detections, indent=2, ensure_ascii=False))

            # 执行可视化绘制
            visualize_detections(viz_img, detections, output_visualization_path)
        else:
            print("\n⚠️ 未检测到有效目标。")

    except ImageDetectionError as e:
        print(f"\n❌ 发生错误: {e}")
    except Exception as e:
        print(f"\n💥 未知错误: {e}")
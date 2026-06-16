import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

model_path = r"C:\pablo\05_self_code\yolo-sam-lama\models\glm\ZhipuAI\glm-edge-v-5b"
image_path = r"C:\pablo\05_self_code\yolo-sam-lama\data\person\person_road_01.jpeg"

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# 纯 CPU 运行：并且使用 float32 精度
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float32, # CPU 运行必须用 float32 保证算子对齐
    trust_remote_code=True,
    device_map="cpu"
)

image = Image.open(image_path).convert("RGB")

messages = [{"role": "user", "content": "<image>\n请描述这张图片"}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

# CPU 上图片张量保持 float32
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
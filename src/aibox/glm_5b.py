import torch
from transformers import AutoTokenizer, AutoImageProcessor, AutoModelForCausalLM, AutoProcessor
from PIL import Image

# 修改为你的 2b 或者是 5b 路径
model_path = r"C:\pablo\05_self_code\yolo-sam-lama\models\glm\ZhipuAI\glm-edge-v-2b"
image_path = r"C:\pablo\05_self_code\yolo-sam-lama\data\person\person_road_01.jpeg" # 替换为你的本地图片路径

print("正在加载 Processor 和 Tokenizer...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("正在加载模型（CPU 纯净 float32 模式，请耐心等待）...")
# 必须完全去掉 bitsandbytes、float16 和 device_map="auto"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,  # 只有 float32 才能在 CPU 上保证算子维度绝对对齐
    trust_remote_code=True
).cpu()                         # 显式指定在 CPU 上

# 准备图片
image = Image.open(image_path).convert("RGB")

# 官方标准格式
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "请描述这张图片"}
        ]
    }
]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# 特征提取
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,              # 直接在内部完成 token 化
    return_tensors="pt",
    return_dict=True
)
inputs_text = tokenizer(prompt, return_tensors="pt")
inputs.update(inputs_text)

# 确保所有输入张量都在 CPU 上，且图片为 float32

inputs = {k: v for k, v in inputs.items()}
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

inputs.pop("aspect_ratio_ids", None)
inputs.pop("aspect_ratio_mask", None)
inputs.pop("num_tiles", None)
print("开始生成推理（CPU 推理较慢，大约需要1-2分钟）...")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100)

# 解码
response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n🔥 模型回复：")
print(response)
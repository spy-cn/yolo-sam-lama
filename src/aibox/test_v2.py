import os
import urllib.request
import torch
import torch.nn as nn
import clip
from PIL import Image
import cv2


# 1. 定义 LAION V2 官方的多层感知机 (MLP) 结构
class AestheticPredictorV2(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.input_dim = input_dim
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),  # 倒数第二层输出是 16 维，对应报错中的 [16, 64]
            nn.Linear(16, 1)    # 最后一层是第 7 层 (layers.7)，输出最终的 1 维分数
        )

    def forward(self, x):
        return self.layers(x)
# 别名防御，防止部分旧版 PyTorch 无法识别 HyperbolicTangent
# nn.HyperbolicTangent = nn.Tanh
#
# class AestheticPredictorV2(nn.Module):
#     def __init__(self, input_dim=768):
#         super().__init__()
#         self.input_dim = input_dim
#         # 严格根据报错的维度特征 [128, 1024], [64, 128], [16, 64] 还原的结构
#         self.layers = nn.Sequential(
#             nn.Linear(self.input_dim, 1024),   # layers.0
#             nn.HyperbolicTangent(),            # layers.1
#             nn.Dropout(0.2),                   # layers.2
#             nn.Linear(1024, 128),              # layers.3 (根据报错修正为128)
#             nn.HyperbolicTangent(),            # layers.4
#             nn.Dropout(0.2),                   # layers.5
#             nn.Linear(128, 64),                # layers.6 (根据报错修正为64)
#             nn.HyperbolicTangent(),            # layers.7
#             nn.Dropout(0.1),                   # layers.8
#             nn.Linear(64, 16),                 # layers.9 (根据报错修正为16)
#             nn.HyperbolicTangent(),            # layers.10
#             nn.Linear(16, 1)                   # layers.11
#         )
#
#     def forward(self, x):
#         return self.layers(x)

# 2. 检查并自动下载 V2 的权重文件 (.pth) 到本地
WEIGHT_FILE = "ava+logos-l14-linearMSE.pth"
#WEIGHT_FILE = "ava+logos-l14-reluMSE.pth"
if not os.path.exists(WEIGHT_FILE):
    print("⏳ 正在下载 LAION 美学 V2 权重文件 (几MB大小)...")
    url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/ava+logos-l14-linearMSE.pth"
    urllib.request.urlretrieve(url, WEIGHT_FILE)
    print("✅ 权重下载完成！")

# 3. 全局初始化设备与模型
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载基础的大模型 CLIP (ViT-L/14 是官方指定的特征提取器)
print("正在载入 OpenAI CLIP ViT-L/14 (约1.7GB，初次需等待)...")
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)

# 载入轻量级美学打分头部
model_aesthetic = AestheticPredictorV2(768).to(device)
model_aesthetic.load_state_dict(torch.load(WEIGHT_FILE, map_location=device))
model_aesthetic.eval()


def get_laion_v2_score(cv2_frame):
    """
    传入本地 OpenCV 读取的单帧图像，输出 LAION V2 美学打分
    """
    # 转为 PIL 并经过 CLIP 官方的标准化预处理
    rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    image_tensor = clip_preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # 提取图像特征并进行归一化 (L2 Normalize)
        image_features = clip_model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # 喂入美学层预测
        prediction = model_aesthetic(image_features.float())

    return prediction.item()


if __name__ == "__main__":
    frame = cv2.imread(r"C:\pablo\05_self_code\yolo-sam-lama\data\wires_road\wires_road_02.jpeg")
    print("美学得分:", get_laion_v2_score(frame))

    frame = cv2.imread(r"C:\pablo\05_self_code\yolo-sam-lama\data\wires_road\wires_road_03.jpg")
    print("美学得分:", get_laion_v2_score(frame))

    frame = cv2.imread(r"C:\pablo\05_self_code\yolo-sam-lama\data\weather_road\68.jpg")
    print("美学得分:", get_laion_v2_score(frame))
    frame = cv2.imread(r"C:\pablo\05_self_code\yolo-sam-lama\data\beauty\beauty_01.jpg")
    print("美学得分:", get_laion_v2_score(frame))

    frame = cv2.imread(r"C:\pablo\05_self_code\yolo-sam-lama\data\beauty\beauty_02.png")
    print("美学得分:", get_laion_v2_score(frame))
    frame = cv2.imread(r"C:\pablo\05_self_code\yolo-sam-lama\data\beauty\beauty_03.png")
    print("美学得分:", get_laion_v2_score(frame))
    frame = cv2.imread(r"C:\pablo\05_self_code\yolo-sam-lama\data\beauty\beauty_04.png")
    print("美学得分:", get_laion_v2_score(frame))
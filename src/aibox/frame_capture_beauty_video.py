import cv2
import os
import numpy as np
import subprocess
import json

import torch
import torch.nn as nn
import clip
from PIL import Image

# 视频路径与输出目录
video_path = r"C:\Users\pablozhao\Desktop\Tubedown Download\【4K行车记录】琼库什台后山 → 特克斯八卦城 - 新疆最美盘山公路自驾 90公里风景线（上部）-[ekTV1Y9lNcQ]-[1280x720].mp4"
output_dir = "frames_4"
os.makedirs(output_dir, exist_ok=True)


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
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

WEIGHT_FILE = "ava+logos-l14-linearMSE.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("正在载入 OpenAI CLIP ViT-L/14 ...")
clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
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


def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    resolution_ratio = f"{width}*{height}"

    info = {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "resolution_ratio": resolution_ratio,
    }
    if info["fps"] > 0:
        info["duration"] = round(info["frame_count"] / info["fps"], 2)
    cap.release()
    return info


# 每隔指定秒数抽一帧
def frame_capture_second(interval_sec: float):
    # 打印视频基础信息
    try:
        info = get_video_info(video_path)
        print(f"--- 视频加载成功 ---")
        print(
            f"分辨率: {info['resolution_ratio']} | FPS: {info['fps']} | 总帧数: {info['frame_count']} | 总时长: {info['duration']}秒")
    except Exception as e:
        print(f"视频加载失败: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    fps = info['fps']
    frame_interval = int(fps * interval_sec)

    frame_id = 0
    save_id = 0
    last_saved_frame = None

    print(f"开始抽样，每 {interval_sec} 秒（即每 {frame_interval} 帧）评估一次...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 到达抽样步长
        if frame_id % frame_interval == 0:
            print(f" [评估检测] 正在检查视频第 {frame_id} 帧 (约 {round(frame_id / fps, 1)} 秒处):")

            # 进行高质量判定
            if is_high_quality(frame, prev_frame=last_saved_frame):
                frame_name = f"frame_{save_id:06d}.jpg"
                save_path = os.path.join(output_dir, frame_name)
                cv2.imwrite(save_path, frame)
                print(f"       [成功捕获] 高质量帧已保存至 -> {save_path}")
                save_id += 1
                last_saved_frame = frame.copy()
            print("-" * 50)

        # 每 100 帧打印一次进度，防止长时间没输出以为卡死
        if frame_id % 100 == 0 and frame_id > 0:
            print(f"[系统进度] 已扫描至第 {frame_id}/{info['frame_count']} 帧...")

        frame_id += 1

    cap.release()
    print(f"\n==========================================")
    print(f"  任务完成！总共扫描了 {frame_id} 帧，最终截取高质量帧: {save_id} 张。")
    print(f"==========================================")


# 判断是否为高质量帧（带详细日志原因输出）
def is_high_quality(frame, prev_frame=None):
    # 1. 检查清晰度
    sharp = frame_sharpness(frame)
    if sharp < 120:
        print(f"       [未通过] 原因: 清晰度过低 (当前值: {sharp:.2f} < 阈值: 120) -> 画面可能模糊/抖动")
        return False

    # 2. 检查亮度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    if brightness < 40 or brightness > 220:
        print(f"       [未通过] 原因: 曝光异常 (当前亮度: {brightness:.2f}, 正常范围: 40~220)")
        return False

    # 3. 画面重复度过滤（引入 ROI 裁剪与高阈值）
    if prev_frame is not None:
        diff = frame_diff_roi(prev_frame, frame)
        # 工业经验值：剔除车头后，若中间动态风景差异 MSE < 1500，说明车辆可能在等红灯、堵车、或处于极度相似的直道远景中
        if diff < 1500:
            print(f"       [未通过] 与上一帧太相似 (ROI帧差 MSE: {diff:.2f} < 阈值: 1500)")
            return False
        else:
            print(f"       [检查通过] 画面有足够物理位移 (ROI帧差 MSE: {diff:.2f})")

    aesthetic_score = get_laion_v2_score(frame)
    print(f"       [美学得分]  : {aesthetic_score:.3f}")

    print(f"       [通过验证] (清晰度: {sharp:.2f} | 亮度: {brightness:.2f})")
    return True


def frame_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def frame_diff(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return np.mean((g1.astype(float) - g2.astype(float)) ** 2)


def frame_diff_roi(img1, img2):
    """
    针对行车记录仪优化的帧差计算：
    裁剪掉顶部的天空(20%)和底部的车头(30%)，只比对中间 20% 到 70% 之间的区域
    """
    h, w, _ = img1.shape
    start_y = int(h * 0.2)  # 纵坐标 20% 开始
    end_y = int(h * 0.7)  # 纵坐标 70% 结束

    # 裁剪 ROI 区域
    roi1 = img1[start_y:end_y, :]
    roi2 = img2[start_y:end_y, :]

    g1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

    return np.mean((g1.astype(float) - g2.astype(float)) ** 2)

if __name__ == "__main__":
    frame_capture_second(5)


"""
整体评分大约在5.5分左右
1.0 - 3.0 分：废片 / 严重缺陷 (Poor / Defective)
3.0 - 4.5 分：不及格 / 普通快照 (Below Average / Amateur Snapshot)
4.5 - 6.0 分：及格 / 良好 (Average / Good)
6.0 - 7.5 分：优秀 / 专业级 (Very Good / Professional)
7.5 - 9.0 分：卓越 / 艺术级 (Excellent / Artistic)
9.0 - 10.0 分：大师级 / 完美 (Masterpiece / Perfect)
"""
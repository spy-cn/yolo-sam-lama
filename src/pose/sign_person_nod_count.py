import logging
import time
import cv2
import numpy as np
from ultralytics import YOLO

# ----------------- 日志配置 (Logging) -----------------
# 配置标准日志输出格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NodDetector")

# 1. 加载模型（使用 pose 模型）
model = YOLO("../../models/yolov8n-pose.pt")

cap = cv2.VideoCapture(0)  # 0 开启本地摄像头

# ----------------- 超参数配置 -----------------
WINDOW_SIZE = 15  # 滤波滑动窗口大小
THRESHOLD_RATIO = 0.08  # 自适应阈值系数（低头幅度超过头宽的8%）
TIME_WINDOW = 5.0  # 检测的时间窗口：5 秒
MAX_NODS_ALLOWED = 3  # 5秒内允许的最大点头次数，超过则警告

# ----------------- 核心数据结构 -----------------
user_data = {}


def send_to_warning(track_id):
    print(f"{track_id},起来，别睡觉!")


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    current_time = time.time()

    # 2. 开启 persist=True 启动内置追踪器 (ByteTrack)
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

    if results[0].keypoints is not None and results[0].boxes.id is not None:
        # 获取当前帧所有检测到的人的：关键点、Bounding Box 和 Track ID
        keypoints_list = results[0].keypoints.xy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for kpts, track_id in zip(keypoints_list, track_ids):
            # 确保关键点完整（17个点）且检测到了鼻子(0)
            if len(kpts) <= 0 or kpts[0][1] == 0:
                continue

            # 初始化新出现的目标
            if track_id not in user_data:
                user_data[track_id] = {
                    "history_y": [],
                    "is_nodding": False,
                    "nod_timestamps": [],
                }
                # 【日志 1】新目标检测通知
                logger.info(f"检测到新目标进入画面 -> ID: {track_id}")

            person = user_data[track_id]

            # 3. 动态自适应阈值计算（利用双耳间距作为人头宽度的尺度基准）
            left_ear_x = kpts[3][0]
            right_ear_x = kpts[4][0]
            head_width = abs(left_ear_x - right_ear_x)

            if head_width == 0:
                head_width = 100

            dynamic_threshold = head_width * THRESHOLD_RATIO
            nose_y = kpts[0][1]

            # 4. 点头状态机判定
            person["history_y"].append(nose_y)
            if len(person["history_y"]) > WINDOW_SIZE:
                person["history_y"].pop(0)

            if len(person["history_y"]) == WINDOW_SIZE:
                smooth_y = np.mean(person["history_y"])
                base_y = person["history_y"][0]
                diff = smooth_y - base_y

                # 低头判定
                if diff > dynamic_threshold and not person["is_nodding"]:
                    person["is_nodding"] = True
                # 抬头复位，触发一次完整点头
                elif diff < (dynamic_threshold / 2) and person["is_nodding"]:
                    person["is_nodding"] = False

                    # 记录触发时间
                    person["nod_timestamps"].append(current_time)

                    # 清洗窗口防止单次动作反复触发
                    person["history_y"] = person["history_y"][int(WINDOW_SIZE / 2):]

                    # 更新当前窗口内的有效计数，用于日志打印
                    temp_count = sum(1 for t in person["nod_timestamps"] if current_time - t <= TIME_WINDOW)
                    # 【日志 2】成功捕获单次点头
                    logger.info(f"目标 ID:{track_id} 发生一次点头动作 | 近5秒累计: {temp_count}次")

            # 5. 核心逻辑：5秒时间窗口滑窗与警告判定
            # 移除 5 秒之前的所有老时间戳
            person["nod_timestamps"] = [
                t for t in person["nod_timestamps"] if current_time - t <= TIME_WINDOW
            ]

            recent_nod_count = len(person["nod_timestamps"])

            # 6. 画面绘制与报警
            # 获取该目标框的坐标用于画字
            bbox = results[0].boxes[results[0].boxes.id == track_id].xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, bbox[:4])

            # 显示当前人在 5s 内的点头次数
            cv2.putText(
                frame,
                f"ID:{track_id} Nods(5s): {recent_nod_count}",
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # 触发频繁点头警告
            if recent_nod_count >= MAX_NODS_ALLOWED:
                # 【日志 3】触发频繁点头触发警告级别日志
                logger.warning(f"⚠️ 频繁点头警告!! 目标 ID:{track_id} 在 5 秒内连续点头 {recent_nod_count} 次！")

                # 绘制显眼的红色警告框和文字
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    frame,
                    "WARNING: Frequent Nodding!",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                send_to_warning(track_id)

    # 渲染 YOLO 骨骼线条
    annotated_frame = results[0].plot() if len(results) > 0 else frame

    cv2.imshow("Multi-Person Nod Tracking & Alarm", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
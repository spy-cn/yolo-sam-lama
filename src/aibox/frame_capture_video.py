import cv2
import os

video_path = r"C:\pablo\05_self_code\yolo-sam-lama\data\video\高速行车记录仪视频.mp4"
output_dir = "frames_2"
os.makedirs(output_dir, exist_ok=True)


def frame_capture_second(interval_sec: float):
    """
    每隔 interval_sec 抽取一帧
    :param interval_sec: 间隔秒数
    :return:
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS:", fps)

    frame_interval = int(fps * interval_sec)

    frame_id = 0
    save_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            frame_name = f"frame_{save_id:06d}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            save_id += 1

        frame_id += 1

    cap.release()
    print("抽帧完成")


def frame_capture_fps(target_fps: int):
    """
    按照固定帧进行抽取
    :param target_fps: 固定帧大小
    :return:
    """
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(original_fps / target_fps)

    frame_id = 0
    save_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % step == 0:
            cv2.imwrite(f"{output_dir}/frame_{save_id:06d}.jpg", frame)
            save_id += 1

        frame_id += 1

    cap.release()


if __name__ == "__main__":
    # 每5帧抽一帧
    target_fps = 5
    frame_capture_fps(target_fps)

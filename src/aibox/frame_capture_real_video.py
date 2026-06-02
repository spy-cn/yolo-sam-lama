from pathlib import Path
import cv2
import os

PROJECT_ROOT = Path(__file__).parent.parent.parent

output_dir = PROJECT_ROOT / "data/video_frames"
os.makedirs(output_dir, exist_ok=True)


def frame_capture_fps(cap, target_fps: int):
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print("Camera FPS:", original_fps)

    step = max(1, int(original_fps / target_fps))

    frame_id = 0
    save_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        cv2.imshow("Camera", frame)

        if frame_id % step == 0:
            frame_name = f"frame_{save_id:06d}.jpg"
            cv2.imwrite(str(output_dir / frame_name), frame)
            print(f"保存: {frame_name}")
            save_id += 1

        frame_id += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":

    # 连接视频源
    # cap = cv2.VideoCapture(
    #     "rtsp://admin:12345@192.168.1.64:554/stream1",
    #     cv2.CAP_FFMPEG
    # )
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    try:
        frame_capture_fps(cap, target_fps=5)
    finally:
        cap.release()
        cv2.destroyAllWindows()
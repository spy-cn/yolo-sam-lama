from pathlib import Path
import cv2
import numpy as np
import os

PROJECT_ROOT = Path(__file__).parent.parent.parent
output_dir = PROJECT_ROOT / "data/video_frames"
os.makedirs(output_dir, exist_ok=True)


def frame_sharpness(frame):
    """
    :param frame:
    :return: 返回值越大，图像越清晰；越小，越模糊
    """
    # 灰度图（清晰度只关心结构变化、去除颜色干扰、灰度图更快、更稳定）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Laplacian 找图像中变化剧烈的地方，模糊图像 变化少，响应大，清晰图像，变化多，响应大
    # var 方差，图像清晰度的量化指标
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def is_high_quality(frame, prev_frame=None):
    """
    选取高质量帧
    :param frame: 当前帧
    :param prev_frame: 上一帧
    :return:
    """
    sharp = frame_sharpness(frame)
    #清晰度评分小于120分舍弃(太模糊)
    if sharp < 120:
        return False

    #计算整张图的亮度 太黑、太爆
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    if brightness < 40 or brightness > 220:
        return False

    if prev_frame is not None:
        # 和上一帧的区别
        diff = frame_diff(prev_frame, frame)
        if diff < 300:
            return False

    return True


def frame_diff(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return np.mean((g1.astype(float) - g2.astype(float)) ** 2)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    prev_frame = None
    save_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if is_high_quality(frame, prev_frame):
            name = output_dir / f"frame_{save_id:06d}.jpg"
            cv2.imwrite(str(name), frame)
            save_id += 1
            print(f"✅ 保存高质量帧: {name}")

        prev_frame = frame.copy()

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
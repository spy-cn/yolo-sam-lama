from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent.parent.parent
left_path = str(BASE_DIR / "data/fish_eye/v1" / "1.jpg")
right_path = str(BASE_DIR / "data/fish_eye/v1" / "2.jpg")
out_path = str(BASE_DIR / "data/fish_eye/v1" / "12_out.jpg")


left_img = cv2.imread(left_path)
right_img = cv2.imread(right_path)

stitcher = cv2.Stitcher_create()

status, pano = stitcher.stitch([left_img, right_img])

if status == cv2.Stitcher_OK:
    cv2.imwrite(out_path, pano)
    print("拼接成功")
else:
    print("失败:", status)

from pathlib import Path

import cv2
from PIL import Image


def merge_4_images(img_paths, output_path):
    images = [Image.open(p).convert("RGB") for p in img_paths]

    # 统一尺寸（可选）
    w, h = images[0].size
    images = [img.resize((w, h)) for img in images]

    # 创建大画布
    canvas_w = w * 2
    canvas_h = h * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h))

    # 粘贴位置
    positions = [
        (0, 0), (w, 0),
        (0, h), (w, h)
    ]

    for img, pos in zip(images, positions):
        canvas.paste(img, pos)

    canvas.save(output_path)



def create_360_panorama(image_paths):
    # 1. 读取四张图片
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)

    if len(images) < 4:
        print("请确保有四张有效图片")
        return

    # 2. 创建 Stitcher 对象
    # cv2.Stitcher.create(flags)
    # flags 可以选择 cv2.Stitcher_PANORAMA (默认) 或 cv2.Stitcher_SCANS
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)

    # 3. 执行拼接
    # OpenCV 的高级 API 会自动处理特征点提取、匹配和融合
    status, panorama = stitcher.stitch(images)

    # 4. 检查结果
    if status == cv2.Stitcher_OK:
        # 保存结果
        cv2.imwrite("360_panorama.jpg", panorama)
        print("360全景图合成成功！")
        # 显示结果（按任意键关闭窗口）
        cv2.imshow("Panorama", panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"拼接失败，错误代码: {status}")
        # 常见错误代码：
        # cv2.Stitcher_ERR_NEED_MORE_IMGS (1) - 图片太少或重叠区域不足
        # cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL (2) - 单应性矩阵估计失败（通常是镜头畸变没矫正或图片没对准）
        # cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL (3) - 相机参数调整失败





BASE_DIR = Path(__file__).resolve().parent.parent.parent

img_1 = str(BASE_DIR / "data/fish_eye" / "front.png")
img_2 = str(BASE_DIR / "data/fish_eye" / "back.png")
img_3 = str(BASE_DIR / "data/fish_eye" / "left.png")
img_4 = str(BASE_DIR / "data/fish_eye" / "right.png")

OUT_DIR = str(BASE_DIR / "data/out_img/merge_result.jpg")
image_files = [img_1, img_2, img_3, img_4]
# merge_4_images(
#     image_files,
#     OUT_DIR
# )
# 使用你的四张图片路径

create_360_panorama(image_files)
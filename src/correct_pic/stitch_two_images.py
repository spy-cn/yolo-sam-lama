from pathlib import Path

from skimage.transform import warp
from stitching import Stitcher
import cv2

# 更可靠的基础路径
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
img_left = str(ROOT_DIR / "data/to_join_img/4_IMG_8155.JPG")
img_right = str(ROOT_DIR / "data/to_join_img/4_IMG_8156.JPG")

# 检查文件存在
for path in [img_left, img_right]:
    if not Path(path).exists():
        raise FileNotFoundError(f"图片不存在: {path}")

# 根据你的库版本选用正确的参数名，示例用 warper_type
stitcher = Stitcher(warper_type="spherical")   # 或 "spherical" 等

panorama = stitcher.stitch([img_left, img_right])

if panorama is not None:
    cv2.imwrite("panorama_output.jpg", panorama)
    print("拼接完成")
else:
    print("拼接失败，请检查图像间是否有足够重叠且纹理丰富。")
from pathlib import Path

import cv2
import numpy as np
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 配置输入图片路径
INPUT_IMG = str(BASE_DIR / "data/wires_road"/ "wires_road_02.jpeg")
# 1. 读取无法被 YOLO 识别的电线图片
image = cv2.imread(INPUT_IMG)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. 创建 LSD 检测器 (OpenCV ximgproc 模块或内置，不同版本调用略有不同)
# 如果 cv2.createLineSegmentDetector() 报错，说明需要安装 opencv-contrib-python
lsd = cv2.createLineSegmentDetector(0)

# 3. 检测直线段
# lines 返回的是一个数组，每个元素包含 [x1, y1, x2, y2] 一条线段的起点和终点
lines = lsd.detect(gray)[0]

# 4. 过滤并绘制电线
output = image.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # 【工业过滤逻辑】：电线通常比较长
        # 计算线段长度，过滤掉树枝、杂质引起的小碎线（比如长度小于 40 像素的不要）
        length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if length < 40:
            continue

        # 计算角度（电线在空中通常是横向或斜向延伸，可以过滤掉垂直的电线杆线条）
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
        if angle > 80:  # 接近 90 度垂直的线过滤掉
            continue

        # 绘制检测到的电线
        cv2.line(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

# 5. 显示结果
cv2.imshow("LSD Powerline Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
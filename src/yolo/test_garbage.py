from pathlib import Path

import cv2
import numpy as np

# 1. 读取单张公路图片
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# 配置输入输出路径
img_path = str(BASE_DIR / "data/trash_cans" / "roadside_garbage.jpg")
image = cv2.imread(img_path)
output = image.copy()
h, w, _ = image.shape

# 2. 步骤一：创建路面 ROI 掩膜（过滤掉天空、绿化带）
# 实际项目中可以通过固定多边形坐标，或者让用户手动绘制
road_mask = np.zeros((h, w), dtype=np.uint8)
# 假设路面呈梯形（根据你具体的摄像头视角调整坐标）
roi_corners = np.array([[(int(w * 0.2), h), (int(w * 0.45), int(h * 0.5)),
                         (int(w * 0.55), int(h * 0.5)), (int(w * 0.8), h)]], dtype=np.int32)
cv2.fillPoly(road_mask, roi_corners, 255)

# 3. 步骤二：【锥形桶检测】—— 基于 HSV 颜色空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# 橙色锥形桶的 HSV 范围（根据实际光照可能需要微调）
lower_orange = np.array([0, 100, 100])
upper_orange = np.array([15, 255, 255])
cone_mask = cv2.inRange(hsv, lower_orange, upper_orange)
# 限制在路面 ROI 范围内
cone_mask = cv2.bitwise_and(cone_mask, road_mask)

# 4. 步骤三：【散落垃圾检测】—— 基于边缘与纹理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# 使用 Canny 算子提取边缘
edges = cv2.Canny(blurred, 30, 100)
# 限制在路面 ROI 范围内
road_edges = cv2.bitwise_and(edges, road_mask)

# 通过形态学膨胀和闭运算，把散落垃圾的边缘碎片连成一个整体块
kernel = np.ones((7, 7), np.uint8)
garbage_mask = cv2.dilate(road_edges, kernel, iterations=1)
garbage_mask = cv2.morphologyEx(garbage_mask, cv2.MORPH_CLOSE, kernel)

# 5. 步骤四：提取并过滤轮廓
# 处理锥形桶轮廓
contours_cone, _ = cv2.findContours(cone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours_cone:
    if cv2.contourArea(c) > 50:  # 过滤太小的噪点
        x, y, w_box, h_box = cv2.boundingRect(c)
        # 几何筛选：锥形桶通常是高大于宽的纵向物体
        if h_box / w_box > 1.2:
            cv2.rectangle(output, (x, y), (x + w_box, y + h_box), (0, 165, 255), 2)
            cv2.putText(output, "Cone", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

# 处理垃圾杂物轮廓
contours_garbage, _ = cv2.findContours(garbage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours_garbage:
    area = cv2.contourArea(c)
    if 100 < area < 5000:  # 限制面积范围，防止把大片车道线或者整辆车圈进去
        x, y, w_box, h_box = cv2.boundingRect(c)

        # 再次确认该区域不是由于白色车道线引起的（车道线通常很长且很窄）
        if w_box / h_box > 5 or h_box / w_box > 5:
            continue

        cv2.rectangle(output, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
        cv2.putText(output, "Garbage/Debris", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# 6. 显示结果
cv2.imshow("ROI Road Mask", road_mask)
cv2.imshow("Detected Obstacles", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
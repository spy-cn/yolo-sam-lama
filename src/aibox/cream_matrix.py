import cv2
import numpy as np
import glob

# ==================== 1. 参数设置 ====================
# 棋盘格内部角点的行数和列数（注意：是内角点，不是黑白格子的数量！）
# 例如：如果格子是 10x7，那么内角点就是 9x6
CHESSBOARD_SIZE = (10, 7)

# 棋盘格每个正方形格子的实际物理尺寸（单位可以是毫米 mm 或米 m）
# 这个值主要影响外参（平移向量）的绝对单位，不影响内参矩阵的像素值比例
SQUARE_SIZE = 25  # 假设每个格子是 25mm

# 终止条件：达到最大迭代次数 30 或精确度达到 0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ==================== 2. 初始化数据结构 ====================
# 准备三维世界坐标系下的点 (0,0,0), (1,0,0), (2,0,0) ...
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# 存储所有图像的世界坐标系点和图像坐标系点
object_points = []  # 3d point in real world space
image_points = []  # 2d points in image plane.

# 读取指定文件夹下的所有图片（根据你的实际路径修改）
images = glob.glob(r'C:\pablo\05_self_code\yolo-sam-lama\data\calibration_images\*.jpg')

if not images:
    print("❌ 未找到图片，请检查路径是否正确！")
    exit()

print(f"📸 找到 {len(images)} 张图片，开始提取角点...")

# ==================== 3. 提取角点 ====================
img_size = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img_size is None:
        img_size = gray.shape[::-1]  # 获取图像的分辨率 (W, H)

    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    # 如果找到了，添加目标点和图像点
    if ret == True:
        object_points.append(objp)

        # 亚像素级角点精准化
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        image_points.append(corners2)

        # 可选：绘制并显示角点（如果不想看可以注释掉）
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
        cv2.imshow('Chessboard Corners', cv2.resize(img, (1280, 1707)))
        cv2.waitKey(100)  # 每张图停顿 100 毫秒

cv2.destroyAllWindows()

# ==================== 4. 相机标定 ====================
print("⏳ 正在计算相机内参...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img_size, None, None)

if ret:
    print("\n🎉 标定成功！")
    # ==================== 5. 打印结果 ====================
    print("\n💡 【相机内参矩阵 (Camera Matrix)】:")
    print(mtx)
    print("\n> 其中：")
    print(f"  fx = {mtx[0, 0]:.2f} (X轴焦距-像素单位)")
    print(f"  fy = {mtx[1, 1]:.2f} (Y轴焦距-像素单位)")
    print(f"  cx = {mtx[0, 2]:.2f} (主点X坐标-光心)")
    print(f"  cy = {mtx[1, 2]:.2f} (主点Y坐标-光心)")

    print("\n💡 【畸变系数 (Distortion Coefficients)】:")
    print(f"  k1, k2, p1, p2, k3 = {dist.ravel()}")

    # ==================== 6. 计算重投影误差（评估标定好坏） ====================
    total_error = 0
    for i in range(len(object_points)):
        imgpoints2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    mean_error = total_error / len(object_points)
    print(f"\n📊 平均重投影误差 (Mean Reprojection Error): {mean_error:.4f} 像素")
    print("> 提示：该误差越接近 0 越好，通常小于 0.5 像素说明标定结果很理想。")

else:
    print("❌ 标定失败，请确保拍摄的棋盘格清晰且角点完全可见。")
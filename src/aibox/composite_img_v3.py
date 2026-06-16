import cv2
import numpy as np


def stitch_images_manual(left_path, right_path, output_path="result_v2.jpg"):
    # 1. 读取图像
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)

    if img_left is None or img_right is None:
        print("错误：无法读取图片，请检查路径。")
        return

    # === 新增：图像预处理（针对鱼眼图） ===
    # 高斯模糊可以减少噪点和轻微反光，有助于SIFT提取更稳定的特征
    blur_ksize = 3  # 如果图片分辨率高，可以设为 5
    img_left_proc = cv2.GaussianBlur(img_left, (blur_ksize, blur_ksize), 0)
    img_right_proc = cv2.GaussianBlur(img_right, (blur_ksize, blur_ksize), 0)

    # 可选：如果亮度差异大，可以尝试直方图均衡化（慎用，可能会破坏鱼眼颜色）
    # img_left_proc = cv2.cvtColor(img_left_proc, cv2.COLOR_BGR2YUV)
    # img_left_proc[:,:,0] = cv2.equalizeHist(img_left_proc[:,:,0])
    # img_left_proc = cv2.cvtColor(img_left_proc, cv2.COLOR_YUV2BGR)

    # img_right_proc = cv2.cvtColor(img_right_proc, cv2.COLOR_BGR2YUV)
    # img_right_proc[:,:,0] = cv2.equalizeHist(img_right_proc[:,:,0])
    # img_right_proc = cv2.cvtColor(img_right_proc, cv2.COLOR_YUV2BGR)

    # 2. 初始化 SIFT 探测器
    # 注意：SIFT 在 OpenCV 4.x 中可能属于额外模块，如果报错请用 ORB
    try:
        sift = cv2.SIFT_create()
    except:
        print("警告：SIFT 不可用，自动切换到 ORB (速度更快但精度略低)")
        sift = cv2.ORB_create(nfeatures=2000)  # ORB 对鱼眼有时效果更好

    # 3. 检测特征点并计算描述符
    kp_left, des_left = sift.detectAndCompute(img_left_proc, None)
    kp_right, des_right = sift.detectAndCompute(img_right_proc, None)

    print(f"左图特征点: {len(kp_left)}, 右图特征点: {len(kp_right)}")

    if des_left is None or des_right is None or len(des_left) < 2 or len(des_right) < 2:
        print("错误：特征描述符为空，图片可能没有纹理或 SIFT 提取失败。")
        return

    # 4. 使用 FLANN 匹配器
    # 如果是 ORB，需要用不同的参数
    if isinstance(sift, cv2.ORB):
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    else:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(des_left, des_right, k=2)
    except Exception as e:
        print(f"匹配过程出错: {e}")
        return

    # 5. 过滤匹配点 (Lowe's Ratio Test)
    # 针对鱼眼图，有时需要放宽比例到 0.8 甚至 0.9，避免漏掉好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:  # 从 0.7 放宽到 0.8
            good_matches.append(m)

    print(f"有效匹配点数量: {len(good_matches)}")

    if len(good_matches) < 4:
        print("错误：匹配点太少，无法计算变换矩阵。")

        # === 调试：画出特征点看看 ===
        img_with_kp_left = cv2.drawKeypoints(img_left, kp_left, None, color=(0, 255, 0), flags=0)
        img_with_kp_right = cv2.drawKeypoints(img_right, kp_right, None, color=(0, 255, 0), flags=0)
        cv2.imshow("Left Keypoints", img_with_kp_left)
        cv2.imshow("Right Keypoints", img_with_kp_right)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # 6. 提取匹配点的坐标
    pts_left = np.float32([kp_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 7. 使用 RANSAC 计算单应性矩阵 H
    H, mask = cv2.findHomography(pts_right, pts_left, cv2.RANSAC, 5.0)

    # ... (后续代码保持不变，参考你原来的逻辑) ...

    # 快速测试：如果没有黑边且匹配上了，再补全后面的逻辑
    print("成功找到单应性矩阵，继续拼接...")

    # 以下是简化的后续步骤（复用你原逻辑）
    h_l, w_l = img_left.shape[:2]
    h_r, w_r = img_right.shape[:2]

    corners_right = np.float32([[0, 0], [0, h_r], [w_r, h_r], [w_r, 0]]).reshape(-1, 1, 2)
    warped_corners_right = cv2.perspectiveTransform(corners_right, H)

    corners_left = np.float32([[0, 0], [0, h_l], [w_l, h_l], [w_l, 0]]).reshape(-1, 1, 2)
    all_corners = np.concatenate((corners_left, warped_corners_right), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    H_final = H_translation.dot(H)

    canvas_width = x_max - x_min
    canvas_height = y_max - y_min
    img_right_warped = cv2.warpPerspective(img_right, H_final, (canvas_width, canvas_height))

    result = np.zeros_like(img_right_warped)
    result[
        translation_dist[1]: h_l + translation_dist[1],
        translation_dist[0]: w_l + translation_dist[0],
    ] = img_left

    # 简单的相加融合（避免黑边最直接的方法）
    mask = (img_right_warped > 0).astype(np.uint8) * 255
    result = cv2.addWeighted(result, 1.0, img_right_warped, 1.0, 0)

    # 保存结果
    cv2.imwrite(output_path, result)
    print(f"拼接完成！结果已保存至: {output_path}")

    # 显示结果
    cv2.imshow("Stitched Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    left_path = str(BASE_DIR / "data/fish_eye/v5_role" / "left.jpg")
    right_path = str(BASE_DIR / "data/fish_eye/v5_role" / "right.jpg")
    out_path = str(BASE_DIR / "data/fish_eye/v5_role" / "5out.jpg")

    stitch_images_manual(left_path, right_path, out_path)
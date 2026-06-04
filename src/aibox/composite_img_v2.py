import cv2
import numpy as np
import matplotlib.pyplot as plt


def crop_black_borders(image):
    """自动裁剪投影后的黑边。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return image[y:y + h, x:x + w]


def undistort_fisheye_with_params(img, k1=-0.3):
    """
    使用可调参数对鱼眼图像进行去畸变
    k1: 畸变系数，通常在 -0.1 到 -0.8 之间
    """
    h, w = img.shape[:2]

    # 近似相机内参
    fov_scale = 0.8  # 视场角缩放，越小去畸变越强
    fx, fy = w * fov_scale, h * fov_scale
    cx, cy = w / 2, h / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # 畸变系数
    D = np.array([k1, 0.05, 0, 0], dtype=np.float64)

    try:
        # 计算新的相机矩阵，保持图像完整
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=0.5
        )

        # 去畸变
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
        )
        undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        return undistorted, new_K
    except Exception as e:
        print(f"⚠️ 去畸变失败: {e}")
        return img, K


def test_undistort(image_path, k1_values=[-0.2, -0.4, -0.6]):
    """
    测试不同的去畸变参数，找到最佳值
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取: {image_path}")
        return

    plt.figure(figsize=(15, 5))

    # 显示原图
    plt.subplot(1, len(k1_values) + 1, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')

    # 测试不同的 k1 值
    for i, k1 in enumerate(k1_values):
        undistorted, _ = undistort_fisheye_with_params(img, k1)
        plt.subplot(1, len(k1_values) + 1, i + 2)
        plt.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
        plt.title(f'k1={k1}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('undistort_test.png', dpi=150)
    print("✅ 去畸变测试图已保存: undistort_test.png")
    plt.show()


def manual_stitch_panorama(image_files, output_file='panorama.jpg', k1=-0.4):
    """
    手动实现全景拼接，更灵活的控制
    """
    print("=" * 50)
    print("开始手动拼接...")
    print("=" * 50)

    # 1. 读取并去畸变
    imgs = []
    for f in image_files:
        img = cv2.imread(f)
        if img is None:
            print(f"❌ 无法读取: {f}")
            continue

        # 去畸变
        print(f"🔄 去畸变: {f.split('/')[-1]}")
        img_undistorted, _ = undistort_fisheye_with_params(img, k1)

        # 缩小图像
        scale = 0.5
        img_undistorted = cv2.resize(img_undistorted, None, fx=scale, fy=scale)
        imgs.append(img_undistorted)

    if len(imgs) < 2:
        print("❌ 图像不足")
        return

    print(f"✓ 加载了 {len(imgs)} 张图像")

    # 2. 特征提取和匹配
    orb = cv2.ORB_create(nfeatures=3000)

    # 提取所有图像的特征
    kps_list = []
    des_list = []
    for i, img in enumerate(imgs):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, des = orb.detectAndCompute(gray, None)
        kps_list.append(kps)
        des_list.append(des)
        print(f"✓ 图像{i} 特征点: {len(kps)}")

    # 3. 逐对匹配和拼接
    result = imgs[0]

    for i in range(1, len(imgs)):
        print(f"\n🔗 拼接图像 {i - 1} 和 {i}...")

        # 匹配特征点
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des_list[i - 1], des_list[i], k=2)

        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        print(f"✓ 匹配点对: {len(good_matches)}")

        if len(good_matches) < 10:
            print("⚠️ 匹配点太少，跳过此图像")
            continue

        # 获取匹配点坐标
        src_pts = np.float32([kps_list[i - 1][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps_list[i][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 计算单应性矩阵
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        if H is None:
            print("⚠️ 无法计算单应性矩阵")
            continue

        # 获取图像尺寸
        h1, w1 = result.shape[:2]
        h2, w2 = imgs[i].shape[:2]

        # 计算变换后的角点
        corners = np.float32([
            [0, 0], [0, h2], [w2, h2], [w2, 0]
        ]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)

        # 计算输出图像尺寸
        all_corners = np.vstack([
            [[0, 0]], [[0, h1]], [[w1, h1]], [[w1, 0]],
            transformed_corners
        ])

        xmin = int(np.floor(all_corners[:, 0, 0].min()))
        xmax = int(np.ceil(all_corners[:, 0, 0].max()))
        ymin = int(np.floor(all_corners[:, 0, 1].min()))
        ymax = int(np.ceil(all_corners[:, 0, 1].max()))

        out_w = xmax - xmin
        out_h = ymax - ymin

        # 调整单应性矩阵
        H_adj = H.copy()
        H_adj[0, 2] += abs(xmin)
        H_adj[1, 2] += abs(ymin)

        # 创建输出图像
        output = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        # 变换当前结果
        warped_result = cv2.warpPerspective(result, np.eye(3), (out_w, out_h))
        warped_result[abs(ymin):abs(ymin) + h1, abs(xmin):abs(xmin) + w1] = result

        # 变换新图像
        warped_img = cv2.warpPerspective(imgs[i], H_adj, (out_w, out_h))

        # 简单融合（重叠区域取平均）
        mask1 = (warped_result > 0).astype(np.uint8)
        mask2 = (warped_img > 0).astype(np.uint8)
        overlap = mask1 * mask2

        output = warped_result.copy()
        output[mask2 == 1] = warped_img[mask2 == 1]

        # 重叠区域混合
        if np.sum(overlap) > 0:
            output = cv2.addWeighted(warped_result, 0.5, warped_img, 0.5, 0)
            # 非重叠区域恢复
            output[mask1 == 0] = warped_img[mask1 == 0]
            output[mask2 == 0] = warped_result[mask2 == 0]

        result = output
        print(f"✓ 拼接后尺寸: {result.shape[1]}x{result.shape[0]}")

    # 保存结果
    result = crop_black_borders(result)
    cv2.imwrite(output_file, result)
    print(f"\n✅ 拼接完成！保存至: {output_file}")

    # 显示结果
    plt.figure(figsize=(15, 5))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Panorama Result')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file.replace('.jpg', '_preview.png'), dpi=150)
    plt.show()


# ==================== 主程序 ====================
if __name__ == "__main__":
    left_path = "/Users/spy/Documents/codes/python_code/yolo-sam-lama/data/fish_eye/v2/left.jpg"
    center_path = "/Users/spy/Documents/codes/python_code/yolo-sam-lama/data/fish_eye/v2/center.jpg"
    right_path = "/Users/spy/Documents/codes/python_code/yolo-sam-lama/data/fish_eye/v2/right.jpg"
    output_path = "/Users/spy/Documents/codes/python_code/yolo-sam-lama/src/aibox/result.jpg"

    image_files = [left_path, center_path, right_path]

    # 步骤1：测试去畸变参数
    print("步骤1: 测试去畸变参数")
    print("请查看生成的 undistort_test.png，选择最佳 k1 值")
    test_undistort(center_path, k1_values=[-0.2, -0.4, -0.6, -0.8])

    # 步骤2：使用最佳参数拼接
    print("\n步骤2: 开始拼接")
    best_k1 = -0.4  # 根据步骤1的结果调整这个值
    manual_stitch_panorama(image_files, output_path, k1=best_k1)
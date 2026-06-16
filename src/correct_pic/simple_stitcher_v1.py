from pathlib import Path
import cv2
import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent
to_join_img_left = str(ROOT_DIR / "data/to_join_img/5_left.JPG")
to_join_img_right = str(ROOT_DIR / "data/to_join_img/5_right.JPG")
out_path = str(ROOT_DIR / "data/to_join_img/5_result.jpg")
final_out_puth = str(ROOT_DIR / "data/to_join_img/5_final_result.jpg")


def simple_correct_img():
    img1 = cv2.imread(to_join_img_left)
    img2 = cv2.imread(to_join_img_right)

    stitcher = cv2.Stitcher_create()
    status, pano = stitcher.stitch([img1, img2])

    # 错误码解释字典
    ERROR_MSGS = {
        cv2.Stitcher_OK: "拼接成功",
        cv2.Stitcher_ERR_NEED_MORE_IMGS: "特征点太少，需要更多图片（或图像纹理不足）",
        # 图像纹理太弱（如大片天空、白墙），可换用特征丰富的区域，或调整重叠比例。
        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "单应矩阵估计失败（无法找到足够的匹配点对）",
        # 两图重叠区域不够或视差太大，可增加重叠（30%~50%为宜），或确保是旋转拍摄而非平移。
        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "相机参数调整失败（全局配准出错）",
        # 多图拼接时配准失败；两张图通常不会遇到，检查图像是否有效、有无严重畸变。
    }

    print(f"状态码: {status} -> {ERROR_MSGS.get(status, '未知错误')}")

    if status == cv2.Stitcher_OK:
        cv2.imwrite(out_path, pano)
        print("拼接成功，结果已保存")
    else:
        print(f"拼接失败: {ERROR_MSGS.get(status, '未知错误')}")


def remove_black_borders(img_path, output_path):
    """
    一键去除黑边
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 修复法
    mask = (gray < 10).astype(np.uint8) * 255
    # 先膨胀掩膜，确保覆盖所有黑边
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    result = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)
    cv2.imwrite(output_path, result)
    print(f"方法 inpaint 处理完成，保存至 {output_path}")
    return result


def undistort_image(image_path, output_path):
    """
    去畸变
    :param image_path: 图片路径
    :param output_path: 输出路径
    :return:
    """
    # 1. 读取原图
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # 2. 伪造一个相机内参矩阵 (Camera Matrix)
    # 假设焦距 fx, fy 接近图片的宽度
    fx = w * 0.8
    fy = w * 0.8
    cx = w / 2.0
    cy = h / 2.0
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)

    # 3. 设置畸变系数 (Distortion Coefficients)
    # k1, k2 为径向畸变参数。因为原图是“桶形弯曲”，我们需要用负数来“往外拉”
    # 你可以微调这两个数值（例如 -0.2 到 -0.4 之间）直到线条变直
    k1 = -0.28
    k2 = 0.2
    p1 = 0.00
    p2 = 0.00
    dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float32)

    # 4. 计算去畸变映射
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1)

    # 5. 重映射图像
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # 6. 裁剪掉去畸变后产生的边缘黑边（可选）
    x, y, w_box, h_box = roi
    dst_cropped = dst[y:y + h_box, x:x + w_box]

    # 7. 保存图片
    cv2.imwrite(output_path, dst_cropped)
    print("畸变矫正完成！")



if __name__ == "__main__":
    simple_correct_img()
    remove_black_borders(out_path)
    undistort_image(final_out_puth, "5_undistorted_result.jpg")

#!/usr/bin/env python3
"""
两路前视鱼眼图像拼接系统（免命令行版）
可直接 python 运行
"""

import os
import numpy as np
import cv2

# ============================================================================
# ✅ 用户配置区（在这里改，不用命令行）
# ============================================================================

# ---------- 运行模式 ----------
RUN_MODE = "stitch"   # "stitch" | "calibrate"

# ---------- 输入文件 ----------
LEFT_YAML  = "yaml/left_front.yaml"
RIGHT_YAML = "yaml/right_front.yaml"

LEFT_IMG  = r"C:\pablo\05_self_code\yolo-sam-lama\data\fish_eye\v1\1.jpg"
RIGHT_IMG = r"C:\pablo\05_self_code\yolo-sam-lama\data\fish_eye\v1\2.jpg"

# ---------- 投影尺寸 ----------
PROJ_W = 1280
PROJ_H = 1707

# ---------- 拼接参数 ----------
OVERLAP = 200
WEIGHT_MODE = "linear"   # "linear" | "content"
DO_LUMINANCE_BALANCE = True

# ---------- 输出 ----------
OUTPUT_PATH = "frontview_result.png"
SHOW_RESULT = True

# ============================================================================
# 以下为原系统核心代码（未改动）
# ============================================================================

TOTAL_W = PROJ_W * 2 - OVERLAP
TOTAL_H = PROJ_H

LEFT_DST_POINTS = [(0, 0), (PROJ_W, 0), (0, PROJ_H), (PROJ_W, PROJ_H)]
RIGHT_DST_POINTS = [(0, 0), (PROJ_W, 0), (0, PROJ_H), (PROJ_W, PROJ_H)]

# ---------- YAML 读取 ----------
def load_camera_params(yaml_file):
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"找不到标定参数文件: {yaml_file}")
    fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_READ)
    params = {
        'camera_matrix': fs.getNode("camera_matrix").mat(),
        'dist_coeffs': fs.getNode("dist_coeffs").mat(),
        'resolution': fs.getNode("resolution").mat().flatten() if not fs.getNode("resolution").empty() else None,
        'project_matrix': fs.getNode("project_matrix").mat(),
        'scale_xy': fs.getNode("scale_xy").mat(),
        'shift_xy': fs.getNode("shift_xy").mat(),
    }
    fs.release()
    if params['scale_xy'] is not None:
        params['scale_xy'] = params['scale_xy'].flatten()
    if params['shift_xy'] is not None:
        params['shift_xy'] = params['shift_xy'].flatten()
    return params


def save_camera_params(yaml_file, params):
    fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", params['camera_matrix'])
    fs.write("dist_coeffs", params['dist_coeffs'])
    if params['resolution'] is not None:
        fs.write("resolution", np.int32(params['resolution']))
    if params['project_matrix'] is not None:
        fs.write("project_matrix", params['project_matrix'])
    if params['scale_xy'] is not None:
        fs.write("scale_xy", np.float32(params['scale_xy']))
    if params['shift_xy'] is not None:
        fs.write("shift_xy", np.float32(params['shift_xy']))
    fs.release()


# ---------- 去畸变 ----------
def build_undistort_maps(params):
    K = params['camera_matrix'].copy()
    if params['scale_xy'] is not None:
        K[0, 0] *= params['scale_xy'][0]
        K[1, 1] *= params['scale_xy'][1]
    if params['shift_xy'] is not None:
        K[0, 2] += params['shift_xy'][0]
        K[1, 2] += params['shift_xy'][1]

    resolution = params['resolution']
    if resolution is None:
        resolution = np.array([960, 640])
    w, h = int(resolution[0]), int(resolution[1])

    map1, map2 = cv2.initUndistortRectifyMap(
        params['camera_matrix'],
        params['dist_coeffs'],
        np.eye(3),
        K,
        (w, h),
        cv2.CV_16SC2
    )
    return map1, map2


def undistort(image, map1, map2):
    return cv2.remap(image, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def project(image, project_matrix, proj_shape):
    return cv2.warpPerspective(image, project_matrix, proj_shape)


# ---------- 标定 ----------
class PointSelector:
    POINT_COLOR = (0, 0, 255)
    FILL_COLOR = (0, 255, 255)

    def __init__(self, image, title):
        self.image = image.copy()
        self.title = title
        self.keypoints = []
        self.display_image = image.copy()

    def draw(self):
        self.display_image = self.image.copy()
        for i, pt in enumerate(self.keypoints):
            cv2.circle(self.display_image, pt, 6, self.POINT_COLOR, -1)
            cv2.putText(self.display_image, str(i), (pt[0], pt[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.POINT_COLOR, 2)
        if len(self.keypoints) > 2:
            pts = np.int32(self.keypoints).reshape(-1, 1, 2)
            hull = cv2.convexHull(pts)
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)
            overlay = np.zeros_like(self.display_image)
            overlay[:, :] = self.FILL_COLOR
            overlay = cv2.bitwise_and(overlay, overlay, mask=mask)
            cv2.addWeighted(self.display_image, 1.0, overlay, 0.5, 0.0, self.display_image)
        cv2.imshow(self.title, self.display_image)

    def onclick(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.keypoints) < 4:
            self.keypoints.append((x, y))
            self.draw()

    def loop(self):
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.title, self.onclick)
        self.draw()
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and len(self.keypoints) == 4:
                cv2.destroyWindow(self.title)
                return True
            if key == ord('q'):
                cv2.destroyWindow(self.title)
                return False


def calibrate_camera(yaml_file, image_path, dst_points, label):
    params = load_camera_params(yaml_file)
    img = cv2.imread(image_path)
    if img is None:
        print("图像读取失败:", image_path)
        return False

    map1, map2 = build_undistort_maps(params)
    und = undistort(img, map1, map2)

    gui = PointSelector(und, f"{label} 标定")
    if not gui.loop():
        return False

    src = np.float32(gui.keypoints)
    dst = np.float32(dst_points)
    params['project_matrix'] = cv2.getPerspectiveTransform(src, dst)

    proj = project(und, params['project_matrix'], (PROJ_W, PROJ_H))
    cv2.imshow("Preview", proj)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save_camera_params(yaml_file, params)
    return True


# ---------- 权重 ----------
def compute_feathering_weights(left_img, right_img, overlap_w, mode="linear"):
    h = left_img.shape[0]
    if mode == "linear":
        G = np.zeros((h, overlap_w), dtype=np.float32)
        for x in range(overlap_w):
            G[:, x] = 1.0 - x / max(overlap_w - 1, 1)
        return G
    return np.ones((h, overlap_w), dtype=np.float32) * 0.5


# ---------- 拼接 ----------
def stitch_horizontal(left_img, right_img, overlap_w, weights):
    h, left_w = left_img.shape[:2]
    right_w = right_img.shape[1]
    non_overlap = left_w - overlap_w
    result = np.zeros((h, non_overlap + right_w, 3), dtype=np.uint8)

    result[:, :non_overlap] = left_img[:, :non_overlap]
    result[:, non_overlap:] = right_img

    G = np.stack([weights] * 3, axis=2)
    blended = (left_img[:, -overlap_w:].astype(np.float32) * G +
               right_img[:, :overlap_w].astype(np.float32) * (1 - G)).astype(np.uint8)
    result[:, non_overlap:non_overlap + overlap_w] = blended
    return result


# ---------- 亮度均衡 ----------
def simple_luminance_balance(left_img, right_img, overlap_w):
    l = left_img[:, -overlap_w:]
    r = right_img[:, :overlap_w]
    l_gray = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
    r_gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    l_mean = l_gray.mean()
    r_mean = r_gray.mean()
    if l_mean < r_mean:
        left_img = np.clip(left_img * (r_mean / l_mean), 0, 255).astype(np.uint8)
    else:
        right_img = np.clip(right_img * (l_mean / r_mean), 0, 255).astype(np.uint8)
    return left_img, right_img


# ============================================================================
# 主入口
# ============================================================================

def main():
    if RUN_MODE == "calibrate":
        calibrate_camera(LEFT_YAML, LEFT_IMG, LEFT_DST_POINTS, "Left")
        calibrate_camera(RIGHT_YAML, RIGHT_IMG, RIGHT_DST_POINTS, "Right")
        return

    left_params = load_camera_params(LEFT_YAML)
    right_params = load_camera_params(RIGHT_YAML)

    left_img = cv2.imread(LEFT_IMG)
    right_img = cv2.imread(RIGHT_IMG)

    m1, m2 = build_undistort_maps(left_params)
    left_und = undistort(left_img, m1, m2)
    m1, m2 = build_undistort_maps(right_params)
    right_und = undistort(right_img, m1, m2)

    left_proj = project(left_und, left_params['project_matrix'], (PROJ_W, PROJ_H))
    right_proj = project(right_und, right_params['project_matrix'], (PROJ_W, PROJ_H))

    if DO_LUMINANCE_BALANCE:
        left_proj, right_proj = simple_luminance_balance(left_proj, right_proj, OVERLAP)

    weights = compute_feathering_weights(left_proj, right_proj, OVERLAP, WEIGHT_MODE)
    result = stitch_horizontal(left_proj, right_proj, OVERLAP, weights)

    cv2.imwrite(OUTPUT_PATH, result)
    print("结果已保存:", OUTPUT_PATH)

    if SHOW_RESULT:
        cv2.imshow("Result", result)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
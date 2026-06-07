#!/usr/bin/env python3
"""
两路前视鱼眼图像拼接系统（免命令行版）
可直接 python 运行

功能：将左、右两个前视鱼眼相机的图像进行畸变校正、透视投影和水平拼接，
      输出一张宽视角的全景前视图。
模式：可通过 RUN_MODE 切换 "stitch"（拼接）和 "calibrate"（标定投影矩阵）。
"""

import os
import numpy as np
import cv2

# ============================================================================
# ✅ 用户配置区（在这里修改参数，无需命令行）
# ============================================================================

# ---------- 运行模式 ----------
# "stitch" : 使用已有的 YAML 参数进行拼接
# "calibrate" : 手动选取四个点，计算投影矩阵并保存到 YAML
RUN_MODE = "stitch"

# ---------- 输入文件 ----------
# 左右相机的标定参数文件（YAML格式），包含相机矩阵、畸变系数、投影矩阵等
LEFT_YAML = "yaml/left_front.yaml"
RIGHT_YAML = "yaml/right_front.yaml"

# 左右相机的原始鱼眼图像路径
LEFT_IMG = r"C:\pablo\05_self_code\yolo-sam-lama\data\fish_eye\v1\1.jpg"
RIGHT_IMG = r"C:\pablo\05_self_code\yolo-sam-lama\data\fish_eye\v1\2.jpg"

# ---------- 投影尺寸 ----------
# 单张图像经过透视投影后的宽和高（像素）
PROJ_W = 1280  # 投影宽度
PROJ_H = 1707  # 投影高度

# ---------- 拼接参数 ----------
# 左右投影图水平方向的重叠宽度（像素），决定拼接时的融合区域大小
OVERLAP = 200

# 重叠区域融合权重模式
# "linear" : 线性渐入渐出（左图权重从1线性降到0，右图从0升到1）
# "content" : 等权重融合（各0.5），实际代码中仅实现了linear和均分
WEIGHT_MODE = "linear"

# 是否在拼接前进行亮度均衡，减少左右图像曝光差异
DO_LUMINANCE_BALANCE = True

# ---------- 输出 ----------
# 拼接结果图像的保存路径
OUTPUT_PATH = "frontview_result.png"

# 是否在运行结束后显示结果图像窗口
SHOW_RESULT = True

# ============================================================================
# 以下为系统核心代码（已添加详细注释）
# ============================================================================

# 计算拼接后最终图像的尺寸
# 宽度 = 两张投影图宽度之和 减去 重叠部分宽度
TOTAL_W = PROJ_W * 2 - OVERLAP
TOTAL_H = PROJ_H

# 标定时使用的目标点坐标：投影后图像的四个角（左上、右上、左下、右下）
# 这些点构成了一个与投影尺寸一致的矩形
LEFT_DST_POINTS = [(0, 0), (PROJ_W, 0), (0, PROJ_H), (PROJ_W, PROJ_H)]
RIGHT_DST_POINTS = [(0, 0), (PROJ_W, 0), (0, PROJ_H), (PROJ_W, PROJ_H)]


# ---------- YAML 读取与保存 ----------

def load_camera_params(yaml_file):
    """
    从 YAML 文件中加载相机标定参数。
    参数:
        yaml_file: YAML 文件路径
    返回:
        params: 字典，包含 'camera_matrix' (相机内参矩阵),
                'dist_coeffs' (畸变系数), 'resolution' (图像分辨率),
                'project_matrix' (透视投影矩阵),
                'scale_xy' (缩放因子), 'shift_xy' (平移量)
    """
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
    # 将一维数组转换为更易处理的形状
    if params['scale_xy'] is not None:
        params['scale_xy'] = params['scale_xy'].flatten()
    if params['shift_xy'] is not None:
        params['shift_xy'] = params['shift_xy'].flatten()
    return params


def save_camera_params(yaml_file, params):
    """
    将相机标定参数保存到 YAML 文件，主要用于保存更新后的投影矩阵。
    参数:
        yaml_file: 目标 YAML 文件路径
        params: 同 load_camera_params 返回的字典结构
    """
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


# ---------- 去畸变相关 ----------

def build_undistort_maps(params):
    """
    根据相机参数生成畸变校正的映射表 (map1, map2)。
    这里会考虑可选的缩放 scale_xy 和平移 shift_xy，用于调整有效视场。

    参数:
        params: 相机参数字典
    返回:
        map1, map2: OpenCV remap 使用的映射表（CV_16SC2 格式）
    """
    # 复制原始相机矩阵，避免修改原数据
    K = params['camera_matrix'].copy()
    # 如果存在缩放因子，调整焦距和光心
    if params['scale_xy'] is not None:
        K[0, 0] *= params['scale_xy'][0]  # fx
        K[1, 1] *= params['scale_xy'][1]  # fy
    if params['shift_xy'] is not None:
        K[0, 2] += params['shift_xy'][0]  # cx
        K[1, 2] += params['shift_xy'][1]  # cy

    # 获取图像分辨率，若未提供则使用默认值 960x640
    resolution = params['resolution']
    if resolution is None:
        resolution = np.array([960, 640])
    w, h = int(resolution[0]), int(resolution[1])

    # 计算去畸变映射：使用原始相机矩阵和畸变系数，目标为新相机矩阵 K
    map1, map2 = cv2.initUndistortRectifyMap(
        params['camera_matrix'],  # 原始相机矩阵
        params['dist_coeffs'],  # 畸变系数（k1,k2,p1,p2,k3...）
        np.eye(3),  # 旋转矩阵（无旋转）
        K,  # 新的相机矩阵（用于缩放/平移）
        (w, h),  # 图像尺寸
        cv2.CV_16SC2  # 输出映射格式，速度快，节省内存
    )
    return map1, map2


def undistort(image, map1, map2):
    """
    应用预先计算好的映射表对图像进行畸变校正。
    参数:
        image: 原始鱼眼图像
        map1, map2: 由 build_undistort_maps 生成的映射表
    返回:
        去畸变后的图像，超出边界的部分填充黑色
    """
    return cv2.remap(image, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def project(image, project_matrix, proj_shape):
    """
    对去畸变后的图像应用透视投影变换，转换为指定尺寸的前视平面。
    参数:
        image: 去畸变后的图像
        project_matrix: 3×3 透视变换矩阵
        proj_shape: (宽, 高) 目标尺寸元组
    返回:
        投影后的图像
    """
    return cv2.warpPerspective(image, project_matrix, proj_shape)


# ---------- 标定交互（手动选取控制点） ----------

class PointSelector:
    """
    交互式点选择器：在图像上单击鼠标左键选择四个控制点。
    用于标定投影矩阵，将选中的四边形区域变换到目标矩形。
    """
    POINT_COLOR = (0, 0, 255)  # 已选点的颜色（红色）
    FILL_COLOR = (0, 255, 255)  # 多边形填充颜色（黄色）

    def __init__(self, image, title):
        """
        初始化选择器。
        image: 要显示的图像（通常为去畸变后的图）
        title: 窗口标题
        """
        self.image = image.copy()
        self.title = title
        self.keypoints = []  # 存储已选择的点 (x, y)
        self.display_image = image.copy()

    def draw(self):
        """
        在图像上绘制已选择的点和它们构成的凸多边形（填充半透明色）。
        若已选点超过2个，会画出凸包填充，帮助判断覆盖区域。
        """
        self.display_image = self.image.copy()
        # 绘制已选点及序号
        for i, pt in enumerate(self.keypoints):
            cv2.circle(self.display_image, pt, 6, self.POINT_COLOR, -1)
            cv2.putText(self.display_image, str(i), (pt[0], pt[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.POINT_COLOR, 2)
        # 如果点数大于2，计算凸包并填充半透明区域
        if len(self.keypoints) > 2:
            pts = np.int32(self.keypoints).reshape(-1, 1, 2)
            hull = cv2.convexHull(pts)  # 凸包，保证填充区域是凸四边形
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)
            overlay = np.zeros_like(self.display_image)
            overlay[:, :] = self.FILL_COLOR
            overlay = cv2.bitwise_and(overlay, overlay, mask=mask)
            cv2.addWeighted(self.display_image, 1.0, overlay, 0.5, 0.0, self.display_image)
        cv2.imshow(self.title, self.display_image)

    def onclick(self, event, x, y, flags, param):
        """
        鼠标回调函数：左键单击添加点，最多4个。
        """
        if event == cv2.EVENT_LBUTTONDOWN and len(self.keypoints) < 4:
            self.keypoints.append((x, y))
            self.draw()

    def loop(self):
        """
        进入交互循环，等待用户选择4个点并按回车确认，或按 'q' 退出。
        返回:
            True  选择成功（4个点并按回车）
            False 用户取消（按 'q'）
        """
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.title, self.onclick)
        self.draw()
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and len(self.keypoints) == 4:  # 回车键确认
                cv2.destroyWindow(self.title)
                return True
            if key == ord('q'):  # 'q' 键退出
                cv2.destroyWindow(self.title)
                return False


def calibrate_camera(yaml_file, image_path, dst_points, label):
    """
    对一个相机进行一次标定流程：
      1. 加载 YAML 参数（需已有内参和畸变系数）
      2. 读取图像并去畸变
      3. 通过 PointSelector 让用户选取四个源点
      4. 计算透视变换矩阵并保存到 YAML
      5. 显示投影预览

    参数:
        yaml_file:   相机参数 YAML 路径
        image_path:  用于标定的原始图像路径
        dst_points:  投影后目标矩形的四个角点（固定为 LEFT_DST_POINTS 或 RIGHT_DST_POINTS）
        label:       窗口标签（例如 "Left" 或 "Right"）
    返回:
        True  标定成功并保存
        False 操作取消
    """
    params = load_camera_params(yaml_file)
    img = cv2.imread(image_path)
    if img is None:
        print("图像读取失败:", image_path)
        return False

    # 去畸变
    map1, map2 = build_undistort_maps(params)
    und = undistort(img, map1, map2)

    # 交互选点
    gui = PointSelector(und, f"{label} 标定")
    if not gui.loop():
        return False

    # 计算透视变换矩阵：将用户选取的源点映射到目标矩形
    src = np.float32(gui.keypoints)  # 用户点击的四个点（顺序应一致）
    dst = np.float32(dst_points)  # 目标矩形的四个角
    params['project_matrix'] = cv2.getPerspectiveTransform(src, dst)

    # 生成投影预览
    proj = project(und, params['project_matrix'], (PROJ_W, PROJ_H))
    cv2.imshow("Preview", proj)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存更新后的参数（特别是投影矩阵）
    save_camera_params(yaml_file, params)
    return True


# ---------- 拼接融合相关 ----------

def compute_feathering_weights(left_img, right_img, overlap_w, mode="linear"):
    """
    计算重叠区域的融合权重图，用于水平拼接时的渐入渐出效果。

    参数:
        left_img:   左投影图（仅用于获取高度信息）
        right_img:  右投影图（未直接使用，但保留接口）
        overlap_w:  重叠区域宽度（像素）
        mode:       "linear" 生成线性渐变权重；其他值返回 0.5 均分
    返回:
        权重矩阵 shape=(h, overlap_w)，值在 0~1 之间，表示左图的权重
        （右图权重 = 1 - 左图权重）
    """
    h = left_img.shape[0]  # 图像高度
    if mode == "linear":
        # 线性权重：重叠区从左到右，左图权重由 1 逐渐降为 0
        G = np.zeros((h, overlap_w), dtype=np.float32)
        for x in range(overlap_w):
            # 当 overlap_w 为 1 时避免除零
            G[:, x] = 1.0 - x / max(overlap_w - 1, 1)
        return G
    # 其他模式（包括 "content"）默认均分
    return np.ones((h, overlap_w), dtype=np.float32) * 0.5


def stitch_horizontal(left_img, right_img, overlap_w, weights):
    """
    水平拼接两幅图像，重叠区域按权重融合。

    参数:
        left_img:   左投影图
        right_img:  右投影图
        overlap_w:  重叠宽度（像素）
        weights:    重叠区左图权重，shape=(h, overlap_w)
    返回:
        拼接后的图像，尺寸 (h, left_w + right_w - overlap_w, 3)
    """
    h, left_w = left_img.shape[:2]
    right_w = right_img.shape[1]
    non_overlap = left_w - overlap_w  # 左图不重叠部分的宽度

    # 创建结果画布，宽度 = 左图非重叠部分 + 整个右图
    result = np.zeros((h, non_overlap + right_w, 3), dtype=np.uint8)

    # 复制左图非重叠区域
    result[:, :non_overlap] = left_img[:, :non_overlap]

    # 复制整个右图到右侧（重叠区会被之后覆盖）
    result[:, non_overlap:] = right_img

    # 扩展权重为3通道，以便与彩色图像相乘
    G = np.stack([weights] * 3, axis=2)

    # 重叠区域加权融合：Blended = left * G + right * (1 - G)
    blended = (left_img[:, -overlap_w:].astype(np.float32) * G +
               right_img[:, :overlap_w].astype(np.float32) * (1 - G)).astype(np.uint8)

    # 用融合结果替换画布中的重叠区域
    result[:, non_overlap:non_overlap + overlap_w] = blended
    return result


# ---------- 亮度均衡 ----------

def simple_luminance_balance(left_img, right_img, overlap_w):
    """
    简单的亮度均衡：计算左右图重叠区的平均灰度，调整较暗的图像使其亮度与较亮者匹配。

    参数:
        left_img, right_img: 左右投影图
        overlap_w: 重叠宽度
    返回:
        调整亮度后的 left_img, right_img
    """
    # 提取重叠区域（灰度图）
    l = left_img[:, -overlap_w:]
    r = right_img[:, :overlap_w]
    l_gray = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
    r_gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)

    # 计算平均灰度
    l_mean = l_gray.mean()
    r_mean = r_gray.mean()

    # 若左图较暗，则将左图乘以 (r_mean / l_mean) 提亮；否则提亮右图
    if l_mean < r_mean:
        # 乘法调整可能溢出，需用 clip 限制在 0~255
        left_img = np.clip(left_img * (r_mean / l_mean), 0, 255).astype(np.uint8)
    else:
        right_img = np.clip(right_img * (l_mean / r_mean), 0, 255).astype(np.uint8)
    return left_img, right_img


# ============================================================================
# 主入口
# ============================================================================

def main():
    # ---- 标定模式：手动选取投影变换点，生成投影矩阵并保存 ----
    if RUN_MODE == "calibrate":
        calibrate_camera(LEFT_YAML, LEFT_IMG, LEFT_DST_POINTS, "Left")
        calibrate_camera(RIGHT_YAML, RIGHT_IMG, RIGHT_DST_POINTS, "Right")
        return

    # ---- 拼接模式：加载参数，处理图像，输出全景图 ----

    # 1. 加载左右相机的标定参数
    left_params = load_camera_params(LEFT_YAML)
    right_params = load_camera_params(RIGHT_YAML)

    # 2. 读取原始鱼眼图像
    left_img = cv2.imread(LEFT_IMG)
    right_img = cv2.imread(RIGHT_IMG)

    # 3. 生成去畸变映射并执行校正
    m1, m2 = build_undistort_maps(left_params)
    left_und = undistort(left_img, m1, m2)
    m1, m2 = build_undistort_maps(right_params)
    right_und = undistort(right_img, m1, m2)

    # 4. 透视投影到前视平面
    left_proj = project(left_und, left_params['project_matrix'], (PROJ_W, PROJ_H))
    right_proj = project(right_und, right_params['project_matrix'], (PROJ_W, PROJ_H))

    # 5. 可选的亮度均衡处理
    if DO_LUMINANCE_BALANCE:
        left_proj, right_proj = simple_luminance_balance(left_proj, right_proj, OVERLAP)

    # 6. 计算融合权重并进行水平拼接
    weights = compute_feathering_weights(left_proj, right_proj, OVERLAP, WEIGHT_MODE)
    result = stitch_horizontal(left_proj, right_proj, OVERLAP, weights)

    # 7. 保存结果并（可选）显示
    cv2.imwrite(OUTPUT_PATH, result)
    print("结果已保存:", OUTPUT_PATH)

    if SHOW_RESULT:
        cv2.imshow("Result", result)
        cv2.waitKey(0)  # 按任意键关闭窗口
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
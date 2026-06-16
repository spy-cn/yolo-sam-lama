import cv2
import numpy as np

"""
实现两幅图像的平滑拼接与色彩平衡（图像融合）。用于全景图拼接、多摄像头视野融合等场景
"""


def gstreamer_pipeline(cam_id=0,
                       capture_width=960,
                       capture_height=640,
                       framerate=60,
                       flip_method=2):
    """
    使用 libgstreamer 打开 CSI 摄像头（通常用于 NVIDIA Jetson 系列开发板）。
    构建并返回一个 GStreamer 管道字符串，供 cv2.VideoCapture 使用。
    """
    return ("nvarguscamerasrc sensor-id={} ! ".format(cam_id) +
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "  # 设置分辨率、格式(NV12)和帧率
            "nvvidconv flip-method=%d ! "  # 硬件加速的视频转换，处理翻转
            "video/x-raw, format=(string)BGRx ! "  # 转换为 BGRx 格式
            "videoconvert ! "  # 视频格式转换
            "video/x-raw, format=(string)BGR ! appsink"  # 最终输出 OpenCV 兼容的 BGR 格式并送入应用接收器
            % (capture_width,
               capture_height,
               framerate,
               flip_method
               )
            )


def convert_binary_to_bool(mask):
    """
    将二值图像（单通道，像素值为 0 或 255）转换为布尔/整数掩码（像素值为 0 或 1）。
    """
    # 先转为 float64 归一化到 0.0~1.0，然后再强转为整型 0 或 1
    return (mask.astype(np.float64) / 255.0).astype(int)


def adjust_luminance(gray, factor):
    """
    通过乘法因子调整灰度图像（或单通道图像）的亮度。
    """
    # np.minimum 确保亮度调整后不会超过最大值 255（防止溢出），最后转回 uint8 8位无符号整数
    return np.minimum((gray * factor), 255).astype(np.uint8)


def get_mean_statistisc(gray, mask):
    """
    计算灰度图像在掩码（mask）指定区域内的像素总和。
    注意：此处的 mask 像素值必须是 0 或 1（由 convert_binary_to_bool 转换而来）。
    """
    # 矩阵点乘：mask 为 0 的地方结果为 0，mask 为 1 的地方保留原像素值，最后求和
    return np.sum(gray * mask)


def mean_luminance_ratio(grayA, grayB, mask):
    """
    计算图像 A 和图像 B 在同一个重叠掩码区域内的平均亮度比例。
    """
    return get_mean_statistisc(grayA, mask) / get_mean_statistisc(grayB, mask)


def get_mask(img):
    """
    将输入的彩色/灰度图像转换为二值掩码（Mask）数组。
    有像素内容（大于0）的地方变成 255（白色），没有内容的地方为 0（黑色）。
    """
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 大津法/固定阈值：只要像素值大于 0，就设为 255
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    return mask


def get_overlap_region_mask(imA, imB):
    """
    给定两张已经对齐（大小相同）的拼接图像，通过位运算找出它们的“重叠区域”，
    并将其转换为二值掩码。
    """
    # 按位与运算：只有 imA 和 imB 在该位置都有像素时，结果才不为 0
    overlap = cv2.bitwise_and(imA, imB)
    # 转换成二值掩码
    mask = get_mask(overlap)
    # 使用 2x2 的全 1 结构元素进行 2 次膨胀操作，用以消除边缘小空隙，使重叠区稍微扩大一点
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)
    return mask


def get_outmost_polygon_boundary(img):
    """
    给定一个只包含图像各自非重叠（独立）区域的图像，获取其最外层的多边形边界。
    用于计算重叠区像素到图像边界的距离。
    """
    mask = get_mask(img)
    # 膨胀处理，平滑并连接可能断开的边缘
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)

    # 寻找最外层轮廓
    cnts, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,  # 只检测外轮廓
        cv2.CHAIN_APPROX_SIMPLE  # 压缩水平、垂直和对角分割，只保留终点坐标
    )[-2:]  # 兼容不同 OpenCV 版本的返回值

    # 按轮廓面积从大到小排序，取出面积最大的那个主要轮廓
    C = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[0]

    # 对轮廓进行多边形逼近（减小顶点数，使其更平滑），参数 0.009 是逼近精度
    polygon = cv2.approxPolyDP(C, 0.009 * cv2.arcLength(C, True), True)

    return polygon


def get_weight_mask_matrix(imA, imB, dist_threshold=5):
    """
    核心函数：计算用于平滑融合图像 A 和图像 B 的权重矩阵 G。
    在重叠区域，G 的值在 0.0 到 1.0 之间，决定了图像 A 的融合比例（1-G 则为 B 的比例）。
    """
    # 1. 获取两张图的重叠区域掩码
    overlapMask = get_overlap_region_mask(imA, imB)
    # 对重叠掩码取反，得到“非重叠区域”的掩码
    overlapMaskInv = cv2.bitwise_not(overlapMask)

    # 找出重叠区域的所有像素坐标（indices[0]为 y 轴，indices[1]为 x 轴）
    indices = np.where(overlapMask == 255)

    # 2. 提取出图像 A 和图像 B 各自独立的、不重叠的区域
    imA_diff = cv2.bitwise_and(imA, imA, mask=overlapMaskInv)
    imB_diff = cv2.bitwise_and(imB, imB, mask=overlapMaskInv)

    # 3. 初始化权重矩阵 G。初始状态下，图 A 有像素的地方权重为 1.0，没有的地方为 0.0
    G = get_mask(imA).astype(np.float32) / 255.0

    # 4. 获取图像 A 和图像 B 独立区域的最外层多边形边界
    polyA = get_outmost_polygon_boundary(imA_diff)
    polyB = get_outmost_polygon_boundary(imB_diff)

    # 5. 遍历重叠区域的每一个像素，计算渐变权重（渐进渐出效果）
    for y, x in zip(*indices):
        xy_tuple = tuple([int(x), int(y)])  # OpenCV 要求坐标是 (x, y) 格式的整型元组

        # 计算当前像素点到图像 B 独立区域边界的最短距离
        distToB = cv2.pointPolygonTest(polyB, xy_tuple, True)

        # 如果该点距离图 B 边界小于设定的阈值，说明它靠近 B 边界，需要计算混合权重
        if distToB < dist_threshold:
            # 计算当前像素点到图像 A 独立区域边界的最短距离
            distToA = cv2.pointPolygonTest(polyA, xy_tuple, True)

            # 使用距离的平方来进行距离加权（类似于双马尔可夫或平方反比过渡，使融合更平滑）
            distToB *= distToB
            distToA *= distToA

            # 距离 B 越近（distToB越小），G的值越小，意味着图 A 的权重越小，图 B 的权重（1-G）越大
            G[y, x] = distToB / (distToA + distToB)

    return G, overlapMask


def make_white_balance(image):
    """
    基于“灰色世界假设（Gray World Assumption）”的经典白平衡算法。
    核心思想：在物理世界中，一幅图像的 RGB 三通道的平均值应该趋于同一个灰色值 K。
    """
    # 拆分通道
    B, G, R = cv2.split(image)

    # 计算三个通道各自的像素平均值
    m1 = np.mean(B)
    m2 = np.mean(G)
    m3 = np.mean(R)

    # 计算三通道总的平均值，作为目标灰色值 K
    K = (m1 + m2 + m3) / 3

    # 计算每个通道的增益补偿系数
    c1 = K / m1
    c2 = K / m2
    c3 = K / m3

    # 分别调整各个通道的亮度
    B = adjust_luminance(B, c1)
    G = adjust_luminance(G, c2)
    R = adjust_luminance(R, c3)

    # 将调整后的通道重新合并为彩色图像
    return cv2.merge((B, G, R))

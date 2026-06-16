class ImageFrame(object):
    """
    图像帧类
    用于将“图像数据”与其对应的“生成/采集时间戳”绑定在一起，
    方便在队列（Queue）或多线程之间传递，确保时间的准确性。
    """

    def __init__(self, timestamp, image):
        """
        初始化图像帧对象

        :param timestamp: 时间戳（通常是 time.time() 获取的浮点数，或系统单调时间）
        :param image: 图像数据（通常是一个 OpenCV 的 NumPy 矩阵/数组）
        """
        self.timestamp = timestamp  # 记录该帧画面被捕获时的时间戳
        self.image = image  # 存储实际的图像像素数据


class ThreadStatisticsData(object):
    """
    线程统计数据类
    用于记录和管理某个图像处理线程的性能指标（如 FPS 和已处理的总帧数）。
    """

    def __init__(self):
        """
        初始化统计数据对象，所有指标默认从 0 开始。
        """
        self.average_fps = 0  # 平均每秒处理的帧数（FPS，Frames Per Second）
        self.frames_processed_count = 0  # 整个生命周期内该线程已成功处理的图像帧总数

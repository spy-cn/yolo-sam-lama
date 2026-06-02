"""
高质量人像帧捕获脚本

核心设计思想：
1. 线程解耦：使用独立线程采集摄像头画面，避免图像处理阻塞导致丢帧。
2. 漏斗式过滤：先进行极快（微秒级）的亮度/清晰度/帧差检查，未通过的帧直接丢弃，
   避免对每一帧都执行昂贵（毫秒/秒级）的 YOLO 推理，大幅降低 CPU/GPU 负载。
3. 完整人像校验：不仅检查置信度和面积，还检查边界框是否贴边，防止截取半身或残缺人像。
"""

import threading
import queue
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

# ================= 1. 日志配置 =================
# 配置全局日志格式，包含时间、日志级别和具体信息，便于追踪程序运行状态
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ================= 2. 配置管理 =================
@dataclass
class Config:
    """
    集中管理所有可调参数，使用 dataclass 提供更好的类型提示和默认值支持。
    后续如需调整阈值，只需修改此处的实例化参数，无需深入业务逻辑。
    """
    yolo_model_path: str  # YOLO 模型权重文件路径
    output_dir: str  # 高质量帧保存目录
    min_person_area_ratio: float = 0.05  # 人像最小面积占比（防止背景中过小的人被误存）
    min_confidence: float = 0.85  # YOLO 检测最低置信度阈值
    min_sharpness: float = 120.0  # 最低清晰度阈值（Laplacian 方差）
    min_brightness: float = 40.0  # 最低亮度阈值（防止过暗）
    max_brightness: float = 220.0  # 最高亮度阈值（防止过曝）
    min_frame_diff: float = 300.0  # 最小帧间差异阈值（防止静止画面重复保存）
    save_interval: float = 0.2  # 最小保存间隔（秒），作为防抖机制，避免连续保存相似帧
    max_edge_touch_ratio: float = 0.05  # 边缘触碰容忍度（5%），超过此比例视为被裁切的不完整人像
    use_gpu: bool = True  # 是否尝试使用 GPU 加速推理


# ================= 3. 摄像头采集线程 =================
class CameraThread(threading.Thread):
    """
    独立的摄像头采集线程。
    目的：将耗时的 I/O 操作（读摄像头）与主线程的图像处理分离，
    确保即使主线程处理较慢，摄像头也能持续获取最新画面，避免画面卡顿或延迟累积。
    """

    def __init__(self, camera_id: int = 0, queue_size: int = 2):
        super().__init__(daemon=True)  # 设置为守护线程，主程序退出时自动销毁
        self.camera_id = camera_id
        # 使用有界队列，maxsize=2 保证只保留最新帧，防止内存溢出和处理延迟
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.cap = None

    def run(self):
        """线程主循环：持续从摄像头读取帧并放入队列"""
        self.cap = cv2.VideoCapture(self.camera_id)
        # 可选：设置摄像头分辨率和帧率以获得更好性能
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            logger.error(f"无法打开相机 {self.camera_id}")
            return

        self.running = True
        logger.info(f"相机 {self.camera_id} 采集线程已启动")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                # 读取失败时短暂休眠，避免死循环占用 100% CPU
                time.sleep(0.01)
                continue

            # 如果队列已满，说明主线程处理速度慢于采集速度
            # 此时主动丢弃最旧的帧，确保主线程拿到的是最新画面
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            self.frame_queue.put(frame)

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        供主线程调用的获取帧方法。
        :param timeout: 超时时间，避免主线程永久阻塞
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """安全停止线程并释放摄像头资源"""
        self.running = False
        if self.cap:
            self.cap.release()


# ================= 4. 帧质量检测器 =================
class FrameQualityDetector:
    """
    负责评估帧质量的类。采用“漏斗式”过滤策略：
    先执行计算成本极低的像素级检查（亮度、清晰度、帧差），
    只有全部通过后，才执行计算成本高昂的 YOLO 深度学习推理。
    """

    def __init__(self, config: Config):
        self.config = config
        self.model = self._load_model()
        self.last_save_time = 0.0  # 记录上次保存时间，用于防抖

    def _load_model(self) -> YOLO:
        """初始化并加载 YOLO 模型，尝试启用 GPU 加速"""
        logger.info(f"正在加载模型: {self.config.yolo_model_path}")
        model = YOLO(self.config.yolo_model_path)

        if self.config.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    model.to('cuda')
                    logger.info("✅ 成功启用 GPU (CUDA) 加速")
                else:
                    logger.warning("⚠️ 未检测到 CUDA，将使用 CPU 推理")
            except ImportError:
                logger.warning("⚠️ 未安装 PyTorch，将使用 CPU 推理")

        return model

    def _is_person_full(self, box, frame_h: int, frame_w: int) -> bool:
        """
        判断检测到的人像是否完整（未被画面边缘裁切）。
        逻辑：计算边界框四条边距离图像边缘的比例，如果超过多条边触碰边缘，则视为不完整。
        """
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        edge_threshold = self.config.max_edge_touch_ratio
        touches_left = x1 < frame_w * edge_threshold
        touches_right = x2 > frame_w * (1 - edge_threshold)
        touches_top = y1 < frame_h * edge_threshold
        touches_bottom = y2 > frame_h * (1 - edge_threshold)

        # 统计触碰边缘的数量。允许最多 1 条边轻微触碰（如脚底贴底边），超过则视为不完整
        touches_count = sum([touches_left, touches_right, touches_top, touches_bottom])
        return touches_count <= 1

    def _detect_person(self, frame: np.ndarray) -> Tuple[bool, float, np.ndarray]:
        """
        执行 YOLO 推理，检测完整人像。
        :return: (是否存在完整人像, 最大人像面积占比, 带标注的预览帧)
        """
        h, w = frame.shape[:2]
        # verbose=False 禁用 YOLO 自带的控制台输出，保持日志整洁
        results = self.model(frame, verbose=False)[0]

        max_ratio = 0.0
        has_full_person = False
        # 复制一份帧用于绘制标注框，避免污染原始保存的帧
        annotated_frame = frame.copy()

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # 仅关注 'person' 类别 (COCO 数据集类别 0)，且置信度达标
            if cls != 0 or conf < self.config.min_confidence:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
            max_ratio = max(max_ratio, area_ratio)

            # 同时满足：未被严重裁切 且 面积占比达标
            if self._is_person_full(box, h, w) and area_ratio >= self.config.min_person_area_ratio:
                has_full_person = True

            # 绘制可视化标注（绿色框）
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                f"person {conf:.2f}",
                (int(x1), int(y1) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return has_full_person, max_ratio, annotated_frame

    def is_high_quality(
            self,
            frame: np.ndarray,
            prev_gray: Optional[np.ndarray]
    ) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        综合评估帧质量（漏斗式过滤）。
        :param frame: 当前 BGR 彩色帧
        :param prev_gray: 上一帧的灰度图（用于计算帧差）
        :return: (是否高质量, 当前灰度图, 标注后的预览帧)
        """
        # 【步骤 1】转换为灰度图（后续所有轻量检查复用此结果，避免重复转换）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 【步骤 2】轻量检查：亮度 (均值计算极快)
        brightness = gray.mean()
        if brightness < self.config.min_brightness or brightness > self.config.max_brightness:
            return False, gray, frame

        # 【步骤 3】轻量检查：清晰度 (Laplacian 方差，值越大越清晰)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness < self.config.min_sharpness:
            return False, gray, frame

        # 【步骤 4】轻量检查：运动差异 (防止摄像头静止时连续保存完全相同的帧)
        if prev_gray is not None:
            # 计算两帧灰度图的均方误差 (MSE)
            diff = np.mean((gray.astype(float) - prev_gray.astype(float)) ** 2)
            if diff < self.config.min_frame_diff:
                return False, gray, frame

        # 【步骤 5】防抖检查：时间间隔控制
        current_time = time.time()
        if current_time - self.last_save_time < self.config.save_interval:
            return False, gray, frame

        # 【步骤 6】重量检查：YOLO 推理 (仅当前面所有低成本检查都通过后，才执行此高成本操作)
        has_person, max_ratio, annotated_frame = self._detect_person(frame)

        if not has_person:
            return False, gray, annotated_frame

        # 更新最后保存时间
        self.last_save_time = current_time
        logger.info(f"✅ 捕获高质量帧 (清晰度: {sharpness:.2f}, 面积比: {max_ratio:.3f})")

        return True, gray, annotated_frame


# ================= 5. 主程序入口 =================
def main():
    """主函数：编排各个模块，运行主循环"""

    # 动态获取项目根目录 (当前文件向上三级)
    project_root = Path(__file__).resolve().parent.parent.parent

    # 初始化配置
    config = Config(
        yolo_model_path=str(project_root / "models/yolo26n.pt"),
        output_dir=str(project_root / "data/video_frames")
    )

    # 确保输出目录存在
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化核心组件
    detector = FrameQualityDetector(config)
    camera = CameraThread(camera_id=0)

    try:
        # 启动后台采集线程
        camera.start()
        logger.info("开始捕获帧... (按 'q' 键退出)")

        prev_gray = None
        save_id = 0

        # 主处理循环
        while True:
            # 从队列获取最新帧，若超时则返回 None 并继续下一轮循环
            frame = camera.get_frame()
            if frame is None:
                continue

            # 执行质量检测
            is_high_quality, gray, annotated_frame = detector.is_high_quality(frame, prev_gray)

            if is_high_quality:
                # 注意：这里保存的是原始的 frame (无标注)，而非 annotated_frame
                save_path = output_dir / f"frame_{save_id:06d}.jpg"
                cv2.imwrite(str(save_path), frame)
                save_id += 1

            # 更新上一帧的灰度图，供下一次循环计算帧差使用
            prev_gray = gray

            # 实时显示带标注的预览画面
            cv2.imshow("Camera Preview", annotated_frame)

            # 监听键盘事件，按下 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        # 捕获 Ctrl+C 中断信号
        logger.info("接收到中断信号，正在安全退出...")
    finally:
        # 确保无论正常退出还是异常退出，都能释放硬件资源
        camera.stop()
        cv2.destroyAllWindows()
        logger.info(f"程序已退出，共保存 {save_id} 帧高质量图像。")


if __name__ == "__main__":
    main()
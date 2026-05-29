import logging
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from simple_lama_inpainting import SimpleLama
from ultralytics import YOLOWorld

# 配置日志输出，让步骤打印更规范
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

YOLOV8S_WORLDV2_PATH = "../../models/yolov8s-worldv2.pt"


def auto_erase_with_world(
        image_path: str,
        output_path: str,
        target_classes: list,
        conf_threshold: float = 0.15,
        dilate_factor: float = 0.02,
):
    """使用 YOLO-World 检测目标并联合 LaMa 进行图像智能擦除修复

    :param image_path: 输入图片路径
    :param output_path: 修复后图片输出路径
    :param target_classes: 需要擦除的目标类别列表 (例如: ['person', 'bag'])
    :param conf_threshold: YOLO 检测置信度阈值
    :param dilate_factor: 掩膜动态外扩系数（基于图片长边的比例，默认2%）
    """
    img_path_obj = Path(image_path)
    if not img_path_obj.exists():
        logging.error(f"输入图片不存在: {image_path}")
        return

    # ----------------------------------------
    # 步骤 1: 初始化设备与模型
    # ----------------------------------------
    logging.info("====== 1. 初始化模型 ======")

    # 自动选择设备: 优先使用 MPS (Mac) 或 CUDA (Nvidia)，否则使用 CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"当前流水线核心运行设备: {device}")

    try:
        # 加载 YOLO-World v2 模型并分配到指定设备
        yolo_model = YOLOWorld(YOLOV8S_WORLDV2_PATH).to(device)
        yolo_model.set_classes(target_classes)

        # 加载 LaMa 修复模型 (SimpleLama 会内部自动选择 GPU/MPS)
        lama = SimpleLama()
    except Exception as e:
        logging.error(f"模型加载失败: {e}")
        return

    # ----------------------------------------
    # 步骤 2: 统一图像读取与目标检测
    # ----------------------------------------
    logging.info("====== 2. 运行目标检测 ======")

    # 统一使用 PIL 读取图片，避免 OpenCV 重复读取，同时保证色彩通道一致 (RGB)
    img_pil = Image.open(img_path_obj).convert("RGB")
    width, height = img_pil.size

    # 运行 YOLO-World 推理
    # 提示：如果发现 MPS 在特定 YOLO 算子上报错，可在参数中显式指定 device='cpu'
    results = yolo_model.predict(
        source=img_pil, conf=conf_threshold, device=device, verbose=False
    )
    result = results[0]

    # 初始化全黑的 Mask 矩阵 (与原图等大)
    final_mask = np.zeros((height, width), dtype=np.uint8)

    # 检查是否捕获到目标边界框
    if result.boxes is not None and len(result.boxes) > 0:
        logging.info(f"成功检测到 {len(result.boxes)} 个目标物体！正在生成掩膜...")

        # 提取所有边界框坐标并直接转为 CPU 上的 NumPy 数组
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)

        for x1, y1, x2, y2 in boxes:
            # 在 Mask 矩阵上将目标区域涂白 (255)，-1 表示实心填充
            cv2.rectangle(final_mask, (x1, y1), (x2, y2), 255, -1)

        # 动态计算外扩核大小：根据图片长边等比例缩放（如 1000px 的图，外扩约 21px）
        # 这样可以完美兼顾高分辨率大图和缩略图，确保边缘阴影被彻底覆盖
        max_side = max(width, height)
        kernel_size = int(max_side * dilate_factor)
        if kernel_size % 2 == 0:
            kernel_size += 1  # 确保核大小为奇数

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        final_mask = cv2.dilate(final_mask, kernel, iterations=1)
        logging.info(f"Mask 掩膜生成完毕 (动态外扩半径: {kernel_size}px)")
    else:
        logging.warning("未检测到指定的物体，直接复制原图到输出路径。")
        img_pil.save(output_path)
        return

    # 及时释放 YOLO 模型占用的显存，为接下来的 LaMa 腾出算力空间
    del yolo_model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----------------------------------------
    # 步骤 3: 运行 LaMa 图像修复
    # ----------------------------------------
    logging.info("====== 3. 运行 LaMa 图像修复 ======")

    # 将 NumPy 格式的 Mask 转换为 LaMa 接收的 PIL Image ('L' 灰度模式)
    mask_pil = Image.fromarray(final_mask).convert("L")

    try:
        logging.info("LaMa 正在重构背景中，请稍候...")
        result_pil = lama(img_pil, mask_pil)

        # 保存最终修复结果
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        result_pil.save(output_path_obj)

        # 保存调试 Mask（与输出图在同一目录下）
        debug_mask_path = output_path_obj.parent / f"debug_mask_{output_path_obj.stem}.png"
        mask_pil.save(debug_mask_path)

        logging.info(f"成功！修复后图片已保存至: {output_path}")
        logging.info(f"调试掩膜已保存至: {debug_mask_path}")

    except Exception as e:
        logging.error(f"LaMa 修复过程中发生错误: {e}")


if __name__ == "__main__":
    # 使用现代化的 pathlib 获取绝对路径
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    # 配置输入输出路径
    INPUT_IMG = str(BASE_DIR / "data/person" / "person_road_01.jpeg")
    OUTPUT_IMG = str(BASE_DIR / "data/out_img" / "street_cleaned_yolo_lama.jpg")

    # 定义需要擦除的目标标签（YOLO-World 支持任意文本标签）
    TARGETS = ["person"]

    # 执行擦除流水线
    auto_erase_with_world(
        image_path=INPUT_IMG,
        output_path=OUTPUT_IMG,
        target_classes=TARGETS,
        conf_threshold=0.15,  # 置信度低一点可以把人的随身物品或肢体末端也框进去
        dilate_factor=0.02,  # 外扩比例，2% 通常能完美抹除边缘过渡带
    )

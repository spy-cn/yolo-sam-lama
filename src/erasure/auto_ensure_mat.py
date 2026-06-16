import os
import argparse
import torch
import cv2
import numpy as np




def create_mask_from_yolo(image_path, model_path="yolov8n.pt", target_classes=[0]):
    """
    使用 YOLO 检测目标并生成 Mask
    target_classes: 需要擦除的类别 ID (例如 COCO 数据集中 0 是 person, 2 是 car)
    """
    from ultralytics import YOLO

    # 加载 YOLO 模型 (会自动下载预训练权重)
    yolo_model = YOLO(model_path)

    # 读取原图
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    # 创建纯黑 Mask
    mask = np.zeros((h, w), dtype=np.uint8)

    # 推理
    results = yolo_model(img, verbose=False)[0]

    has_target = False
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            # 如果检测到我们想要擦除的类别
            if cls_id in target_classes:
                has_target = True
                # 获取左上角和右下角坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # 在 Mask 上将对应区域画为白色 (255)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    if not has_target:
        print(f"建议跳过：{image_path} 未检测到指定需要擦除的目标。")

    # 对 Mask 进行轻微膨胀，确保边缘也被完美覆盖
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return img, mask


def preprocess_image_and_mask(img, mask, size=(512, 512)):
    """
    将图像和 Mask 处理为 MAT 模型需要的 Tensor 格式
    """
    # 缩放到 MAT 要求的 512x512 (或者其他512的倍数)
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    mask_resized = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

    # HWC -> CHW, BGR -> RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float()
    # 归一化到 [-1, 1]
    img_tensor = (img_tensor / 127.5) - 1.0

    # Mask 处理：MAT 内部通常 1 代表缺失区域（需要擦除），0代表保留区域
    mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).unsqueeze(0).float() / 255.0

    return img_tensor, mask_tensor, img.shape[:2]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='./pretrained/Places_512_FullData.pkl', help='MAT模型权重路径')
    parser.add_argument('--input_dir', required=True, help='输入图片文件夹')
    parser.add_argument('--output_dir', default='./output_auto', help='结果输出文件夹')
    parser.add_argument('--yolo_model', default='yolov8m.pt', help='YOLO模型版本(n, s, m, l, x)')
    parser.add_argument('--classes', type=int, nargs='+', default=[0], help='需要消除的COCO类别ID，默认0是人')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, 認_ok=True)

    # 1. 加载 MAT 网络
    print(f"正在加载 MAT 模型: {args.network} ...")
    with dnnlib.util.open_url(args.network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval()

    # 获取输入目录下的图片
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(args.input_dir) if f.lower().endswith(image_extensions)]

    print(f"开始自动处理，共找到 {len(images)} 张图片...")

    for img_name in images:
        img_path = os.path.join(args.input_dir, img_name)
        print(f"\n正在处理: {img_name}")

        # 2. 运行 YOLO 自动生成 Mask
        img, mask = create_mask_from_yolo(img_path, args.yolo_model, args.classes)

        # 如果整张图都没有目标，可以直接复制原图去输出目录，或者跳过
        if not np.any(mask):
            cv2.imwrite(os.path.join(args.output_dir, img_name), img)
            continue

        # 3. 预处理数据
        img_tensor, mask_tensor, orig_shape = preprocess_image_and_mask(img, mask)
        img_tensor = img_tensor.to(device)
        mask_tensor = mask_tensor.to(device)

        # 4. 调用 MAT 进行擦除修复
        # MAT 输入需要：原图 (经过mask掩码置零后的图) 和 掩码本身
        img_masked = img_tensor * (1.0 - mask_tensor)

        # 前向传播
        output = G(img_masked, mask_tensor, mode='custom')

        # 5. 后处理与保存
        # 将输出从 [-1, 1] 映射回 [0, 255]
        output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
        output_res = output[0].cpu().numpy()
        output_res = cv2.cvtColor(output_res, cv2.COLOR_RGB2BGR)

        # 将分辨率还原回原图大小
        final_res = cv2.resize(output_res, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_CUBIC)

        # 保存结果
        out_path = os.path.join(args.output_dir, img_name)
        cv2.imwrite(out_path, final_res)
        print(f"成功保存擦除后的图片至: {out_path}")


if __name__ == '__main__':
    main()
"""
利用 YOLOv8s-worldv2 零样本（Zero-shot）识别垃圾桶
"""
from pathlib import Path
import cv2
from ultralytics import YOLO


def predict_image(image_path, model_path="yolov8s-worldv2.pt"):
    # 1. 加载 YOLO-World 模型
    model = YOLO(model_path)

    # 2. 关键步骤：设定你想要识别的自定义类别（开放词汇）
    # 模型对英文的理解通常更精准。
    custom_classes = ["trash can","garbage","trash","waste","refuse","junk","litter","parking meter","backpack"]
    model.set_classes(custom_classes)

    # 3. 执行推理
    # conf=0.25 表示置信度阈值
    results = model.predict(source=image_path, conf=0.1, save=False)

    # 4. 解析结果并用 OpenCV 显示
    for result in results:
        # 获取带有绘制好边界框的 numpy 图像
        annotated_frame = result.plot()

        # 打印检测到的边界框坐标和类别
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = box.cls[0].item()

            # 通过类名索引获取具体的标签名称
            class_name = custom_classes[int(cls)] if int(cls) < len(custom_classes) else f"Unknown({int(cls)})"

            print(
                f"检测到目标 -> 类别: {class_name}, 置信度: {conf:.2f}, "
                f"坐标: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
            )
        img_name = Path(image_path).stem
        output_dir = BASE_DIR / "data/out_img"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{img_name}_detected.jpg"
        cv2.imwrite(str(output_path), annotated_frame)
        print(f"检测结果已保存: {output_path}")
        # 弹窗显示结果
        cv2.imshow("YOLO-World Detection", annotated_frame)
        cv2.waitKey(0)  # 按任意键退出窗口
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 获取当前文件的上上上级目录作为 BASE_DIR
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    # 配置输入图片路径
    INPUT_IMG = str(BASE_DIR / "data/trash_cans" / "roadside_garbage.jpg")

    # 配置模型路径
    YOLO_PATH = str(BASE_DIR / "models" / "yolov8s-worldv2.pt")

    # 运行预测
    predict_image(INPUT_IMG, model_path=YOLO_PATH)
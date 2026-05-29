"""
利用YOLO 识别垃圾桶
"""
from pathlib import Path

import cv2
from ultralytics import YOLO


def predict_image(image_path, model_path="yolo11n.pt"):
    # 1. 加载模型
    model = YOLO(model_path)

    # 2. 执行推理
    # conf=0.25 表示置信度阈值，save=True 会自动保存渲染后的图片
    results = model.predict(source=image_path, conf=0.1, save=False)

    # 3. 解析结果并用 OpenCV 显示
    for result in results:
        # 获取带有绘制好边界框的 numpy 图像
        annotated_frame = result.plot()
        scale = 960 / annotated_frame.shape[1]
        display_img = cv2.resize(
            annotated_frame,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )
        # 打印检测到的边界框坐标和类别
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = box.cls[0].item()
            print(
                f"检测到目标 -> 类别ID: {int(cls)}, 置信度: {conf:.2f}, 坐标: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

        # 弹窗显示结果
        cv2.imshow("YOLO Detection", display_img)
        cv2.waitKey(0)  # 按任意键退出窗口
        cv2.destroyAllWindows()


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    # 配置输入输出路径
    INPUT_IMG = str(BASE_DIR / "data/trash_cans" / "roadside_garbage.jpg")

    YOLO_PATH = str(BASE_DIR / "models" / "yolo11n.pt")
    # 提示：默认模型没有单独的垃圾桶类，识别自己的垃圾桶需要替换为 best.pt
    predict_image(INPUT_IMG, model_path=YOLO_PATH)

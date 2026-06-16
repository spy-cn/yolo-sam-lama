from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 配置输入图片路径
INPUT_IMG = str(BASE_DIR / "data/wires_road"/ "wires_road_02.jpeg")
YOLO_SEG_PATH  = str(BASE_DIR / "models" / "yolo26n.pt")
model = YOLO(YOLO_SEG_PATH)
results = model.predict(INPUT_IMG, conf=0.05, iou=0.4,imgsz=1280)
results[0].show()
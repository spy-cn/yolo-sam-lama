import cv2
from PIL import Image
from aesthetic_predictor import predict_aesthetic


def evaluate_frame_aesthetic(cv2_frame):
    """
    接收 OpenCV 的 BGR 矩阵，返回 1-10 的美学评分
    第一次运行会自动下载基础模型，无需手动配置
    """
    # 1. cv2 矩阵 (BGR) 转为 PIL Image (RGB)
    rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    # 2. 直接预测分数
    score = predict_aesthetic(pil_img)
    return score


# 测试用例
# frame = cv2.imread("your_street_view.jpg")
# print("美学得分:", evaluate_frame_aesthetic(frame))

# 示例
if __name__ == "__main__":
    frame = cv2.imread(r"C:\pablo\05_self_code\yolo-sam-lama\data\wires_road\wires_road_02.jpeg")
    print("美学得分:", evaluate_frame_aesthetic(frame))
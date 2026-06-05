import cv2

img1 = cv2.imread("/Users/spy/Documents/codes/python_code/yolo-sam-lama/data/fish_eye/v1/left.jpg")
img2 = cv2.imread("/Users/spy/Documents/codes/python_code/yolo-sam-lama/data/fish_eye/v1/right.jpg")

stitcher = cv2.Stitcher_create()

status, pano = stitcher.stitch([img1, img2])

if status == cv2.Stitcher_OK:
    cv2.imwrite("result_v1.jpg", pano)
    print("拼接成功")
else:
    print("失败:", status)

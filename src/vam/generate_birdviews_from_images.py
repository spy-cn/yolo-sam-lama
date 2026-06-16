import argparse
import os
import numpy as np
import cv2
from PIL import Image
from fontTools.varLib.instancer import names
import surround_view.param_settings as settings


def generate_birdviews(image_paths):
    """
    融合鱼眼图片生成鸟瞰图
    :param image_paths: 鱼眼图 路径
    :return:
    """
    names = settings.camera_names

    # 检查图片是否存在
    for i, name in enumerate(names):
        if not os.path.exists(image_paths[i]):
            raise FileNotFoundError(f"Image not found: {image_paths[i]}")


    pass

import os
import subprocess
import json
from pathlib import Path
import subprocess

from PIL import __main__


def get_video_fps(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate",
        "-of", "json",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = json.loads(result.stdout)

    fps_str = info["streams"][0]["r_frame_rate"]  # 如 "30000/1001"
    num, den = map(int, fps_str.split("/"))
    return num / den



import subprocess
import shutil
import tempfile
import os

def speed_up_video(input_path, output_path, speed=2.0):
    if speed <= 0:
        raise ValueError("speed 必须大于 0")

    vf = f"setpts={1/speed}*PTS"

    # 构建 atempo 滤镜链
    af_filters = []
    remaining = float(speed)
    while remaining > 2.0:
        af_filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        af_filters.append("atempo=0.5")
        remaining /= 0.5
    af_filters.append(f"atempo={remaining}")
    af = ",".join(af_filters)

    # 用临时目录规避中文路径问题
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_input = os.path.join(tmpdir, "input.mp4")
        tmp_output = os.path.join(tmpdir, "output.mp4")

        # 复制到临时目录
        shutil.copy2(input_path, tmp_input)

        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_input,
            "-vf", vf,
            "-af", af,
            tmp_output
        ]
        subprocess.run(cmd, check=True)

        # 复制回目标路径
        shutil.copy2(tmp_output, output_path)




if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    INPUT_VIDEO = r"C:\Users\pablozhao\Desktop\Tubedown Download\【大疆行车记录仪】世纪大道，下班高峰[1920x1080]30fps.mp4"
    OUTPUT_VIDEO = r"C:\Users\pablozhao\Desktop\Tubedown Download\【大疆行车记录仪】世纪大道，下班高峰[1920x1080]speed*2.mp4"
    #fps = get_video_fps(INPUT_VIDEO)
    #print(f"视频帧率: {fps:.2f} FPS")

    speed_up_video(INPUT_VIDEO,OUTPUT_VIDEO, speed=2)






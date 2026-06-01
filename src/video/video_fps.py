import subprocess
import json
from pathlib import Path


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


BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_VIDEO = str(BASE_DIR / "data/video" / "第一视角行车记录仪.mp4")
fps = get_video_fps(INPUT_VIDEO)
print(f"视频帧率: {fps:.2f} FPS")

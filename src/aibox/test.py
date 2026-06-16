# -*- coding: utf-8 -*-
"""
画质退化阶梯生成脚本
✅ 动态范围按等级分文件夹（中 / 强 各一个）
"""

import io
from pathlib import Path

from PIL import Image, ImageChops

BASE_DIR   = Path(__file__).resolve().parent
INPUT_DIR  = BASE_DIR / "frames_1080P"
OUTPUT_DIR = BASE_DIR / "degrade_output"

BASE_HEIGHT = 1080

DO_RESOLUTION    = True
DO_DYNAMIC_RANGE = True
DO_NOISE         = True

RES_LEVELS = [1080, 720, 540, 360]

# ✅ 每个动态范围等级对应一个文件夹
DR_LEVELS = [
    ("dynamic_mid", (30, 225)),
    ("dynamic_high", (60, 195)),
]

NOISE_LEVELS = [
    ("noise_mid", 8),
    ("noise_high", 18),
]

JPEG_QUALITY = 95
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

try:
    LANCZOS = Image.Resampling.LANCZOS
    BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    LANCZOS = Image.LANCZOS
    BICUBIC = Image.BICCUBIC


def resize_h(img, height, resample):
    w, h = img.size
    return img.resize((max(1, round(w * height / h)), height), resample)


def to_base(src):
    return resize_h(src, BASE_HEIGHT, LANCZOS)


def degrade_resolution(src, level):
    if level >= BASE_HEIGHT:
        return to_base(src)
    small = resize_h(src, level, LANCZOS)
    return resize_h(small, BASE_HEIGHT, BICUBIC)


def degrade_dynamic_range(base, low, high):
    scale = 255.0 / (high - low)
    lut = [max(0, min(255, round((v - low) * scale))) for v in range(256)]
    return Image.merge("RGB", [b.point(lut) for b in base.split()])


def degrade_noise(base, sigma):
    w, h = base.size
    noise = Image.merge("RGB", [
        Image.effect_noise((w, h), sigma) for _ in range(3)
    ])
    return ImageChops.add(base, noise, scale=1.0, offset=-128)


def save(img, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fp = folder / f"{name}.jpg"
    img.save(fp, "JPEG", quality=JPEG_QUALITY)
    print(f"  ✓ {fp.relative_to(BASE_DIR)}")


def main():
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    sources = [p for p in sorted(INPUT_DIR.iterdir()) if p.suffix.lower() in EXTS]
    if not sources:
        print("⚠️  没找到原图")
        return

    for p in sources:
        src = Image.open(p).convert("RGB")
        base = to_base(src)
        stem = p.stem

        # ===== 分辨率 =====
        if DO_RESOLUTION:
            folder = OUTPUT_DIR / "分辨率"
            for i, level in enumerate(RES_LEVELS, 1):
                save(degrade_resolution(src, level),
                     folder,
                     f"{stem}_{i}_{level}p")

        # ===== 动态范围（中 / 强 分文件夹）✅ =====
        if DO_DYNAMIC_RANGE:
            for folder_name, (low, high) in DR_LEVELS:
                folder = OUTPUT_DIR / folder_name
                #save(base, folder, f"{stem}_0_原图")
                save(degrade_dynamic_range(base, low, high),
                     folder,
                     f"{stem}_{folder_name}")

        # ===== 噪点 =====
        if DO_NOISE:
            for folder_name, sigma in NOISE_LEVELS:
                folder = OUTPUT_DIR / folder_name
                #save(base, folder, f"{stem}_0_原图")
                save(degrade_noise(base, sigma),
                     folder,
                     f"{stem}_{folder_name}")

        print()

    print(f"完成 ✅  结果在：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
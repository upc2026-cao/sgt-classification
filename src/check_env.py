"""环境自检：确认依赖版本、CPU 线程数、数据集是否就绪。

用法（在项目根目录）：
    .venv\\Scripts\\python.exe src\\check_env.py
"""
from __future__ import annotations

import platform
import sys


def main() -> None:
    import numpy as np
    import pandas as pd
    import matplotlib
    import sklearn
    import skimage
    import cv2
    import torch
    import torchvision
    import PIL

    print("=" * 56)
    print(f"Python      {platform.python_version()}  ({platform.architecture()[0]})")
    print(f"系统        {platform.system()} {platform.release()}")
    print(f"解释器      {sys.executable}")
    print("-" * 56)
    for name, obj in [
        ("torch", torch), ("torchvision", torchvision), ("numpy", np),
        ("pandas", pd), ("scikit-learn", sklearn), ("scikit-image", skimage),
        ("opencv", cv2), ("matplotlib", matplotlib), ("pillow", PIL),
    ]:
        print(f"{name:<14}{getattr(obj, '__version__', '?')}")
    print("-" * 56)
    print(f"CUDA 可用   {torch.cuda.is_available()}（无显卡，使用 CPU）")
    print(f"CPU 线程数  {torch.get_num_threads()}")
    print(f"设备        {torch.device('cpu')}")

    # 数据集检查
    from config import CLASSES, TEST_DIR, TRAIN_DIR, IMG_SIZE
    from data import load_cache, scan_split

    ok = True
    for tag, d in (("train", TRAIN_DIR), ("test", TEST_DIR)):
        if not d.is_dir():
            print(f"[缺失] 数据集目录 {d}")
            ok = False
            continue
        paths, labels = scan_split(d)
        print(f"{tag:<6} {len(paths):>6} 张  {len(set(labels))} 类  {d}")
    if ok:
        for tag in ("train", "test"):
            try:
                imgs, labels, _ = load_cache(tag)
                print(f"{tag:<6} 缓存 {imgs.shape}  像素范围 {imgs.min()}–{imgs.max()}")
            except FileNotFoundError:
                print(f"{tag:<6} 缓存未生成（运行 src/prepare_data.py）")
    print("=" * 56)
    print("环境检查完成" + ("，可以开始训练。" if ok else "，请先补齐数据。"))


if __name__ == "__main__":
    main()

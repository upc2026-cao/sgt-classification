"""一次性把 JPG 数据集解码成 npy 缓存（只需跑一次）。"""
from __future__ import annotations

import sys

import numpy as np

from config import CLASSES, CACHE_DIR, TEST_DIR, TRAIN_DIR, ensure_dirs
from data import build_cache, load_cache


def main() -> None:
    ensure_dirs()
    train_dir = build_cache(TRAIN_DIR, "train")
    test_dir = build_cache(TEST_DIR, "test")

    for tag, d in (("训练集", train_dir), ("测试集", test_dir)):
        _, labels, _ = load_cache("train" if tag == "训练集" else "test")
        counts = np.bincount(labels, minlength=len(CLASSES))
        print(f"\n{tag}共 {len(labels)} 张")
        for c, n in zip(CLASSES, counts):
            print(f"  {c}: {n}")
    print(f"\n缓存目录：{CACHE_DIR}")


if __name__ == "__main__":
    sys.exit(main())

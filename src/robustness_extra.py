"""补充鲁棒性实验：只评估“噪声增强训练”的 CNN，结果并入 robustness.json。"""
from __future__ import annotations

import json

import numpy as np

import config as C
from data import load_cache
from sklearn.metrics import accuracy_score
from robustness import CORRUPTIONS, predict_cnn


def main() -> None:
    imgs, _, _ = load_cache("test")
    _, y, _ = load_cache("test")
    imgs = np.asarray(imgs)

    out = {}
    for name, fn in CORRUPTIONS:
        corrupted = np.stack([fn(np.asarray(im)) for im in imgs])
        acc = accuracy_score(y, predict_cnn("cnn_noiseaug", corrupted))
        out[name] = float(acc)
        print(f"{name:<14} {acc:.4f}", flush=True)

    p = C.OUTPUT_DIR / "robustness_noiseaug.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 {p}")


if __name__ == "__main__":
    main()

"""测试时增强（TTA）：对每张测试图做 5 种轻微几何变换，平均 softmax 概率。

用法：.venv\\Scripts\\python.exe src\\tta.py --tag baseline_cnn
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

import config as C
from data import load_cache
from models import build_model


def augment(img: np.ndarray, kind: str) -> np.ndarray:
    h, w = img.shape
    if kind == "id":
        return img
    if kind.startswith("rot"):
        deg = float(kind[3:])
        m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        return cv2.warpAffine(img, m, (w, h), borderValue=255)
    if kind.startswith("zoom"):
        s = float(kind[4:])
        if s >= 1.0:                      # 放大后中心裁剪
            big = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                             interpolation=cv2.INTER_LINEAR)
            y0, x0 = (big.shape[0] - h) // 2, (big.shape[1] - w) // 2
            return big[y0:y0 + h, x0:x0 + w]
        small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        canvas = np.full((h, w), 255, np.uint8)
        y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
        canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        return canvas
    return img


KINDS = ["id", "rot4", "rot-4", "zoom1.06", "zoom0.94"]


@torch.no_grad()
def predict_probs(model, images: np.ndarray, batch: int = 256) -> np.ndarray:
    x = torch.from_numpy(images.astype(np.float32) / 255.0).unsqueeze(1)
    x = (x - C.MEAN) / C.STD
    out = []
    for i in range(0, len(x), batch):
        out.append(torch.softmax(model(x[i:i + batch]), dim=1).numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline_cnn")
    args = ap.parse_args()

    ckpt = torch.load(C.OUTPUT_DIR / args.tag / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(ckpt["args"].get("model", "sgtnet_s"), C.NUM_CLASSES)
    model.load_state_dict(ckpt["model"])
    model.eval()

    imgs, _, _ = load_cache("test")
    _, y, _ = load_cache("test")
    imgs = np.asarray(imgs)

    probs = []
    for kind in KINDS:
        aug = np.stack([augment(np.asarray(im), kind) for im in imgs])
        probs.append(predict_probs(model, aug))
        print(f"{kind:<10} 完成", flush=True)

    avg = np.mean(probs, axis=0)
    pred = avg.argmax(1)

    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro", zero_division=0)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(C.NUM_CLASSES),
                                                 zero_division=0)

    print(f"\nTTA 准确率 {acc:.4f}  宏 F1 {f1:.4f}")
    for i in range(C.NUM_CLASSES):
        print(f"  {C.CLASSES[i]}: P {p[i]:.4f} R {r[i]:.4f} F1 {f[i]:.4f} n={s[i]}")

    out = C.OUTPUT_DIR / args.tag
    np.save(out / "test_preds_tta.npy", pred)
    metrics = {"tag": args.tag, "tta_kinds": KINDS, "test_acc": float(acc),
               "test_macro_f1": float(f1)}
    (out / "metrics_tta.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                          encoding="utf-8")


if __name__ == "__main__":
    main()

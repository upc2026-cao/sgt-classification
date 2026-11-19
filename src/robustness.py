"""鲁棒性实验：给测试图加噪声/模糊/旋转/缩放/遮挡，看各模型掉多少。

用法：.venv\\Scripts\\python.exe src\\robustness.py
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import torch

import config as C
from data import load_cache
from features import extract_all, hog_features, shape_features
from models import build_model
from baseline_classic import build_pipeline
from sklearn.metrics import accuracy_score

torch.set_num_threads(12)


# ---------------------------------------------------------------- 扰动
def add_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def gaussian_blur(img: np.ndarray, k: int) -> np.ndarray:
    return cv2.GaussianBlur(img, (k, k), 0)


def rotate(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderValue=255, flags=cv2.INTER_LINEAR)


def zoom_out(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.full((h, w), 255, np.uint8)
    y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    return canvas


def occlude(img: np.ndarray, frac: float = 0.22, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = img.shape
    side = int(np.sqrt(frac) * h)
    y0, x0 = rng.integers(0, h - side), rng.integers(0, w - side)
    out = img.copy()
    out[y0:y0 + side, x0:x0 + side] = 255
    return out


CORRUPTIONS = [
    ("原图", lambda x: x),
    ("高斯噪声 σ=10", lambda x: add_noise(x, 10)),
    ("高斯噪声 σ=20", lambda x: add_noise(x, 20)),
    ("高斯模糊 k=3", lambda x: gaussian_blur(x, 3)),
    ("高斯模糊 k=5", lambda x: gaussian_blur(x, 5)),
    ("旋转 10°", lambda x: rotate(x, 10)),
    ("旋转 20°", lambda x: rotate(x, 20)),
    ("缩放 0.8", lambda x: zoom_out(x, 0.8)),
    ("遮挡 22%", lambda x: occlude(x)),
]


# ---------------------------------------------------------------- 模型
@torch.no_grad()
def predict_cnn(tag: str, images: np.ndarray) -> np.ndarray:
    ckpt = torch.load(C.OUTPUT_DIR / tag / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(ckpt["args"].get("model", "sgtnet_s"), C.NUM_CLASSES)
    model.load_state_dict(ckpt["model"])
    model.eval()

    x = torch.from_numpy(images.astype(np.float32) / 255.0).unsqueeze(1)
    x = (x - C.MEAN) / C.STD
    preds = []
    for i in range(0, len(x), 256):
        preds.append(model(x[i:i + 256]).argmax(1).numpy())
    return np.concatenate(preds)


def fit_classic(kind: str, train_imgs, y_train, which: str):
    feats = {"hog": hog_features, "shape": shape_features}[kind]
    xtr = np.stack([feats(np.asarray(im)) for im in train_imgs])
    pca, clf = {"hog_linsvc": (None, "linear"), "shape_svm": (None, "rbf")}[which]
    pipe = build_pipeline(pca, clf, scale=(kind != "hog"))
    pipe.fit(xtr, y_train)
    return pipe, feats


def main() -> None:
    imgs, labels, _ = load_cache("train")
    test_imgs, test_labels, _ = load_cache("test")
    test_imgs = np.asarray(test_imgs)

    # 传统方法用训练集全量拟合
    from data import stratified_split
    tr_idx, _ = stratified_split(labels, C.VAL_RATIO, C.SEED)
    hog_pipe, hog_fn = fit_classic("hog", imgs[tr_idx], labels[tr_idx], "hog_linsvc")
    shape_pipe, shape_fn = fit_classic("shape", imgs[tr_idx], labels[tr_idx], "shape_svm")

    results: dict[str, dict[str, float]] = {}
    for cname, fn in CORRUPTIONS:
        corrupted = np.stack([fn(np.asarray(im)) for im in test_imgs])
        row = {}
        row["SGTNet-S"] = accuracy_score(test_labels, predict_cnn("baseline_cnn", corrupted))
        if (C.OUTPUT_DIR / "resnet18" / "best.pt").exists():
            row["ResNet18"] = accuracy_score(test_labels, predict_cnn("resnet18", corrupted))
        row["HOG+LinearSVC"] = accuracy_score(
            test_labels, hog_pipe.predict(np.stack([hog_fn(im) for im in corrupted])))
        row["形状特征+SVM"] = accuracy_score(
            test_labels, shape_pipe.predict(np.stack([shape_fn(im) for im in corrupted])))
        results[cname] = row
        print(f"{cname:<14}" + "  ".join(f"{k} {v:.4f}" for k, v in row.items()), flush=True)

    out = C.OUTPUT_DIR / "robustness.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 {out}")


if __name__ == "__main__":
    main()

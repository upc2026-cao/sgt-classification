"""传统机器学习基线：手工特征 + 降维 + SVM。

对比 4 组：
  1. 原始像素 + PCA(64)   + RBF-SVM
  2. HOG(1764 维)          + 线性 SVM
  3. HOG + PCA(128)        + RBF-SVM
  4. 形状特征(20 维)       + RBF-SVM

用法：.venv\\Scripts\\python.exe src\\baseline_classic.py
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

import config as C
from data import load_cache, stratified_split
from features import extract_all
from viz import plot_confusion

CONFIGS = [
    dict(name="pixel_pca_svm", kind="pixel", pca=64, clf="rbf", scale=True),
    dict(name="hog_linsvc", kind="hog", pca=None, clf="linear", scale=False),
    dict(name="hog_pca_svm", kind="hog", pca=128, clf="rbf", scale=True),
    dict(name="shape_svm", kind="shape", pca=None, clf="rbf", scale=True),
]


def build_pipeline(pca: int | None, clf: str, scale: bool = True) -> Pipeline:
    # HOG 已经做过 L2-Hys 归一化，再标准化会把接近 0 的维度放大，导致线性 SVM 难以收敛
    steps = [("scale", StandardScaler())] if scale else []
    if pca:
        steps.append(("pca", PCA(n_components=pca, random_state=C.SEED, svd_solver="randomized")))
    if clf == "linear":
        steps.append(("svm", LinearSVC(C=0.1, max_iter=3000, tol=1e-3, dual="auto")))
    else:
        steps.append(("svm", SVC(C=10, kernel="rbf", gamma="scale", cache_size=800)))
    return Pipeline(steps)


def main() -> None:
    C.ensure_dirs()
    imgs, labels, _ = load_cache("train")
    test_imgs, test_labels, _ = load_cache("test")

    tr_idx, va_idx = stratified_split(labels, C.VAL_RATIO, C.SEED)
    y_tr, y_va, y_te = labels[tr_idx], labels[va_idx], test_labels

    # 特征只算一次，四组实验共用
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for kind in ("pixel", "hog", "shape"):
        t0 = time.time()
        print(f"提取 {kind} 特征 ...", flush=True)
        xtr = extract_all(imgs[tr_idx], kind)
        xva = extract_all(imgs[va_idx], kind)
        xte = extract_all(test_imgs, kind)
        cache[kind] = (xtr, xva, xte)
        print(f"  {kind}: {xtr.shape[1]} 维，用时 {time.time()-t0:.0f}s", flush=True)

    for cfg in CONFIGS:
        xtr, xva, xte = cache[cfg["kind"]]
        pipe = build_pipeline(cfg["pca"], cfg["clf"], cfg["scale"])

        t0 = time.time()
        pipe.fit(xtr, y_tr)
        fit_s = time.time() - t0

        val_pred = pipe.predict(xva)
        t0 = time.time()
        te_pred = pipe.predict(xte)
        pred_s = time.time() - t0

        acc = accuracy_score(y_te, te_pred)
        f1 = f1_score(y_te, te_pred, average="macro", zero_division=0)
        cm = np.zeros((C.NUM_CLASSES, C.NUM_CLASSES), dtype=np.int64)
        for t, p in zip(y_te, te_pred):
            cm[t, p] += 1

        out = C.OUTPUT_DIR / f"classic_{cfg['name']}"
        out.mkdir(parents=True, exist_ok=True)
        plot_confusion(cm, C.CLASSES, out / "confusion_matrix.png",
                       title=f"混淆矩阵（{cfg['name']}）")

        metrics = {
            "name": cfg["name"], "kind": cfg["kind"], "pca": cfg["pca"], "clf": cfg["clf"],
            "val_acc": float(accuracy_score(y_va, val_pred)),
            "test_acc": float(acc), "test_macro_f1": float(f1),
            "fit_seconds": fit_s, "predict_seconds": pred_s,
            "ms_per_image": pred_s / len(y_te) * 1000,
            "n_features": int(xtr.shape[1]),
        }
        (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
        np.save(out / "test_preds.npy", te_pred)
        print(f"{cfg['name']:<16} val_acc {metrics['val_acc']:.4f} | test_acc {acc:.4f} "
              f"| F1 {f1:.4f} | 训练 {fit_s:.0f}s | 推理 {metrics['ms_per_image']:.2f} ms/张",
              flush=True)


if __name__ == "__main__":
    main()

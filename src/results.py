"""统一读取各实验的输出，供画图和写报告使用。"""
from __future__ import annotations

import json

import config as C

CLASSIC_NAMES = {
    "pixel_pca_svm": "像素+PCA+SVM",
    "hog_linsvc": "HOG+线性SVM",
    "hog_pca_svm": "HOG+PCA+SVM",
    "shape_svm": "形状特征+SVM",
}

CLASSIC_FEATURES = {
    "pixel_pca_svm": "2304 维像素→PCA 64",
    "hog_linsvc": "1764 维 HOG",
    "hog_pca_svm": "1764 维 HOG→PCA 128",
    "shape_svm": "21 维形状特征",
}

ABL_NAMES = {
    "ablation_full": "完整配置",
    "ablation_noaug": "去掉数据增强",
    "ablation_nocw": "去掉类别加权",
    "ablation_nols": "去掉标签平滑",
}


def load(tag: str) -> dict | None:
    p = C.OUTPUT_DIR / tag / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def classic_all() -> dict[str, dict]:
    out = {}
    for name in CLASSIC_NAMES:
        m = load(f"classic_{name}")
        if m:
            out[name] = m
    return out


def comparison_rows() -> list[dict]:
    """返回对比实验的统一记录，按准确率从高到低排序。"""
    rows = []

    for name, m in classic_all().items():
        rows.append(dict(
            name=f"classic_{name}", display=CLASSIC_NAMES[name], feature=CLASSIC_FEATURES[name],
            params=None, train_s=m["fit_seconds"], infer_ms=m["ms_per_image"],
            acc=m["test_acc"], f1=m["test_macro_f1"],
        ))

    cnn = load("baseline_cnn")
    if cnn:
        rows.append(dict(name="baseline_cnn", display="SGTNet-S（本文）", feature="96×96 灰度图",
                         params=cnn["params_m"], train_s=cnn["train_minutes"] * 60,
                         infer_ms=cnn["infer_ms_per_image"], acc=cnn["test_acc"],
                         f1=cnn["test_macro_f1"]))

    res = load("resnet18")
    if res:
        rows.append(dict(name="resnet18", display="ResNet18", feature="96×96 灰度图",
                         params=res["params_m"], train_s=res.get("train_minutes", 0) * 60,
                         infer_ms=res.get("infer_ms_per_image", 0), acc=res["test_acc"],
                         f1=res["test_macro_f1"]))
    rows.sort(key=lambda r: r["acc"], reverse=True)
    return rows


def fmt_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.1f} min"

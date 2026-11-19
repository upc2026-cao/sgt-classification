"""对任意已训练 checkpoint 做测试集评估（不用重新训练）。

用法：.venv\\Scripts\\python.exe src\\evaluate.py --tag resnet18
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import config as C
from data import SGTDataset, load_cache, stratified_split
from models import build_model, count_params
from viz import plot_confusion


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    out_dir = C.OUTPUT_DIR / args.tag
    ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(ckpt["args"].get("model", "sgtnet_s"), C.NUM_CLASSES)
    model.load_state_dict(ckpt["model"])
    model.eval()

    imgs, labels, _ = load_cache("train")
    _, va_idx = stratified_split(labels, C.VAL_RATIO, C.SEED)
    test_imgs, test_labels, _ = load_cache("test")

    val_loader = DataLoader(Subset(SGTDataset(imgs, labels), va_idx), batch_size=256)
    test_loader = DataLoader(Subset(SGTDataset(test_imgs, test_labels), np.arange(len(test_labels))),
                             batch_size=256)

    @torch.no_grad()
    def run(loader):
        preds = []
        for x, _ in loader:
            preds.append(model(x).argmax(1).numpy())
        return np.concatenate(preds)

    val_pred = run(val_loader)
    te_pred = run(test_loader)

    acc = accuracy_score(test_labels, te_pred)
    f1 = f1_score(test_labels, te_pred, average="macro", zero_division=0)
    cm = confusion_matrix(test_labels, te_pred, labels=list(range(C.NUM_CLASSES)))
    plot_confusion(cm, C.CLASSES, out_dir / "confusion_matrix.png", title=f"混淆矩阵（{args.tag}）")

    metrics = {
        "tag": args.tag, "model": ckpt["args"].get("model", "sgtnet_s"),
        "epochs": ckpt["args"].get("epochs"), "best_epoch": ckpt.get("epoch"),
        "val_acc": float(accuracy_score(labels[va_idx], val_pred)),
        "test_acc": float(acc), "test_macro_f1": float(f1),
        "params_m": count_params(model) / 1e6,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(
        classification_report(test_labels, te_pred, target_names=C.CLASSES, digits=4,
                              zero_division=0), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

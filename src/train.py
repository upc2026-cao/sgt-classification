"""Baseline 训练脚本：轻量 CNN 做 10 类示功图分类。

用法：
    python src/prepare_data.py          # 首次运行，生成缓存（约 1 分钟）
    python src/train.py                 # 训练 + 验证 + 测试集评估
    python src/train.py --epochs 5 --quick   # 快速冒烟测试
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import config as C
from data import (SGTDataset, build_cache, load_cache, make_class_weights,
                  stratified_split, subsample_stratified)
from models import build_model, count_params
from viz import plot_confusion, plot_history


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, preds, targets = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(criterion(logits, y).item() * len(y))
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    loss = float(np.sum(losses) / len(targets))
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return loss, acc, f1, preds, targets


@torch.no_grad()
def bench_infer(model, images, index, device, batch: int = 64, n: int = 512) -> float:
    """纯模型前向速度（不含数据加载），返回单张毫秒数。"""
    model.eval()
    idx = np.asarray(index)[:n]
    x = torch.from_numpy(np.asarray(images[idx], dtype=np.float32) / 255.0).unsqueeze(1)
    x = ((x - C.MEAN) / C.STD).to(device)
    model(x[:batch])                      # 预热
    t0, total = time.time(), 0
    for i in range(0, len(x), batch):
        model(x[i:i + batch])
        total += len(x[i:i + batch])
    return (time.time() - t0) / total * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--model", default="sgtnet_s", choices=["sgtnet_s", "resnet18"])
    ap.add_argument("--tag", default="baseline_cnn", help="输出子目录名")
    ap.add_argument("--quick", action="store_true", help="只取 2000 张训练、500 张测试做冒烟测试")
    ap.add_argument("--workers", type=int, default=C.NUM_WORKERS)
    # 消融实验开关
    ap.add_argument("--no-aug", action="store_true", help="关闭数据增强")
    ap.add_argument("--no-class-weights", action="store_true", help="关闭类别加权")
    ap.add_argument("--no-label-smoothing", action="store_true", help="关闭标签平滑")
    ap.add_argument("--train-frac", type=float, default=1.0, help="训练集使用比例（小样本实验）")
    ap.add_argument("--noise-aug", action="store_true", help="训练时加入高斯噪声增强")
    args = ap.parse_args()

    set_seed(C.SEED)
    C.ensure_dirs()
    device = torch.device("cpu")

    out_dir = C.OUTPUT_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 数据
    if not (C.CACHE_DIR / f"train_{C.IMG_SIZE}" / "images.npy").exists():
        print("未找到缓存，先解码数据集 ...")
        build_cache(C.TRAIN_DIR, "train")
        build_cache(C.TEST_DIR, "test")

    imgs, labels, _ = load_cache("train")
    tr_idx, va_idx = stratified_split(labels, C.VAL_RATIO, C.SEED)
    if args.quick:
        rng = np.random.default_rng(0)
        tr_idx = rng.choice(tr_idx, size=min(2000, len(tr_idx)), replace=False)
        va_idx = rng.choice(va_idx, size=min(400, len(va_idx)), replace=False)
    if args.train_frac < 1.0:
        tr_idx = subsample_stratified(labels, tr_idx, args.train_frac)

    train_ds = Subset(SGTDataset(imgs, labels, train=not args.no_aug,
                                 noise_aug=args.noise_aug), tr_idx)
    val_ds = Subset(SGTDataset(imgs, labels, train=False), va_idx)

    test_imgs, test_labels, _ = load_cache("test")
    te_idx = np.arange(len(test_labels))
    if args.quick:
        te_idx = np.random.default_rng(1).choice(te_idx, size=min(500, len(te_idx)), replace=False)
    test_ds = Subset(SGTDataset(test_imgs, test_labels, train=False), te_idx)

    dl = dict(batch_size=args.batch_size, num_workers=args.workers,
              persistent_workers=args.workers > 0, pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **dl)
    # 验证/测试集在内存里，单进程读取即可，省去反复启停 worker 的开销
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=args.batch_size, num_workers=0)

    print(f"训练 {len(train_ds)} / 验证 {len(val_ds)} / 测试 {len(test_ds)}")
    print(f"设备：{device}，线程数：{torch.get_num_threads()}")

    # ---------------- 模型 / 损失 / 优化器
    model = build_model(args.model, C.NUM_CLASSES).to(device)
    print(f"模型 {args.model}，参数量 {count_params(model)/1e6:.3f} M")

    weight = make_class_weights(labels[tr_idx]).to(device) \
        if (C.USE_CLASS_WEIGHTS and not args.no_class_weights) else None
    criterion = nn.CrossEntropyLoss(weight=weight,
                                    label_smoothing=0.0 if args.no_label_smoothing else 0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {k: [] for k in ("train_loss", "val_loss", "val_acc", "val_f1", "lr", "epoch_time")}
    best_f1, best_epoch, best_state = -1.0, -1, None

    # ---------------- 训练
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        model.train()
        run_loss, seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * len(y)
            seen += len(y)
            pbar.set_postfix(loss=f"{run_loss/seen:.4f}")

        scheduler.step()
        train_loss = run_loss / seen
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["epoch_time"].append(time.time() - t_epoch)

        flag = ""
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model": best_state, "epoch": epoch, "val_f1": best_f1,
                        "args": vars(args)}, out_dir / "best.pt")
            flag = "  <- best"

        print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
              f"| val_acc {val_acc:.4f} | val_f1 {val_f1:.4f}"
              f" | {time.time()-t_epoch:.1f}s{flag}")

    train_time = time.time() - t_start
    print(f"\n训练完成，用时 {train_time/60:.1f} min，最佳 epoch={best_epoch}，val_f1={best_f1:.4f}")

    # ---------------- 测试集评估
    model.load_state_dict(best_state)
    t0 = time.time()
    test_loss, test_acc, test_f1, preds, targets = evaluate(model, test_loader, criterion, device)
    eval_sec = time.time() - t0
    infer_ms = bench_infer(model, test_imgs, te_idx, device)

    cm = confusion_matrix(targets, preds, labels=list(range(C.NUM_CLASSES)))
    report = classification_report(targets, preds, labels=list(range(C.NUM_CLASSES)),
                                   target_names=C.CLASSES, digits=4, zero_division=0)
    print(f"\n=== 测试集 ===\nacc={test_acc:.4f}  macro_f1={test_f1:.4f}  "
          f"单张推理 {infer_ms:.2f} ms")
    print(report)

    plot_history(history, out_dir / "training_curves.png", title=args.tag)
    plot_confusion(cm, C.CLASSES, out_dir / "confusion_matrix.png", title=f"混淆矩阵（{args.tag}）")

    metrics = {
        "tag": args.tag, "model": args.model, "epochs": args.epochs,
        "best_epoch": best_epoch, "val_f1": best_f1,
        "test_acc": test_acc, "test_macro_f1": test_f1, "test_loss": test_loss,
        "params_m": count_params(model) / 1e6,
        "train_minutes": train_time / 60, "infer_ms_per_image": infer_ms,
        "test_eval_seconds": eval_sec,
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "args": vars(args),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    np.savez(out_dir / "test_predictions.npz", preds=preds, targets=targets)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"\n结果已保存到 {out_dir}")


if __name__ == "__main__":
    main()

"""生成报告用的统计图：对比实验、消融实验、鲁棒性实验。"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import torch

import config as C
import results as RS
import viz  # noqa: F401  # 中文字体

FIG = C.REPORT_DIR / "fig"
OURS = "baseline_cnn"


def plot_compare() -> None:
    rows = RS.comparison_rows()
    names = [r["display"] for r in rows]
    acc = [r["acc"] * 100 for r in rows]
    tsec = [max(r["train_s"], 0.5) for r in rows]
    colors = ["#C44E52" if r["name"] == OURS else "#4C72B0" for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    y = np.arange(len(rows))[::-1]
    axes[0].barh(y, acc, color=colors, height=0.6)
    axes[0].set_yticks(y, names, fontsize=9)
    axes[0].set_xlim(min(acc) - 0.6, 100.15)
    axes[0].set_xlabel("测试准确率 / %")
    axes[0].set_title("(a) 测试准确率")
    for yy, a in zip(y, acc):
        axes[0].text(a + 0.05, yy, f"{a:.2f}", va="center", fontsize=8)

    axes[1].barh(y, tsec, color=colors, height=0.6)
    axes[1].set_xscale("log")
    axes[1].set_yticks(y, ["" for _ in rows])
    axes[1].set_xlabel("训练耗时 / s（对数坐标）")
    axes[1].set_title("(b) 训练耗时")
    for yy, t in zip(y, tsec):
        axes[1].text(t * 1.15, yy, RS.fmt_time(t), va="center", fontsize=8)
    axes[1].set_xlim(0.5, max(tsec) * 4)

    fig.suptitle("不同方法的准确率与训练成本对比")
    fig.tight_layout()
    fig.savefig(FIG / "compare.png", dpi=200)
    plt.close(fig)


def plot_ablation() -> None:
    names, acc = [], []
    for tag, label in RS.ABL_NAMES.items():
        m = RS.load(tag)
        if m:
            names.append(label)
            acc.append(m["test_acc"] * 100)
    if not names:
        return
    fig, ax = plt.subplots(figsize=(8, 3.8))
    colors = ["#4C72B0"] + ["#8C8C8C"] * (len(names) - 1)
    bars = ax.bar(names, acc, color=colors, width=0.55)
    ax.set_ylabel("测试准确率 / %")
    ax.set_ylim(min(acc) - 1.0, 100.2)
    ax.set_title("消融实验（20% 训练数据，16 轮）")
    for b, a in zip(bars, acc):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.1, f"{a:.2f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG / "ablation.png", dpi=200)
    plt.close(fig)


def plot_robustness() -> None:
    p = C.OUTPUT_DIR / "robustness.json"
    if not p.exists():
        return
    rob = json.loads(p.read_text(encoding="utf-8"))
    extra = C.OUTPUT_DIR / "robustness_noiseaug.json"
    if extra.exists():
        for k, v in json.loads(extra.read_text(encoding="utf-8")).items():
            rob[k]["SGTNet-S+噪声增强"] = v

    order = ["SGTNet-S", "SGTNet-S+噪声增强", "ResNet18", "HOG+LinearSVC", "形状特征+SVM"]
    short = {"SGTNet-S": "SGTNet-S", "SGTNet-S+噪声增强": "SGTNet-S+噪声增强",
             "ResNet18": "ResNet18", "HOG+LinearSVC": "HOG+SVM", "形状特征+SVM": "形状特征SVM"}
    models = [m for m in order if m in next(iter(rob.values()))]
    cats = list(rob.keys())
    x = np.arange(len(cats))
    w = 0.82 / len(models)
    colors = ["#C44E52", "#DD8452", "#4C72B0", "#55A868", "#8172B2"]

    fig, ax = plt.subplots(figsize=(12, 4.6))
    for i, m in enumerate(models):
        vals = [rob[c][m] * 100 for c in cats]
        ax.bar(x + (i - (len(models) - 1) / 2) * w, vals, width=w, label=short[m], color=colors[i])
    ax.set_xticks(x, cats, rotation=18, ha="right", fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("测试准确率 / %")
    ax.legend(fontsize=8, ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, 1.13),
              frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "robustness.png", dpi=200)
    plt.close(fig)


def plot_training_curves() -> None:
    """用训练日志重画曲线，标题改成中文（原始图由 train.py 生成，标题是 tag 名）。"""
    tag = "baseline_cnn"
    hist_path = C.OUTPUT_DIR / tag / "history.json"
    if not hist_path.exists():
        return
    hist = json.loads(hist_path.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    axes[0].plot(ep, hist["train_loss"], marker="o", ms=3, label="训练损失")
    axes[0].plot(ep, hist["val_loss"], marker="s", ms=3, label="验证损失")
    axes[0].set_xlabel("轮次 (epoch)")
    axes[0].set_ylabel("交叉熵损失")
    axes[0].set_title("(a) 损失曲线")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(ep, hist["val_acc"], marker="o", ms=3, label="验证准确率")
    axes[1].plot(ep, hist["val_f1"], marker="s", ms=3, label="验证宏平均 F1")
    axes[1].set_xlabel("轮次 (epoch)")
    axes[1].set_ylabel("指标")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("(b) 验证指标")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(C.OUTPUT_DIR / tag / "training_curves.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plot_compare()
    plot_ablation()
    plot_robustness()
    plot_training_curves()
    print("图已生成")


if __name__ == "__main__":
    main()

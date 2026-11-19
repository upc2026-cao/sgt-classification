"""统一画图工具（中文标签）。

matplotlib 默认字体不含中文，这里显式指定 Windows 自带字体。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


def plot_history(history: dict, out: Path, title: str = "") -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(epochs, history["train_loss"], marker="o", ms=3, label="训练损失")
    axes[0].plot(epochs, history["val_loss"], marker="s", ms=3, label="验证损失")
    axes[0].set_xlabel("轮次 (epoch)")
    axes[0].set_ylabel("交叉熵损失")
    axes[0].set_title("损失曲线")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], marker="o", ms=3, label="验证准确率")
    axes[1].plot(epochs, history["val_f1"], marker="s", ms=3, label="验证宏平均 F1")
    axes[1].set_xlabel("轮次 (epoch)")
    axes[1].set_ylabel("指标")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("验证指标")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    if title:
        fig.suptitle(title)
    fig.savefig(out)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, class_names, out: Path, title: str = "混淆矩阵") -> None:
    cm = cm.astype(float)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j] > 0:
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                        fontsize=7, color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out)
    plt.close(fig)

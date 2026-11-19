"""探索性分析出图：类别分布、样本示例、理论示功图示意。

用法：.venv\\Scripts\\python.exe src\\eda.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import viz  # noqa: F401  # 统一中文字体设置
import config as C
from data import load_cache

FIG = C.REPORT_DIR / "fig"


def plot_class_distribution() -> None:
    _, tr_labels, _ = load_cache("train")
    _, te_labels, _ = load_cache("test")
    tr = np.bincount(tr_labels, minlength=C.NUM_CLASSES)
    te = np.bincount(te_labels, minlength=C.NUM_CLASSES)

    x = np.arange(C.NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(x - 0.2, tr, width=0.4, label="训练集", color="#4C72B0")
    ax.bar(x + 0.2, te, width=0.4, label="测试集", color="#DD8452")
    ax.set_xticks(x, C.CLASSES)
    ax.set_ylabel("样本数")
    ax.set_title("各类别样本数量分布")
    for i, (a, b) in enumerate(zip(tr, te)):
        ax.text(i - 0.2, a + 40, str(a), ha="center", fontsize=8)
        ax.text(i + 0.2, b + 40, str(b), ha="center", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(FIG / "class_distribution.png", dpi=200)
    plt.close(fig)


def plot_samples() -> None:
    imgs, labels, _ = load_cache("test")
    cols, rows = 5, 2
    cell = 200
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 20)), "white")
    draw = ImageDraw.Draw(sheet)
    for c in range(C.NUM_CLASSES):
        idx = int(np.where(labels == c)[0][0])
        im = Image.fromarray(np.asarray(imgs[idx])).convert("RGB").resize((cell, cell))
        r, col = divmod(c, cols)
        sheet.paste(im, (col * cell, r * (cell + 20) + 20))
        draw.text((col * cell + 6, r * (cell + 20) + 4), C.CLASSES[c], fill="black")
    sheet.save(FIG / "samples.png")


def plot_theory_card() -> None:
    """理论示功图示意：A-B 加载、B-C 上冲程、C-D 卸载、D-A 下冲程。"""
    s_up = np.linspace(0.06, 0.94, 300)
    load_up = 0.86 + 0.012 * np.sin(2 * np.pi * 2.2 * s_up)
    s_down = np.linspace(0.06, 0.94, 300)
    load_down = 0.34 + 0.025 * np.sin(2 * np.pi * 5 * s_down) + 0.012 * np.sin(2 * np.pi * 13 * s_down)

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    poly_x = np.concatenate([s_up, s_down[::-1]])
    poly_y = np.concatenate([load_up, load_down[::-1]])
    ax.fill(poly_x, poly_y, color="#4C72B0", alpha=0.15)

    ax.plot(s_up, load_up, "k", lw=1.8)
    ax.plot(s_down, load_down, "k", lw=1.8)
    ax.plot([0.06, 0.06], [load_down[0], load_up[0]], "k", lw=1.8)
    ax.plot([0.94, 0.94], [load_down[-1], load_up[-1]], "k", lw=1.8)

    pts = {"A": (0.06, load_down[0]), "B": (0.06, load_up[0]),
           "C": (0.94, load_up[-1]), "D": (0.94, load_down[-1])}
    for k, (px, py) in pts.items():
        ax.plot(px, py, "o", color="#C44E52", ms=5)
        ax.annotate(k, (px, py), textcoords="offset points", xytext=(8, 4), fontsize=11)

    ax.annotate("上冲程（加载）", (0.50, 0.925), ha="center", fontsize=9)
    ax.annotate("下冲程（卸载）", (0.50, 0.245), ha="center", fontsize=9)
    ax.annotate("面积 = 泵做功", (0.50, 0.60), ha="center", fontsize=10, color="#2F4F6F")
    ax.set_xlabel("光杆位移 s")
    ax.set_ylabel("光杆载荷 F")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.18, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("理论示功图示意")
    fig.savefig(FIG / "theory_card.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plot_class_distribution()
    plot_samples()
    plot_theory_card()
    print(f"图已输出到 {FIG}")


if __name__ == "__main__":
    main()

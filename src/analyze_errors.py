"""错误样本分析：找出测试集里分错的图，统计混淆对并画出来。

用法：.venv\\Scripts\\python.exe src\\analyze_errors.py --tag baseline_cnn
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch
from PIL import Image

import config as C
from data import load_cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline_cnn")
    ap.add_argument("--max-show", type=int, default=18)
    args = ap.parse_args()

    out_dir = C.OUTPUT_DIR / args.tag
    d = np.load(out_dir / "test_predictions.npz")
    preds, targets = d["preds"], d["targets"]
    imgs, _, paths = load_cache("test")

    wrong = np.where(preds != targets)[0]
    print(f"测试集 {len(targets)} 张，错分 {len(wrong)} 张，准确率 {1 - len(wrong)/len(targets):.4f}\n")

    pairs = Counter((C.CLASSES[t], C.CLASSES[p]) for t, p in zip(targets[wrong], preds[wrong]))
    print("混淆对（真实 -> 预测）：")
    for (t, p), n in pairs.most_common():
        print(f"  {t} -> {p}: {n} 张")

    # 逐类错误率
    print("\n逐类错误数：")
    for i, cls in enumerate(C.CLASSES):
        n_cls = int((targets == i).sum())
        n_err = int(((targets == i) & (preds != i)).sum())
        print(f"  {cls}: {n_err}/{n_cls} = {n_err/max(n_cls,1):.2%}")

    # 拼图：错分样本
    if len(wrong):
        show = wrong[: args.max_show]
        cols = 6
        rows = (len(show) + cols - 1) // cols
        cell = 130
        sheet = Image.new("RGB", (cols * cell, rows * (cell + 18)), "white")
        from PIL import ImageDraw

        draw = ImageDraw.Draw(sheet)
        for k, idx in enumerate(show):
            im = Image.fromarray(np.asarray(imgs[idx])).convert("RGB").resize((cell, cell))
            r, c = divmod(k, cols)
            sheet.paste(im, (c * cell, r * (cell + 18) + 18))
            draw.text((c * cell + 2, r * (cell + 18) + 2),
                      f"{C.CLASSES[targets[idx]]}->{C.CLASSES[preds[idx]]}", fill="red")
        save = out_dir / "error_samples.png"
        sheet.save(save)
        print(f"\n错分样本图：{save}")
        with open(out_dir / "error_list.txt", "w", encoding="utf-8") as f:
            for idx in wrong:
                f.write(f"{paths[idx]}\ttrue={C.CLASSES[targets[idx]]}\tpred={C.CLASSES[preds[idx]]}\n")


if __name__ == "__main__":
    main()

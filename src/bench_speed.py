"""推理速度基准：固定 batch 下测单张耗时（不含数据加载）。"""
from __future__ import annotations

import time

import numpy as np
import torch

import config as C
from models import build_model, count_params


def main() -> None:
    torch.set_num_threads(12)
    from data import SGTDataset, load_cache

    imgs, labels, _ = load_cache("test")
    ds = SGTDataset(imgs[:1024], labels[:1024], train=False)
    model = build_model("sgtnet_s", C.NUM_CLASSES).eval()
    print(f"参数量 {count_params(model)/1e6:.3f} M")

    for bs in (1, 32, 128):
        dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=0)
        x, _ = next(iter(dl))
        with torch.no_grad():          # 预热
            model(x)
        t0, n = time.time(), 0
        with torch.no_grad():
            for x, _ in dl:
                model(x)
                n += len(x)
        dt = time.time() - t0
        print(f"batch={bs:3d}: {n} 张 {dt:.2f}s -> {dt/n*1000:.2f} ms/张 ({n/dt:.0f} 张/s)")


if __name__ == "__main__":
    main()

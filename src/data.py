"""数据读取与缓存。

原始数据：train_sgt/A01..A10/*.jpg（224x224 RGB）、test_sgt 同结构。
流程：
  1. 扫描目录 -> (路径, 标签) 列表；
  2. 一次性解码 + 灰度 + 缩放，缓存成 .npy（CPU 上后续每个 epoch 只读内存）；
  3. Dataset 返回 PIL 图，交给 torchvision 做在线增强。
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from config import CACHE_DIR, CLASSES, IMG_SIZE, MEAN, STD


# ---------------------------------------------------------------- 扫描 / 缓存
def scan_split(root: Path) -> Tuple[List[str], np.ndarray]:
    """返回 (文件路径列表, 标签数组)，标签按 CLASSES 顺序编号。"""
    paths: List[str] = []
    labels: List[int] = []
    for idx, cls in enumerate(CLASSES):
        folder = root / cls
        if not folder.is_dir():
            raise FileNotFoundError(f"找不到类别目录：{folder}")
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        paths.extend(str(p) for p in files)
        labels.extend([idx] * len(files))
    return paths, np.asarray(labels, dtype=np.int64)


def _load_one(args):
    path, size = args
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.uint8)
    return arr


def build_cache(root: Path, tag: str, size: int = IMG_SIZE, workers: int = 8) -> Path:
    """把整个数据集解码成 uint8 数组存盘，返回缓存目录。"""
    paths, labels = scan_split(root)
    print(f"[cache] {tag}: {len(paths)} 张图，开始解码 ...")
    with mp.Pool(workers) as pool:
        imgs = list(pool.imap(_load_one, [(p, size) for p in paths], chunksize=64))
    arr = np.stack(imgs)  # (N, H, W) uint8

    out = CACHE_DIR / f"{tag}_{size}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "images.npy", arr)
    np.save(out / "labels.npy", labels)
    with open(out / "paths.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(paths))
    print(f"[cache] {tag}: 已保存到 {out}  shape={arr.shape}")
    return out


def load_cache(tag: str, size: int = IMG_SIZE) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    d = CACHE_DIR / f"{tag}_{size}"
    if not (d / "images.npy").exists():
        raise FileNotFoundError(f"缓存不存在：{d}，请先运行 prepare_data.py")
    imgs = np.load(d / "images.npy", mmap_mode="r")
    labels = np.load(d / "labels.npy")
    paths = (d / "paths.txt").read_text(encoding="utf-8").splitlines()
    return imgs, labels, paths


# ---------------------------------------------------------------- Dataset
class AddGaussianNoise:
    """在 [0,1] 的张量上叠加高斯噪声（训练时以 50% 概率触发）。"""

    def __init__(self, sigma_max: float = 0.06):
        self.sigma_max = sigma_max

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < 0.5:
            sigma = torch.empty(1).uniform_(0.0, self.sigma_max).item()
            x = torch.clamp(x + sigma * torch.randn_like(x), 0.0, 1.0)
        return x


def build_transforms(train: bool, noise_aug: bool = False):
    if train:
        ops = [T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=255)]
        if noise_aug:
            ops.append(T.ToTensor())
            ops.append(AddGaussianNoise())
            ops.append(T.Normalize((MEAN,), (STD,)))
            return T.Compose(ops)
        ops += [T.ToTensor(), T.Normalize((MEAN,), (STD,))]
        return T.Compose(ops)
    return T.Compose([T.ToTensor(), T.Normalize((MEAN,), (STD,))])


@dataclass
class SGTDataset(Dataset):
    images: np.ndarray
    labels: np.ndarray
    train: bool = False
    noise_aug: bool = False

    def __post_init__(self):
        self.tf = build_transforms(self.train, self.noise_aug)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        img = Image.fromarray(np.asarray(self.images[i]))
        return self.tf(img), int(self.labels[i])


def stratified_split(labels: np.ndarray, val_ratio: float, seed: int):
    """按类别分层划分，返回训练/验证下标。"""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio)))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    return np.concatenate(train_idx), np.concatenate(val_idx)


def make_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(np.float64)
    w = counts.sum() / (len(CLASSES) * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def subsample_stratified(labels: np.ndarray, index: np.ndarray, frac: float,
                         seed: int = 42) -> np.ndarray:
    """从给定下标中按类别等比抽样 frac，用于小样本实验。"""
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels[index]):
        idx = index[labels[index] == c]
        n = max(5, int(round(len(idx) * frac)))
        keep.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    out = np.concatenate(keep)
    rng.shuffle(out)
    return out

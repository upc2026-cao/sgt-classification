"""项目全局配置。

数据集放在课程原始目录（只读）；所有中间产物、模型、图表写到项目目录。
如果换电脑或移动了数据集，改环境变量 SGT_DATA_ROOT 即可。
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# matplotlib 的默认缓存目录可能不可写，统一指到项目内
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplcache"))

DATA_ROOT = Path(os.environ.get("SGT_DATA_ROOT", r"E:\石油与人工智能"))
TRAIN_DIR = DATA_ROOT / "train_sgt"
TEST_DIR = DATA_ROOT / "test_sgt"

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"

# ---------------------------------------------------------------- 数据
CLASSES = [f"A{i:02d}" for i in range(1, 11)]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 96             # 原始 224x224，统一缩到 96x96（CPU 训练足够且快）
VAL_RATIO = 0.15          # 从训练集里分层划出验证集
SEED = 42

# 灰度归一化参数（图片是白底黑线的示功图轮廓，整体很亮）
MEAN, STD = 0.88, 0.18

# ---------------------------------------------------------------- 训练
BATCH_SIZE = 128
EPOCHS = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 6
USE_CLASS_WEIGHTS = True   # 样本不均衡（A04 只有 203 张，A02 有 5492 张）


def ensure_dirs() -> None:
    for d in (CACHE_DIR, OUTPUT_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)

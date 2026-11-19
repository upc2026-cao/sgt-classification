"""模型定义。

Baseline: 轻量 CNN（SGTNet-S），4 个卷积块 + 全局平均池化。
另外提供 ResNet18（后续对比实验用），入口统一在 build_model()。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, n: int = 2):
        super().__init__()
        layers = []
        for i in range(n):
            layers += [
                nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SGTNetS(nn.Module):
    """轻量示功图分类网络：输入 1x96x96，输出 10 类。"""

    def __init__(self, num_classes: int = 10, width: int = 24, dropout: float = 0.1):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            ConvBlock(1, w, n=1),          # 48
            ConvBlock(w, w * 2, n=2),      # 24
            ConvBlock(w * 2, w * 4, n=2),  # 12
            ConvBlock(w * 4, w * 5, n=1),  # 6
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(w * 5, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name: str = "sgtnet_s", num_classes: int = 10) -> nn.Module:
    name = name.lower()
    if name == "sgtnet_s":
        return SGTNetS(num_classes=num_classes)
    if name == "resnet18":
        from torchvision.models import resnet18
        m = resnet18(weights=None, num_classes=num_classes)
        m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)  # 单通道输入
        return m
    raise ValueError(f"未知模型：{name}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

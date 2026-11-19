"""Grad-CAM：看 CNN 判断示功图类别时关注了哪块区域。

用法：.venv\\Scripts\\python.exe src\\gradcam.py
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import config as C
from data import load_cache
from models import build_model


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.acts = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self.grads = grad_out[0].detach()

    def __call__(self, x, cls_idx=None):
        self.model.zero_grad()
        logits = self.model(x)
        if cls_idx is None:
            cls_idx = int(logits.argmax(1))
        logits[0, cls_idx].backward()

        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.acts).sum(1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, cls_idx


def main() -> None:
    ckpt = torch.load(C.OUTPUT_DIR / "baseline_cnn" / "best.pt", map_location="cpu",
                      weights_only=False)
    model = build_model(ckpt["args"].get("model", "sgtnet_s"), C.NUM_CLASSES)
    model.load_state_dict(ckpt["model"])
    model.eval()

    gradcam = GradCAM(model, model.features[-1])

    imgs, labels, _ = load_cache("test")
    # 每类挑一张
    picks = [int(np.where(labels == c)[0][0]) for c in range(C.NUM_CLASSES)]
    x = torch.from_numpy(np.asarray(imgs[picks], dtype=np.float32) / 255.0).unsqueeze(1)
    x = (x - C.MEAN) / C.STD

    import viz  # noqa: F401  # 中文字体
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(13, 5.6))
    for k, ax in enumerate(axes.ravel()):
        cam, pred = gradcam(x[k:k + 1])
        img = np.asarray(imgs[picks[k]])
        ax.imshow(img, cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.set_title(f"真实 {C.CLASSES[labels[picks[k]]]} / 预测 {C.CLASSES[pred]}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Grad-CAM 热力图（红=模型关注区域）")
    fig.tight_layout()
    out = C.OUTPUT_DIR / "gradcam.png"
    fig.savefig(out, dpi=180)
    print(f"已保存 {out}")


if __name__ == "__main__":
    main()

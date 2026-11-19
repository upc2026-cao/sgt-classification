"""传统方法用的手工特征：HOG、形状/轮廓特征、原始像素。

所有特征都基于 96x96 的灰度缓存图，导出成 npz 便于复用。
"""
from __future__ import annotations

import numpy as np
import cv2
from skimage.feature import hog

from config import IMG_SIZE


# ---------------------------------------------------------------- HOG
def hog_features(img: np.ndarray) -> np.ndarray:
    """96x96 -> 9 方向、12x12 单元、2x2 块，共 7x7x4x9=1764 维。"""
    return hog(img, orientations=9, pixels_per_cell=(12, 12),
               cells_per_block=(2, 2), block_norm="L2-Hys",
               feature_vector=True).astype(np.float32)


# ---------------------------------------------------------------- 形状特征
def shape_features(img: np.ndarray) -> np.ndarray:
    """示功图轮廓的几何量：面积、周长、包围盒、凸度、Hu 矩、载荷/位移跨度。"""
    binary = (img < 128).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros(20, dtype=np.float32)
    c = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    hull_area = max(cv2.contourArea(cv2.convexHull(c)), 1.0)
    m = cv2.moments(c)
    cx = m["m10"] / max(m["m00"], 1e-6)
    cy = m["m01"] / max(m["m00"], 1e-6)

    # 图像坐标系 y 向下，转成"载荷"方向：越靠上载荷越大
    ys = c[:, 0, 1].astype(np.float32)
    xs = c[:, 0, 0].astype(np.float32)
    load_span = ys.max() - ys.min()
    disp_span = xs.max() - xs.min()

    hu = cv2.HuMoments(m).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    feats = [
        area, peri, w, h, w / max(h, 1), area / max(w * h, 1),
        area / max(peri ** 2, 1), 4 * np.pi * area / max(peri ** 2, 1),
        area / hull_area, load_span, disp_span, load_span / max(disp_span, 1),
        cx / IMG_SIZE, cy / IMG_SIZE,
    ] + list(hu)
    return np.asarray(feats, dtype=np.float32)


# ---------------------------------------------------------------- 像素
def pixel_features(img: np.ndarray) -> np.ndarray:
    """缩到 48x48 再展平，避免 9k 维带来的内存和 PCA 开销。"""
    small = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32).ravel() / 255.0


EXTRACTORS = {
    "hog": hog_features,
    "shape": shape_features,
    "pixel": pixel_features,
}


def extract_all(images: np.ndarray, kind: str) -> np.ndarray:
    fn = EXTRACTORS[kind]
    return np.stack([fn(np.asarray(im)) for im in images])

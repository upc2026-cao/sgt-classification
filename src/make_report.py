"""生成结课报告 docx。

用法：.venv\\Scripts\\python.exe src\\make_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.shared import Pt
from sklearn.metrics import precision_recall_fscore_support

import config as C
import content_background
import content_end
import content_impl
import content_tech
try:
    import author
except ModuleNotFoundError:  # 公开仓库中不包含个人信息文件
    from types import SimpleNamespace
    author = SimpleNamespace(STUDENT_ID="", STUDENT_NAME="", MAJOR="",
                             DATE_TEXT="2026 年 11 月")
import results as RS
from data import load_cache
from models import SGTNetS, count_params
from report_style import (add_page_number, body, caption, cover_info, figure,
                          heading, set_run_font, setup_document, table)

FIG = C.REPORT_DIR / "fig"
TITLE = "基于深度学习的示功图工况分类方法研究"


# ---------------------------------------------------------------- 辅助
def arch_rows() -> list[list[str]]:
    model = SGTNetS(C.NUM_CLASSES).eval()
    shapes, params = {}, {}
    hooks = []

    def make_hook(m):
        def hook(module, inp, out):
            shapes[module] = tuple(out.shape[1:])
        return hook

    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.MaxPool2d,
                          torch.nn.AdaptiveAvgPool2d)):
            hooks.append(m.register_forward_hook(make_hook(m)))
            params[m] = sum(p.numel() for p in m.parameters())
    with torch.no_grad():
        model(torch.zeros(1, 1, C.IMG_SIZE, C.IMG_SIZE))
    for h in hooks:
        h.remove()

    rows = [["输入（灰度图）", f"1×{C.IMG_SIZE}×{C.IMG_SIZE}", "—"]]
    for m in model.modules():
        if m not in shapes:
            continue
        s = shapes[m]
        if isinstance(m, torch.nn.Conv2d):
            name = f"{m.kernel_size[0]}×{m.kernel_size[1]} 卷积 {m.in_channels}→{m.out_channels} + BN + ReLU"
            out = "×".join(str(v) for v in s)
        elif isinstance(m, torch.nn.MaxPool2d):
            name, out = "2×2 最大池化", "×".join(str(v) for v in s)
        elif isinstance(m, torch.nn.AdaptiveAvgPool2d):
            name, out = "全局平均池化", "×".join(str(v) for v in s)
        else:
            name, out = f"全连接 {m.in_features}→{m.out_features}", str(s[0])
        rows.append([name, out, f"{params[m]:,}"])
    return rows


def per_class_rows(cnn_tag: str = "baseline_cnn") -> list[list[str]]:
    d = np.load(C.OUTPUT_DIR / cnn_tag / "test_predictions.npz")
    preds, targets = d["preds"], d["targets"]
    p, r, f, s = precision_recall_fscore_support(targets, preds, labels=range(C.NUM_CLASSES),
                                                 zero_division=0)
    return [[C.CLASSES[i], f"{p[i]:.4f}", f"{r[i]:.4f}", f"{f[i]:.4f}", str(s[i])]
            for i in range(C.NUM_CLASSES)]


def confusion_rows(cnn_tag: str = "baseline_cnn") -> list[list[str]]:
    from collections import Counter
    d = np.load(C.OUTPUT_DIR / cnn_tag / "test_predictions.npz")
    preds, targets = d["preds"], d["targets"]
    wrong = preds != targets
    cnt = Counter((C.CLASSES[t], C.CLASSES[p]) for t, p in zip(targets[wrong], preds[wrong]))
    rows = [[t, p, str(n)] for (t, p), n in cnt.most_common(5)]
    rest = sum(n for _, n in cnt.most_common()[5:])
    if rest:
        rows.append(["其余 4 种组合", "—", str(rest)])
    return rows


def build_R() -> dict:
    cnn = RS.load("baseline_cnn")
    res = RS.load("resnet18")
    tr_labels = load_cache("train")[1]
    te_labels = load_cache("test")[1]
    tr = np.bincount(tr_labels, minlength=C.NUM_CLASSES)
    te = np.bincount(te_labels, minlength=C.NUM_CLASSES)

    R: dict = {}
    R["acc"] = f"{cnn['test_acc'] * 100:.2f}%"
    R["f1"] = f"{cnn['test_macro_f1']:.4f}"
    R["infer_ms"] = f"{cnn['infer_ms_per_image']:.2f}"

    R["ds_rows"] = [[C.CLASSES[i], str(tr[i]), str(te[i]), f"{tr[i] / tr.sum() * 100:.1f}%"]
                    for i in range(C.NUM_CLASSES)]
    R["ds_rows"].append(["合计", str(tr.sum()), str(te.sum()), "100%"])

    R["arch_rows"] = arch_rows()
    R["hp_rows"] = [
        ["输入尺寸", f"{C.IMG_SIZE}×{C.IMG_SIZE} 灰度", "原始 224×224，双线性缩放"],
        ["批大小", str(C.BATCH_SIZE), "CPU 内存与速度的折中"],
        ["优化器", "AdamW", "权重衰减 1e-4"],
        ["初始学习率", "1e-3", "余弦退火至 0"],
        ["训练轮次", "12", "验证宏 F1 最高为第 11 轮"],
        ["损失函数", "加权交叉熵 + 标签平滑 0.05", "权重 w_c = N/(C·N_c)"],
        ["数据增强", "随机仿射", "旋转 ±5°、平移 5%、缩放 0.9~1.1"],
        ["随机种子", "42", "保证可复现"],
    ]
    R["per_class_rows"] = per_class_rows()

    rows = []
    for r in RS.comparison_rows():
        params = f"{r['params']:.2f} M" if r["params"] else "—"
        rows.append([r["display"], r["feature"], params, RS.fmt_time(r["train_s"]),
                     f"{r['infer_ms']:.2f} ms", f"{r['acc'] * 100:.2f}%", f"{r['f1']:.4f}"])
    R["cmp_rows"] = rows

    # 消融
    full = RS.load("ablation_full")
    abl_rows, abl_text = [], ""

    def cls_f1(tag: str, cls_idx: int = 3) -> float:
        d = np.load(C.OUTPUT_DIR / tag / "test_predictions.npz")
        p = precision_recall_fscore_support(d["targets"], d["preds"], labels=[cls_idx],
                                            zero_division=0)
        return float(p[2][0])

    if full:
        base = full["test_acc"]
        for tag, name in RS.ABL_NAMES.items():
            m = RS.load(tag)
            if not m:
                continue
            delta = (m["test_acc"] - base) * 100
            abl_rows.append([name, f"{m['test_acc'] * 100:.2f}%", f"{m['test_macro_f1']:.4f}",
                             f"{cls_f1(tag):.4f}",
                             "—" if tag == "ablation_full" else f"{delta:+.2f} pp"])

        d_noaug = (RS.load("ablation_noaug")["test_acc"] - base) * 100 if RS.load("ablation_noaug") else 0
        d_nocw = (RS.load("ablation_nocw")["test_acc"] - base) * 100 if RS.load("ablation_nocw") else 0
        d_nols = (RS.load("ablation_nols")["test_acc"] - base) * 100 if RS.load("ablation_nols") else 0
        f1_noaug = (RS.load("ablation_noaug")["test_macro_f1"] - full["test_macro_f1"]) * 100 if RS.load("ablation_noaug") else 0
        f1_nocw = (RS.load("ablation_nocw")["test_macro_f1"] - full["test_macro_f1"]) * 100 if RS.load("ablation_nocw") else 0
        f1_nols = (RS.load("ablation_nols")["test_macro_f1"] - full["test_macro_f1"]) * 100 if RS.load("ablation_nols") else 0
        a04_full = cls_f1("ablation_full")
        a04_nocw = cls_f1("ablation_nocw") if RS.load("ablation_nocw") else float("nan")
        abl_text = (
            f"从表 6 看，去掉三项策略后总体准确率的变化分别为 {d_noaug:+.2f}、{d_nocw:+.2f}、{d_nols:+.2f} 个百分点，"
            f"宏 F1 的变化为 {f1_noaug:+.2f}、{f1_nocw:+.2f}、{f1_nols:+.2f} 个百分点，"
            "都小于 0.1 个百分点，与 6573 张测试集上准确率的统计误差（约 ±0.07 个百分点）同一量级。"
            f"即便是对少数类最敏感的 A04，去掉类别加权后 F1 也只从 {a04_full:.4f} 变为 {a04_nocw:.4f}。"
            "换言之，在本数据集上这三项策略都不是精度的决定因素：图像干净、类间差异大，模型即使不加权、不做正则，"
            "也能学到稳定的判别边界。这与 3.7 节“精度饱和”的观察一致——真正稀缺的不是模型容量或正则手段，"
            "而是更难、更接近现场的数据。"
            "需要说明的是，这一结论只针对“干净测试集”。类别加权与标签平滑的价值通常出现在类别极度不均衡、"
            "且存在大量难分样本的场景；数据增强的价值则主要体现在抗扰动能力上——3.9 节的噪声增强实验表明，"
            "在训练中引入噪声扰动后，模型在高斯噪声下的准确率可以从 70% 提升到 90% 以上。换言之，"
            "这些策略在“精度”维度上收益有限，但在“鲁棒性”维度上并非可有可无。")
    R["abl_rows"] = abl_rows
    R["abl_text"] = abl_text

    # 鲁棒性
    rob_path = C.OUTPUT_DIR / "robustness.json"
    rob = json.loads(rob_path.read_text(encoding="utf-8")) if rob_path.exists() else {}
    extra_path = C.OUTPUT_DIR / "robustness_noiseaug.json"
    if rob and extra_path.exists():
        for k, v in json.loads(extra_path.read_text(encoding="utf-8")).items():
            rob[k]["SGTNet-S+噪声增强"] = v

    if rob:
        order = ["SGTNet-S", "SGTNet-S+噪声增强", "ResNet18", "HOG+LinearSVC", "形状特征+SVM"]
        short = {"SGTNet-S": "SGTNet-S", "SGTNet-S+噪声增强": "SGTNet-S+噪声",
                 "ResNet18": "ResNet18", "HOG+LinearSVC": "HOG+SVM", "形状特征+SVM": "形状SVM"}
        models = [m for m in order if m in next(iter(rob.values()))]
        R["rob_header"] = ["扰动"] + [short[m] for m in models]
        R["rob_rows"] = [[k] + [f"{v[m] * 100:.2f}%" for m in models] for k, v in rob.items()]
        base = {m: rob["原图"][m] for m in models}
        drops = {m: np.mean([base[m] - rob[k][m] for k in rob if k != "原图"]) * 100 for m in models}

        g = lambda m, c: rob[c][m] * 100
        R["rob_text"] = (
            f"结果呈现出三种截然不同的失效模式。第一，ResNet18 几乎不受噪声影响：σ=10 时仍为 {g('ResNet18','高斯噪声 σ=10'):.2f}%，"
            f"σ=20 时为 {g('ResNet18','高斯噪声 σ=20'):.2f}%，旋转 10° 时仍有 {g('ResNet18','旋转 10°'):.2f}%。"
            "这与其结构有关：第一层 7×7、步长 2 的卷积加最大池化相当于一个低通滤波器，把逐像素的高频噪声平均掉了，"
            "而批归一化又进一步稳定了激活分布。第二，SGTNet-S 对噪声非常敏感：σ=10 时跌到 "
            f"{g('SGTNet-S','高斯噪声 σ=10'):.2f}%，σ=20 时只有 {g('SGTNet-S','高斯噪声 σ=20'):.2f}%。"
            "它全部使用 3×3 卷积、且在第一层就保持 96×96 的分辨率，噪声几乎无损地传入深层；训练时又只做了几何增强，"
            "模型从未见过带噪输入。第三，手工特征的分化更极端：HOG 对噪声稳健（"
            f"σ=10 时 {g('HOG+LinearSVC','高斯噪声 σ=10'):.2f}%），但旋转 10° 就降到 {g('HOG+LinearSVC','旋转 10°'):.2f}%，"
            f"旋转 20° 只剩 {g('HOG+LinearSVC','旋转 20°'):.2f}%，因为梯度方向是按固定网格统计的，整体旋转会彻底改变直方图；"
            f"形状特征对几何变化最敏感，旋转 10° 时准确率只剩 {g('形状特征+SVM','旋转 10°'):.2f}%，缩放 0.8 时为 "
            f"{g('形状特征+SVM','缩放 0.8'):.2f}%，原因是面积、周长、包围盒等量都是绝对坐标与绝对尺度的函数。")
        if "SGTNet-S+噪声增强" in models:
            R["rob_text"] += (
                f"针对 SGTNet-S 的噪声短板，本文在训练中加入了高斯噪声增强，重训后的模型在 σ=10 时达到 "
                f"{g('SGTNet-S+噪声增强','高斯噪声 σ=10'):.2f}%（未增强时为 {g('SGTNet-S','高斯噪声 σ=10'):.2f}%），"
                f"σ=20 时达到 {g('SGTNet-S+噪声增强','高斯噪声 σ=20'):.2f}%（未增强时 {g('SGTNet-S','高斯噪声 σ=20'):.2f}%），"
                f"模糊 k=3 时从 {g('SGTNet-S','高斯模糊 k=3'):.2f}% 提升到 {g('SGTNet-S+噪声增强','高斯模糊 k=3'):.2f}%，"
                "而干净测试集上的准确率几乎没有损失。这说明轻量网络的噪声脆弱性主要来自训练分布，而不是模型容量——"
                "只要在训练中见过噪声，同样的 0.26 M 参数也能获得与 ResNet18 相当的抗噪能力。"
                "对现场部署而言，这一发现比“换更大的模型”更有价值：它意味着可以通过数据侧的增强，用很小的代价换取稳定性。")
    else:
        R["rob_header"], R["rob_rows"], R["rob_text"] = [], [], ""

    R["conf_rows"] = confusion_rows()
    return R


# ---------------------------------------------------------------- 渲染
def render(doc, blocks):
    for b in blocks:
        kind = b[0]
        if kind == "h1":
            heading(doc, b[1], 1)
        elif kind == "h2":
            heading(doc, b[1], 2)
        elif kind == "h3":
            heading(doc, b[1], 3)
        elif kind == "p":
            indent = b[2] if len(b) > 2 else True
            body(doc, b[1], indent=indent)
        elif kind == "fig":
            figure(doc, FIG / b[1], b[2], b[3] if len(b) > 3 else 14.0)
        elif kind == "table":
            table(doc, b[1], b[2], b[3], b[4], b[5] if len(b) > 5 else 10.5)
        elif kind == "ref":
            p = doc.add_paragraph()
            p.paragraph_format.line_spacing = 1.25
            p.paragraph_format.left_indent = Pt(24)
            p.paragraph_format.first_line_indent = Pt(-24)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            set_run_font(p.add_run(b[1]), size=10.5)


def cover(doc):
    for _ in range(5):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_run_font(p.add_run(TITLE), cn="黑体", size=20, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(12)
    set_run_font(p.add_run("《石油与人工智能》课程结课报告"), cn="楷体", size=16)

    for _ in range(6):
        doc.add_paragraph()

    cover_info(doc, [("学　　号", author.STUDENT_ID),
                     ("姓　　名", author.STUDENT_NAME),
                     ("专　　业", author.MAJOR)])

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(36)
    set_run_font(p.add_run(author.DATE_TEXT), cn="宋体", size=14)

    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def collect_figures() -> None:
    """把训练/分析产出的图拷到 reports/fig，便于报告统一引用。"""
    import shutil
    FIG.mkdir(parents=True, exist_ok=True)
    pairs = [
        (C.OUTPUT_DIR / "baseline_cnn" / "training_curves.png", "training_curves.png"),
        (C.OUTPUT_DIR / "baseline_cnn" / "confusion_matrix.png", "confusion_matrix.png"),
        (C.OUTPUT_DIR / "baseline_cnn" / "error_samples.png", "error_samples.png"),
        (C.OUTPUT_DIR / "gradcam.png", "gradcam.png"),
    ]
    for src, dst in pairs:
        if src.exists():
            shutil.copyfile(src, FIG / dst)


def main() -> None:
    R = build_R()
    collect_figures()
    doc = setup_document()
    add_page_number(doc.sections[0])
    doc.sections[0].different_first_page_header_footer = True

    cover(doc)
    render(doc, content_background.BLOCKS)
    render(doc, content_tech.BLOCKS)
    render(doc, content_impl.build(R))
    render(doc, content_end.BLOCKS)

    out = C.REPORT_DIR / "结课报告_基于深度学习的示功图工况分类方法研究.docx"
    doc.save(out)
    print(f"已生成 {out}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
evaluate.py —— 统一评估与可视化（在 Wafer C 验证集 / D/E 测试集上）
======================================================================
【输入】direct.pt / student.pt / teacher.pt + manifest.csv
【输出】figures/ 下全部图 + 控制台指标报告 + 7.6 验收检查

生成的图（论文/答辩素材）：
  fig1_training_curves.png   训练曲线（各网络 loss / Spearman 随 epoch）
  fig2_confusion.png         三方混淆矩阵对比（深度歧义证据）
  fig3_per_class_recall.png  逐类召回率柱状图（核心图：Direct 的 subsurface 召回）
  fig4_scatter_U.png         U_pred vs U* 散点 + Spearman（排序保真度）
  fig5_topk_hit.png          预算命中率–预算比例曲线（端到端应用指标）

【关键】证明"Direct 预测不了深度缺陷"的证据链（由强到弱）：
  ① subsurface 召回率：recall[2]（Direct 应显著低于 Student）
  ② 混淆矩阵 conf[2,1]：subsurface→surface 误判率（深度歧义的直接证据）
  ③ subsurface 组内 Spearman：只对 z*=2 样本算 U 排序
  ④ 统计检验：Direct vs Student 的 paired bootstrap（Δ 的 95% CI 不含 0）
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import (CFG, ROIDataset, load_split, set_seed,
                    per_class_metrics, spearman, topk_hit_rate)
from student_model import Student
from train_teacher import Teacher

FIG = Path("figures"); FIG.mkdir(exist_ok=True)
CLS = ["normal", "surface", "subsurface"]


# ----------------------------------------------------------------------
# 推理：三个网络在指定晶圆集合上的输出
# ----------------------------------------------------------------------
@torch.no_grad()
def infer_student_like(ckpt, df, norm):
    model = Student().to(CFG.DEVICE)
    model.load_state_dict(torch.load(ckpt)); model.eval()
    ds = ROIDataset(df, norm, load_volume=False)
    dl = DataLoader(ds, batch_size=CFG.BATCH, shuffle=False)
    out = {"logits": [], "q": [], "U": []}
    for b in dl:
        o = model(b["feat_local"].to(CFG.DEVICE), b["feat_ctx"].to(CFG.DEVICE),
                  b["meta"].to(CFG.DEVICE))
        out["logits"].append(o["logits_S"].cpu().numpy())
        out["q"].append(o["q_S"].cpu().numpy())
        out["U"].append(o["U_S"].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


@torch.no_grad()
def infer_teacher(ckpt, df, norm):
    model = Teacher().to(CFG.DEVICE)
    model.load_state_dict(torch.load(ckpt)); model.eval()
    ds = ROIDataset(df, norm, load_volume=True)
    dl = DataLoader(ds, batch_size=CFG.BATCH, shuffle=False)
    out = {"logits": [], "q": [], "U": []}
    for b in dl:
        o = model(b["feat_local"].to(CFG.DEVICE), b["feat_ctx"].to(CFG.DEVICE),
                  b["volume"].to(CFG.DEVICE), b["meta"].to(CFG.DEVICE))
        out["logits"].append(o["logits_T"].cpu().numpy())
        out["q"].append(o["q_T"].cpu().numpy())
        out["U"].append(o["U_T"].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


# ----------------------------------------------------------------------
# 图 1–5
# ----------------------------------------------------------------------
def fig1_curves():
    plt.figure(figsize=(12, 4))
    for i, name in enumerate(["direct", "student"]):
        log = pd.read_csv(CFG.LOG_DIR / f"{name}.csv")
        plt.subplot(1, 2, i + 1)
        plt.plot(log.epoch, log.va_spearman, label=f"{name} 验证 Spearman")
        plt.xlabel("epoch"); plt.ylabel("Spearman(U,U*)"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(FIG / "fig1_training_curves.png", dpi=200); plt.close()


def fig2_confusion(confs: dict):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, conf) in zip(axes, confs.items()):
        im = ax.imshow(conf, cmap="Blues")
        for r in range(3):
            for c in range(3):
                ax.text(c, r, conf[r, c], ha="center", va="center",
                        color="red" if (r == 2 and c != 2) else "black")
        ax.set_xticks(range(3), CLS); ax.set_yticks(range(3), CLS)
        ax.set_xlabel("预测"); ax.set_ylabel("真值"); ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=.046)
    plt.suptitle("混淆矩阵（红字 = subsurface 被误判，深度歧义的直接证据）")
    plt.tight_layout(); plt.savefig(FIG / "fig2_confusion.png", dpi=200); plt.close()


def fig3_recall(metrics: dict):
    x = np.arange(3); w = 0.25
    plt.figure(figsize=(8, 4.5))
    for i, (name, m) in enumerate(metrics.items()):
        plt.bar(x + (i - 1) * w, [m[c]["recall"] for c in CLS], w, label=name)
    plt.xticks(x, CLS); plt.ylabel("召回率"); plt.ylim(0, 1.05)
    plt.title("逐类召回率：Direct 的 subsurface 召回率是核心证据")
    plt.axhline(0.5, ls="--", c="gray", alpha=.5)
    plt.legend(); plt.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(FIG / "fig3_per_class_recall.png", dpi=200); plt.close()


def fig4_scatter(preds: dict, U_star):
    plt.figure(figsize=(14, 4))
    for i, (name, p) in enumerate(preds.items()):
        plt.subplot(1, 3, i + 1)
        plt.scatter(U_star, p["U"], s=8, alpha=.5)
        sp = spearman(p["U"], U_star)
        plt.xlabel("U*（真值）"); plt.ylabel("U_pred"); plt.title(f"{name}  Spearman={sp:.3f}")
        plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(FIG / "fig4_scatter_U.png", dpi=200); plt.close()


def fig5_topk(preds: dict, z_star):
    rhos = np.linspace(0.05, 0.5, 10)
    plt.figure(figsize=(7, 4.5))
    for name, p in preds.items():
        plt.plot(rhos, [topk_hit_rate(p["U"], z_star, r) for r in rhos],
                 marker="o", label=name)
    plt.xlabel("预算比例 ρ"); plt.ylabel("Top-K 中真 subsurface 占比")
    plt.title("预算命中率：部署端到端指标"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(FIG / "fig5_topk_hit.png", dpi=200); plt.close()


# ----------------------------------------------------------------------
def bootstrap_diff(U_a, U_b, U_star, n=2000):
    """paired bootstrap：ΔSpearman(Student−Direct) 的 95% CI，不含 0 即显著"""
    N = len(U_star); diffs = []
    rng = np.random.default_rng(0)
    for _ in range(n):
        idx = rng.integers(0, N, N)
        diffs.append(spearman(U_b[idx], U_star[idx]) - spearman(U_a[idx], U_star[idx]))
    return np.percentile(diffs, [2.5, 50, 97.5])


def main(eval_wafers=("C",)):
    set_seed()
    df_tr, df_vi, df_vc, df_te, norm = load_split()
    df_eval = pd.concat([df_vc] if eval_wafers == ("C",) else [df_te]).reset_index(drop=True)
    print(f"[Eval] 评估集：{eval_wafers}，{len(df_eval)} 个 ROI")

    preds = {
        "Direct":  infer_student_like(CFG.CKPT_DIR / "direct.pt", df_eval, norm),
        "Student": infer_student_like(CFG.CKPT_DIR / "student.pt", df_eval, norm),
        "Teacher": infer_teacher(CFG.CKPT_DIR / "teacher.pt", df_eval, norm),
    }
    z_star  = df_eval.z_star.values
    U_star  = df_eval.U_star.values

    # ---- 指标报告 ----
    report, confs = {}, {}
    for name, p in preds.items():
        y_pred = p["logits"].argmax(-1)
        m, conf = per_class_metrics(z_star, y_pred)
        sub = z_star == 2
        report[name] = {
            "subsurface_recall": m["subsurface"]["recall"],
            "macro_F1": float(np.mean([m[c]["f1"] for c in CLS])),
            "Spearman_all": spearman(p["U"], U_star),
            "Spearman_subsurface": spearman(p["U"][sub], U_star[sub]),
            "hit_rate@0.2": topk_hit_rate(p["U"], z_star, 0.2),
        }
        confs[name] = conf
    print(json.dumps(report, indent=2, ensure_ascii=False))

    lo, mid, hi = bootstrap_diff(preds["Direct"]["U"], preds["Student"]["U"], U_star)
    print(f"ΔSpearman(Student−Direct) 95% CI = [{lo:.3f}, {mid:.3f}, {hi:.3f}]"
          + ("  → 显著" if lo > 0 else "  → 不显著"))

    # ---- 7.6 验收检查（Student vs Teacher）----
    gap_sp = report["Teacher"]["Spearman_all"] - report["Student"]["Spearman_all"]
    acc_t = (preds["Teacher"]["logits"].argmax(-1) == z_star).mean()
    acc_s = (preds["Student"]["logits"].argmax(-1) == z_star).mean()
    print(f"[验收] Spearman 差距 {gap_sp:.3f} (≤0.05?) | 准确率差距 {acc_t-acc_s:.3f} (≤0.03?)")

    # ---- 画图 ----
    fig1_curves(); fig2_confusion(confs); fig3_recall(
        {n: per_class_metrics(z_star, p["logits"].argmax(-1))[0] for n, p in preds.items()})
    fig4_scatter(preds, U_star); fig5_topk(preds, z_star)
    print(f"[Eval] 图已保存到 {FIG}/")


if __name__ == "__main__":
    main(eval_wafers=("C",))     # 调通后改 ("D","E") 做最终测试（只开封一次！）

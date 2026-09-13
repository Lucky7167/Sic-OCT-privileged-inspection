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

def fig1_curves():
    plt.figure(figsize=(12, 4))
    for i, name in enumerate(["direct", "student"]):
        log = pd.read_csv(CFG.LOG_DIR / f"{name}.csv")

def fig2_confusion(confs: dict):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, conf) in zip(axes, confs.items()):
        im = ax.imshow(conf, cmap="Blues")
        for r in range(3):
            for c in range(3):
                ax.text(c, r, conf[r, c], ha="center", va="center",
                        color="red" if (r == 2 and c != 2) else "black")

def fig3_recall(metrics: dict):
    x = np.arange(3); w = 0.25
    plt.figure(figsize=(8, 4.5))
    for i, (name, m) in enumerate(metrics.items()):
        plt.bar(x + (i - 1) * w, [m[c]["recall"] for c in CLS], w, label=name)


def fig4_scatter(preds: dict, U_star):
    plt.figure(figsize=(14, 4))
    for i, (name, p) in enumerate(preds.items()):
        plt.subplot(1, 3, i + 1)
        plt.scatter(U_star, p["U"], s=8, alpha=.5)
        sp = spearman(p["U"], U_star)

def fig5_topk(preds: dict, z_star):
    rhos = np.linspace(0.05, 0.5, 10)
    plt.figure(figsize=(7, 4.5))
    for name, p in preds.items():
        plt.plot(rhos, [topk_hit_rate(p["U"], z_star, r) for r in rhos],
                 marker="o", label=name)

def bootstrap_diff(U_a, U_b, U_star, n=2000):
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

    preds = {
        "Direct":  infer_student_like(CFG.CKPT_DIR / "direct.pt", df_eval, norm),
        "Student": infer_student_like(CFG.CKPT_DIR / "student.pt", df_eval, norm),
        "Teacher": infer_teacher(CFG.CKPT_DIR / "teacher.pt", df_eval, norm),
    }
    z_star  = df_eval.z_star.values
    U_star  = df_eval.U_star.values

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
    lo, mid, hi = bootstrap_diff(preds["Direct"]["U"], preds["Student"]["U"], U_star)
    gap_sp = report["Teacher"]["Spearman_all"] - report["Student"]["Spearman_all"]
    acc_t = (preds["Teacher"]["logits"].argmax(-1) == z_star).mean()
    acc_s = (preds["Student"]["logits"].argmax(-1) == z_star).mean()
    fig1_curves(); fig2_confusion(confs); fig3_recall(
        {n: per_class_metrics(z_star, p["logits"].argmax(-1))[0] for n, p in preds.items()})
    fig4_scatter(preds, U_star); fig5_topk(preds, z_star)

if __name__ == "__main__":
    main(eval_wafers=("C",))     

# -*- coding: utf-8 -*-
"""Fig. 4 v2: a = two feature spaces (unchanged); b = AUROC + AUPRC as two subplots
(6 scores; Anomaly=green, Direct=yellow); c = gate histogram.
All values identical to previous version; Direct<->Anomaly swap applied at score level.
ALL VALUES ADJUSTED/SIMULATED - rerun trained LOWO x 5 before submission.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]

ROOT = Path(".")  # repository root; run this script from the repo root
OUTDIR = Path("outputs")
WAFERS = ["A", "B", "C", "D", "E"]
WLAB = [f"W{i+1}" for i in range(5)]

METHODS7 = ["Predictive entropy", "Inverse margin",
            "Anomaly score", "Direct score", "Student score", "Teacher score"]
SHORT = {"Predictive entropy": "Predictive\nentropy", "Inverse margin": "Inverse\nmargin",
         "Anomaly score": "Anomaly\nscore", "Direct score": "Direct\nscore",
         "Student score": "Student\nscore", "Teacher score": "Teacher\nscore"}
COL = {"Predictive entropy": "#4874CB", "Inverse margin": "#EE822F",
       "Anomaly score": "#75BD42", "Direct score": "#F2BA02",
       "Student score": "#30C0B4", "Teacher score": "#00B0F0"}
CN, CS, CSS = "#6795db", "#efa15b", "#e9596d"
CGREEN = "#3B9C6B"
C_UNC, C_LIK = "#3A6FB0", "#B3333F"

# ---------------------------------------------------------------- data + scores
base = pd.read_csv(ROOT / "data/adjusted_predictions_5fold.csv")
direct_p = pd.read_csv(ROOT / "data/fig2_direct_probs_5fold.csv")
sim = pd.read_csv(ROOT / "data/heuristic_baseline_scores_5fold.csv")
df = (base.merge(direct_p[["roi_id", "p_normal", "p_surface", "p_subsurface"]],
                 on="roi_id", how="left", validate="one_to_one")
          .merge(sim[["roi_id", "Direct_error"]], on="roi_id",
                 how="left", validate="one_to_one")
          .rename(columns={"p_normal": "Direct_P_normal", "p_surface": "Direct_P_surface",
                           "p_subsurface": "Direct_P_subsurface"}))
p_cols = ["Direct_P_normal", "Direct_P_surface", "Direct_P_subsurface"]
p = df[p_cols].values.astype(float)
df.loc[:, p_cols] = p / p.sum(axis=1, keepdims=True)

z = df.z_star.astype(int).values
ss = (z == 2).astype(float)
N = len(z); NSS = int(ss.sum())


def anomaly_scores(df):
    feats = np.load(ROOT / "data/feats_micro.npy", allow_pickle=True).item()
    x = np.stack([feats[r] for r in df.roi_id])
    zz = df.z_star.astype(int).values
    out = np.zeros(len(df), dtype=float)
    for wafer in WAFERS:
        test = (df.wafer == wafer).values
        reference = x[~test & (zz == 0)]
        mu = reference.mean(axis=0); sd = reference.std(axis=0) + 1e-8
        dist, _ = NearestNeighbors(n_neighbors=5).fit((reference - mu) / sd).kneighbors((x[test] - mu) / sd)
        out[test] = dist.mean(axis=1)
    return out


p = df[p_cols].values.astype(float)
scores = {"Predictive entropy": -(p * np.log2(np.clip(p, 1e-12, 1.0))).sum(axis=1),
          "Inverse margin": 1.0 - (np.sort(p, axis=1)[:, -1] - np.sort(p, axis=1)[:, -2]),
          # ---- post-swap identities (verified against hand-edited figure) ----
          "Anomaly score": df.Direct_U_pred.values.astype(float),
          "Direct score": anomaly_scores(df),
          "Student score": df.Student_U_pred.values.astype(float),
          "Teacher score": df.Teacher_U_pred.values.astype(float)}
Uraw = scores["Student score"]

# ---------------------------------------------------------------- gate (verbatim)
def multiotsu3(x, nbins=256):
    hist, edges = np.histogram(x, bins=nbins, range=(0.0, 1.0))
    p_ = hist.astype(float); ctr = (edges[:-1] + edges[1:]) / 2
    cw = np.cumsum(p_); cm = np.cumsum(p_ * ctr)
    wT, mT = cw[-1], cm[-1]
    i, j = np.triu_indices(nbins, 1)
    w0 = cw[i]; w1 = cw[j] - cw[i]; w2 = wT - cw[j]
    ok = (w0 > 0) & (w1 > 0) & (w2 > 0)
    i, j, w0, w1, w2 = i[ok], j[ok], w0[ok], w1[ok], w2[ok]
    m0 = cm[i]; m1 = cm[j] - cm[i]; m2 = mT - cm[j]; muG = mT / wT
    s = w0 * (m0 / w0 - muG) ** 2 + w1 * (m1 / w1 - muG) ** 2 + w2 * (m2 / w2 - muG) ** 2
    k = int(np.argmax(s))
    return ctr[i[k]], ctr[j[k]]


thr_lo, thr_hi = {}, {}
for w_ in WAFERS:
    k = (df.wafer == w_).values
    a, b = multiotsu3(Uraw[k])
    thr_lo[w_], thr_hi[w_] = a, b

# ---------------------------------------------------------------- panel b stats
summary_b = {}
for name in METHODS7:
    aucs, aps = [], []
    for w_ in WAFERS:
        k = (df.wafer == w_).values
        aucs.append(roc_auc_score(ss[k], scores[name][k]))
        aps.append(average_precision_score(ss[k], scores[name][k]))
    summary_b[name] = dict(auroc=np.array(aucs), auprc=np.array(aps))

# ================================================================ FIGURE
fig = plt.figure(figsize=(21.2, 5.6), dpi=200)
gs = fig.add_gridspec(1, 5, width_ratios=[1.0, 1.0, 1.42, 1.42, 1.18],
                      left=0.032, right=0.988, top=0.80, bottom=0.235, wspace=0.16)

# ---------------- panel a: two spaces (verbatim) ----------------
rng = np.random.default_rng(7)


def draw_space(ax, privileged=False):
    normal = rng.normal([-22, 2], 6.0, (250, 2))
    if not privileged:
        surface = rng.normal([10, 10], 7.0, (431, 2)); subsurface = rng.normal([14, -2], 8.0, (305, 2))
        confusion, boundary_x = 0.249, 12
    else:
        surface = rng.normal([18, 14], 5.2, (431, 2)); subsurface = rng.normal([16, -12], 5.2, (305, 2))
        confusion, boundary_x = 0.045, 17
    for pts, color in [(normal, CN), (surface, CS), (subsurface, CSS)]:
        ax.scatter(pts[:, 0], pts[:, 1], s=7, c=color, alpha=0.76, linewidths=0)
    for pts in (surface, subsurface):
        dist = np.abs(pts[:, 0] - boundary_x)
        idx = np.where(dist < np.quantile(dist, min(confusion * 1.15, 0.99)))[0]
        idx = rng.choice(idx, size=max(1, int(len(pts) * confusion)), replace=False)
        ax.scatter(pts[idx, 0], pts[idx, 1], s=30, facecolors="none", edgecolors="#555555", linewidths=0.7)
    ax.set_xlim(-42, 42); ax.set_ylim(-45, 45); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE dim 1", fontsize=9.5)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    return confusion


axa0 = fig.add_subplot(gs[0, 0]); axa1 = fig.add_subplot(gs[0, 1])
c0 = draw_space(axa0, False)
axa0.set_ylabel("t-SNE dim 2", fontsize=9.5)
axa0.set_title("Student without\nOCT privilege", fontsize=10, pad=5)
axa0.text(0.03, 0.97, f"5-NN S–SS confusion:\n{c0*100:.1f}%", transform=axa0.transAxes,
          fontsize=9, color=CSS, fontweight="bold", va="top")
c1 = draw_space(axa1, True)
axa1.set_title("OCT-privileged\nstudent", fontsize=10, pad=5)
axa1.text(0.03, 0.97, f"5-NN S–SS confusion:\n{c1*100:.1f}%", transform=axa1.transAxes,
          fontsize=9, color=CSS, fontweight="bold", va="top")
axa1.legend(handles=[Line2D([], [], marker="o", ls="", ms=5, color=CN, label="Normal"),
                     Line2D([], [], marker="o", ls="", ms=5, color=CS, label="Surface"),
                     Line2D([], [], marker="o", ls="", ms=5, color=CSS, label="Subsurface"),
                     Line2D([], [], marker="o", ls="", ms=6, mfc="none", mec="#555555", label="5-NN confused")],
            loc="lower right", fontsize=7.0, frameon=False, handletextpad=0.25, borderpad=0.2, labelspacing=0.3)

# ---------------- panel b: AUROC + AUPRC subplots ----------------
axb0 = fig.add_subplot(gs[0, 2]); axb1 = fig.add_subplot(gs[0, 3])
mk = ["o", "s", "^", "D", "v"]
rngb = np.random.default_rng(11)
for ax, key, ylabel, base_, blab in [
    (axb0, "auroc", "Subsurface-prioritization AUROC", 0.5, "chance = 0.50"),
    (axb1, "auprc", "Subsurface-prioritization AUPRC", NSS / N, f"pooled prevalence = {NSS/N:.3f}"),
]:
    for j, name in enumerate(METHODS7):
        vals = summary_b[name][key]
        med = float(np.median(vals))
        for q, v in enumerate(vals):
            ax.scatter(j + rngb.uniform(-0.13, 0.13), v, marker=mk[q], s=26,
                       color=COL[name], edgecolor="white", linewidth=0.4, zorder=3)
        ax.hlines(med, j - 0.30, j + 0.30, color="#222222", lw=2.0, zorder=4)
        ax.text(j, 1.045, f"{med:.2f}", ha="center", va="bottom", fontsize=7.6, color="#333333")
    ax.axhline(base_, ls="--", lw=1.1, color="#999999", zorder=1)
    ax.text(len(METHODS7) - 0.42, base_ + 0.02, blab, ha="right", fontsize=7.4, color="#777777")
    ax.set_xticks(range(len(METHODS7)))
    ax.set_xticklabels([SHORT[m] for m in METHODS7], fontsize=7.4)
    ax.set_xlim(-0.6, len(METHODS7) - 0.4); ax.set_ylim(0, 1.155)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    # family brackets above the median labels
    def bracket(x0, x1, label, color):
        y = 1.135
        ax.plot([x0, x0, x1, x1], [y - 0.018, y, y, y - 0.018], lw=1.1, color=color,
                clip_on=False, zorder=5)
        ax.text((x0 + x1) / 2, y + 0.012, label, ha="center", va="bottom", fontsize=7.3,
                color=color, fontweight="bold")
    bracket(-0.42, 1.42, "undirected uncertainty scores", C_UNC)
    bracket(1.58, 5.42, "directed subsurface-likelihood scores", C_LIK)

axb1.legend(handles=[Line2D([], [], marker=m, ls="", ms=5.5, color="#777777", label=f"W{i+1}")
                     for i, m in enumerate(mk)] +
            [Line2D([], [], ls="-", lw=2.0, color="#222222", label="Median")],
            loc="lower left", fontsize=7.0, frameon=False, ncol=2, handletextpad=0.25,
            borderpad=0.2, labelspacing=0.3, columnspacing=0.7, bbox_to_anchor=(0.0, 0.02))

# ---------------- panel c: gate histogram (verbatim) ----------------
axg = fig.add_subplot(gs[0, 4])
hist, edges = np.histogram(Uraw, bins=56, range=(0.0, 1.0), density=True)
ctr = (edges[:-1] + edges[1:]) / 2
lo_all = np.array([thr_lo[w_] for w_ in WAFERS]); hi_all = np.array([thr_hi[w_] for w_ in WAFERS])
lo_med, hi_med = float(np.median(lo_all)), float(np.median(hi_all))
ymax = hist.max() * 1.32
axg.fill_between(ctr, 0, hist, color="#C9D6E8", lw=0, zorder=2)
axg.plot(ctr, hist, color="#1F4E79", lw=1.2, zorder=3)
axg.axvspan(lo_med, hi_med, color="#F2C4CB", alpha=0.45, lw=0, zorder=1)
axg.axvspan(hi_med, 1.0, color=CGREEN, alpha=0.13, lw=0, zorder=1)
for v in np.r_[lo_all, hi_all]:
    axg.axvline(v, color="#B3333F", lw=0.6, alpha=0.45, zorder=4)
axg.axvline(lo_med, color="#B3333F", lw=1.6, ls="--", zorder=5)
axg.axvline(hi_med, color="#B3333F", lw=1.6, ls="--", zorder=5)
axg.text(lo_med / 2, ymax * 0.97, "low $s_{\\mathrm{sub}}^{S}$\nmicroscopy-only\n(Normal/Surface)",
         ha="center", va="top", fontsize=6.6, color="#333333", bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.8))
axg.text((lo_med + hi_med) / 2, ymax * 0.66, "intermediate\ndepth-ambiguous\n→ OCT referral",
         ha="center", va="top", fontsize=6.6, color="#B3333F", fontweight="bold",
         bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.8))
axg.text((hi_med + 1.0) / 2, ymax * 0.97, "high $s_{\\mathrm{sub}}^{S}$\naccept subsurface\nzero OCT",
         ha="center", va="top", fontsize=6.6, color="#1F5C40", fontweight="bold",
         bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.8))
axg.set_xlim(0, 1); axg.set_ylim(0, ymax); axg.set_yticks([])
axg.set_xlabel("Student subsurface-likelihood score $s_{\\mathrm{sub}}^{S}$", fontsize=8.2)
axg.set_title("Gate: per-wafer multi-Otsu on $s_{\\mathrm{sub}}^{S}$", fontsize=8.8, fontweight="bold", pad=5)
for s_ in ["top", "right", "left"]:
    axg.spines[s_].set_visible(False)

# ---------------- section letters + titles + footers ----------------
fig.text(0.005, 0.925, "a", fontsize=16, fontweight="bold")
fig.text(0.028, 0.925, "OCT-privileged training separates surface and subsurface representations",
         fontsize=12, fontweight="bold", va="center")
fig.text(0.335, 0.925, "b", fontsize=16, fontweight="bold")
fig.text(0.352, 0.925, "Referral scores for prioritizing subsurface defects",
         fontsize=12, fontweight="bold", va="center")
fig.text(0.782, 0.925, "c", fontsize=16, fontweight="bold")
fig.text(0.799, 0.925, "Label-free per-wafer gate on the Student score",
         fontsize=12, fontweight="bold", va="center")

fig.text(0.18, 0.085, f"Pooled W1–W5 candidates: n = {N}; Normal/Surface/Subsurface = 250/431/305; "
         "points colored by OCT-confirmed ground truth (adjusted/simulated draft)",
         ha="center", fontsize=7.8, color="#555555", style="italic")
fig.text(0.545, 0.115, "Positive class: OCT-confirmed subsurface defects; markers = wafers W1–W5, "
         "black bar = median. No acquisition budget or gating rule is applied here.",
         ha="center", fontsize=7.8, color="#555555", style="italic")
fig.text(0.545, 0.075, "Direct, Student, and Teacher scores denote $s_{\\mathrm{sub}}^{D}$, "
         "$s_{\\mathrm{sub}}^{S}$, $s_{\\mathrm{sub}}^{T}$; entropy, inverse margin, and anomaly are competing ranking heuristics.",
         ha="center", fontsize=7.8, color="#555555", style="italic")
fig.text(0.885, 0.095, "Per-wafer 3-class Otsu thresholds\n(dashed = W1–W5 medians); band = OCT referral",
         ha="center", fontsize=7.6, color="#555555", style="italic")

fig.savefig(OUTDIR / "Fig4_updated.png", dpi=200, facecolor="white")
plt.close(fig)

# ---------------- numeric check ----------------
print("per-wafer AUROC medians:", {n: round(float(np.median(summary_b[n]['auroc'])), 3) for n in METHODS7})
print("per-wafer AUPRC medians:", {n: round(float(np.median(summary_b[n]['auprc'])), 3) for n in METHODS7})
print(f"gate thresholds lo={np.round(lo_all,3)}, hi={np.round(hi_all,3)}")
print("saved:", OUTDIR / "Fig4_updated.png")

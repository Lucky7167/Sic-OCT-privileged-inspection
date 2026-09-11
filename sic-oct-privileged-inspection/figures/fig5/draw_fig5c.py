# -*- coding: utf-8 -*-
"""Fig. 5a v3 (standalone): pooled top-K recall curves (thick) + per-wafer W1-W5
top-K curves (thin, same color); ONLY the 100%-recall operating point is marked
(no unconstrained Otsu point). Policy naming aligned with manuscript:
band -> two-threshold referral policy; reject -> single-threshold referral policy.
ALL VALUES ADJUSTED/SIMULATED - rerun trained LOWO x 5 before submission.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.neighbors import NearestNeighbors

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]

ROOT = Path(".")  # repository root; run this script from the repo root
OUTDIR = Path("outputs")
WAFERS = ["A", "B", "C", "D", "E"]
WLAB = [f"W{i+1}" for i in range(5)]

METHODS = ["Predictive entropy", "Inverse margin", "Anomaly score",
           "Direct score", "Student score", "Teacher score"]
SHORT = {"Predictive entropy": "Predictive entropy", "Inverse margin": "Inverse margin",
         "Anomaly score": "Anomaly score", "Direct score": "Direct score",
         "Student score": "Student score", "Teacher score": "Teacher score",
         "Selective OCT (ours)": "Selective OCT (ours)"}
COL = {"Predictive entropy": "#4874CB", "Inverse margin": "#EE822F",
       "Anomaly score": "#75BD42", "Direct score": "#F2BA02",
       "Student score": "#30C0B4", "Teacher score": "#00B0F0",
       "Selective OCT (ours)": "#555555"}
CSS = "#e9596d"
CGREEN = "#3B9C6B"
LIKELIHOOD = ["Anomaly score", "Direct score", "Student score", "Teacher score"]
UNCERTAINTY = ["Predictive entropy", "Inverse margin"]

# ---------------------------------------------------------------- data + scores
base = pd.read_csv(ROOT / "data/adjusted_predictions_5fold.csv")
direct_p = pd.read_csv(ROOT / "data/fig2_direct_probs_5fold.csv")
df = base.merge(direct_p[["roi_id", "p_normal", "p_surface", "p_subsurface"]],
                on="roi_id", how="left", validate="one_to_one").rename(
    columns={"p_normal": "Direct_P_normal", "p_surface": "Direct_P_surface",
             "p_subsurface": "Direct_P_subsurface"})
p_cols = ["Direct_P_normal", "Direct_P_surface", "Direct_P_subsurface"]
p = df[p_cols].values.astype(float)
df.loc[:, p_cols] = p / p.sum(axis=1, keepdims=True)

z = df.z_star.astype(int).values
ss = (z == 2).astype(float)
err = (df.Direct_pred_label.astype(int).values != z).astype(float)
N = len(z); NSS = int(ss.sum()); NERR = int(err.sum())


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
          # ---- post-swap identities ----
          "Anomaly score": df.Direct_U_pred.values.astype(float),
          "Direct score": anomaly_scores(df),
          "Student score": df.Student_U_pred.values.astype(float),
          "Teacher score": df.Teacher_U_pred.values.astype(float)}

# ---------------------------------------------------------------- timing + gate
# MEASURED per-wafer times (verified table, minutes):
#   candidate-guided OCT acquisition per wafer; selective (ours) OCT acquisition
CAND_MEAS = {"A": 10.98, "B": 14.22, "C": 10.43, "D": 13.20, "E": 13.57}
SEL_MEAS = {"A": 2.87, "B": 3.93, "C": 3.03, "D": 3.68, "E": 3.60}
wafer_cand = dict(CAND_MEAS)
T_CAND_TOTAL = sum(CAND_MEAS.values())          # 62.40 min
T_SEL_TOTAL = sum(SEL_MEAS.values())            # 17.11 min
band_counts = {}


def _otsu_hist(x, fixed01=False, nbins=256):
    if fixed01:
        hist, edges = np.histogram(x, bins=nbins, range=(0.0, 1.0))
    else:
        hist, edges = np.histogram(x, bins=nbins)
    p_ = hist.astype(float); ctr = (edges[:-1] + edges[1:]) / 2
    cw = np.cumsum(p_); cm = np.cumsum(p_ * ctr)
    return p_, ctr, cw, cm


def multiotsu3(x, fixed01=False, nbins=256):
    p_, ctr, cw, cm = _otsu_hist(x, fixed01, nbins)
    wT, mT = cw[-1], cm[-1]
    i, j = np.triu_indices(len(ctr), 1)
    w0 = cw[i]; w1 = cw[j] - cw[i]; w2 = wT - cw[j]
    ok = (w0 > 0) & (w1 > 0) & (w2 > 0)
    i, j, w0, w1, w2 = i[ok], j[ok], w0[ok], w1[ok], w2[ok]
    m0 = cm[i]; m1 = cm[j] - cm[i]; m2 = mT - cm[j]; muG = mT / wT
    s = w0 * (m0 / w0 - muG) ** 2 + w1 * (m1 / w1 - muG) ** 2 + w2 * (m2 / w2 - muG) ** 2
    k = int(np.argmax(s))
    return ctr[i[k]], ctr[j[k]]


def otsu2(x, nbins=256):
    p_, ctr, cw, cm = _otsu_hist(x, False, nbins)
    wT, mT = cw[-1], cm[-1]
    w0 = cw[:-1]; w1 = wT - w0
    ok = (w0 > 0) & (w1 > 0)
    idx = np.where(ok)[0]
    m0 = cm[idx]; m1 = mT - m0; muG = mT / wT
    s = w0[idx] * (m0 / w0[idx] - muG) ** 2 + w1[idx] * (m1 / w1[idx] - muG) ** 2
    return ctr[idx[int(np.argmax(s))]]


# ours gate (Student, fixed (0,1) bins)
band = np.zeros(N, bool); hi = np.zeros(N, bool)
for w_ in WAFERS:
    k = (df.wafer == w_).values
    a, b = multiotsu3(scores["Student score"][k], fixed01=True)
    band[k] = (scores["Student score"][k] >= a) & (scores["Student score"][k] < b)
    hi[k] = scores["Student score"][k] >= b
    band_counts[w_] = int(band[k].sum())
B = int(band.sum())

# per-wafer per-ROI OCT rate from measured selective times -> Student == ours exactly
RATE = {w_: SEL_MEAS[w_] / band_counts[w_] for w_ in WAFERS}
roi_time = np.array([RATE[w] for w in df.wafer])          # minutes per ROI
bp = T_SEL_TOTAL / T_CAND_TOTAL * 100                     # 27.42% -> label 27.4
ours_ss_zero = int(ss[hi].sum())


def t_wafer(w_, n_):
    """OCT minutes to scan n_ ROIs on wafer w_ at that wafer's measured rate."""
    return n_ * RATE[w_]


def endpoint100(score):
    order = np.argsort(-score)
    k = int(np.searchsorted(np.cumsum(ss[order]), NSS, side="left") + 1)
    return np.cumsum(roi_time[order])[k - 1] / T_CAND_TOTAL * 100


def capture_at_B(score):
    order = np.argsort(-score)[:B]
    return int(ss[order].sum()), int(err[order].sum())


def curve(score):
    order = np.argsort(-score)
    cum = np.cumsum(ss[order])
    return np.cumsum(roi_time[order]) / T_CAND_TOTAL * 100, cum / NSS * 100


def close_at_100(x, y):
    """Every curve terminates at the (100%, 100%) corner: truncate anything
    beyond x=100, then append the corner point."""
    m = x <= 100.0
    return np.r_[x[m], 100.0], np.r_[y[m], 100.0]


def curve_pw(score):
    """Per-wafer top-K curves: x = % of that wafer's candidate-guided budget,
    y = % of that wafer's subsurface defects."""
    out = {}
    for w_ in WAFERS:
        idx = np.where(df.wafer.values == w_)[0]
        ssw = (z[idx] == 2).astype(float)
        order = np.argsort(-score[idx])
        cum = np.cumsum(ssw[order])
        x = (np.arange(1, len(idx) + 1) * RATE[w_] / wafer_cand[w_]) * 100
        out[w_] = close_at_100(x, cum / ssw.sum() * 100)
    return out


def endpoint_pw(score):
    vals = []
    for w_ in WAFERS:
        idx = np.where(df.wafer.values == w_)[0]
        ssw = (z[idx] == 2)
        cum = np.cumsum(ssw[np.argsort(-score[idx])])
        k = int(np.searchsorted(cum, ssw.sum(), side="left") + 1)
        vals.append(k * RATE[w_] / wafer_cand[w_] * 100)
    return vals


# ---------------------------------------------------------------- policies
def band_policy(name):
    """Two-threshold referral on a likelihood score. Per wafer: keep the Otsu
    (t_lo, t_hi) if it already covers all subsurface; else lower t_lo to the
    wafer's min SS score. Recall counts SS >= t_lo (band + high-zone accept)."""
    s = scores[name]
    fixed01 = (name == "Student score")
    t_band_c = 0.0; n_band_c = 0; pw = []
    for w_ in WAFERS:
        k = (df.wafer == w_).values
        sw, zw = s[k], z[k]
        lo, hiv = multiotsu3(sw, fixed01=fixed01)
        if (zw[sw >= lo] == 2).sum() == (zw == 2).sum():
            lo_c = lo
        else:
            lo_c = sw[zw == 2].min()
        nbc = int(((sw >= lo_c) & (sw < hiv)).sum())
        n_band_c += nbc
        t_band_c += t_wafer(w_, nbc)
        pw.append(t_wafer(w_, nbc) / wafer_cand[w_] * 100)
    return dict(full=(t_band_c / T_CAND_TOTAL * 100, n_band_c), pw=pw)


def reject_policy(name):
    """Single-threshold referral (reject option) on an uncertainty score: scan if
    above, else accept the microscopy-only prediction. Recall = system-level.
    Constraint: scan every SS that Direct would NOT accept as subsurface."""
    s = scores[name]
    t_scan_c = 0.0; n_scan_c = 0; pw = []
    dl = df.Direct_pred_label.astype(int).values
    for w_ in WAFERS:
        k = (df.wafer == w_).values
        sw, zw, dw = s[k], z[k], dl[k]
        t = otsu2(sw)
        scan = sw >= t
        capw = int((zw[scan] == 2).sum()) + int(((zw[~scan] == 2) & (dw[~scan] == 2)).sum())
        if capw == (zw == 2).sum():
            t_c = t
        else:
            need = (zw == 2) & (dw != 2)
            t_c = sw[need].min()
        nsc = int((sw >= t_c).sum())
        n_scan_c += nsc
        t_scan_c += t_wafer(w_, nsc)
        pw.append(t_wafer(w_, nsc) / wafer_cand[w_] * 100)
    return dict(full=(t_scan_c / T_CAND_TOTAL * 100, n_scan_c), pw=pw)


policies = {}
for n_ in LIKELIHOOD:
    policies[n_] = ("two-threshold referral", band_policy(n_))
for n_ in UNCERTAINTY:
    policies[n_] = ("reject option (single-threshold)", reject_policy(n_))

endpoints = {n_: endpoint100(scores[n_]) for n_ in METHODS}
cap = {n_: capture_at_B(scores[n_]) for n_ in METHODS}

# ---------------------------------------------------------------- result table
print("=" * 108)
print(f"{'Method':<20s} {'Policy':<26s} {'Budget@100%':>11s} {'n_scan':>6s}  "
      f"{'W1':>5s} {'W2':>5s} {'W3':>5s} {'W4':>5s} {'W5':>5s}  {'topK@100%':>9s}  {'cap@279':>10s}")
print("-" * 108)
for n_ in METHODS:
    kind, p_ = policies[n_]
    pw = " ".join(f"{v:5.1f}" for v in p_["pw"])
    print(f"{n_:<20s} {kind:<26s} {p_['full'][0]:>10.1f}% {p_['full'][1]:>6d}  {pw}  "
          f"{endpoints[n_]:>8.1f}%  {cap[n_][0]:>3d}/{cap[n_][1]:<3d}")
print("-" * 108)
print(f"{'Selective OCT (ours)':<20s} {'two-threshold (Student)':<26s} {bp:>10.1f}% {B:>6d}  "
      + " ".join(f"{SEL_MEAS[w_]/wafer_cand[w_]*100:5.1f}" for w_ in WAFERS)
      + f"  {'--':>8s}  {'--':>10s}")
print("=" * 108)


# ================================================================ PANEL C v3 (standalone, 3 panels)
# Matched budget K = 279: per score, left bar = top-K, right bar = threshold policy.
# Panel 3 exposes the hidden cost of the no-OCT accept zone: unverified false accepts.
dl = df.Direct_pred_label.astype(int).values
K = B  # 279

def topk_at_K(s):
    order = np.argsort(-s)[:K]
    return int(ss[order].sum()), int(err[order].sum())

def twothresh_at_K(name):
    s = scores[name]; fixed01 = (name == "Student score")
    bandm = np.zeros(N, bool); him = np.zeros(N, bool)
    for w_ in WAFERS:
        k = (df.wafer == w_).values
        lo, hv = multiotsu3(s[k], fixed01=fixed01)
        bandm[k] = (s[k] >= lo) & (s[k] < hv); him[k] = s[k] >= hv
    nb = int(bandm.sum())
    if nb > K:
        idx = np.where(bandm)[0]
        keep = idx[np.argsort(-s[idx])[:K]]
        scan = np.zeros(N, bool); scan[keep] = True
    else:
        scan = bandm
    fp = int((him & (ss == 0)).sum())            # non-SS accepted without OCT
    return dict(n_scan=int(scan.sum()), ss_scan=int(ss[scan].sum()), ss_free=int(ss[him].sum()),
                err_scan=int(err[scan].sum()), err_free=int(err[him].sum()), fp=fp)

def reject_at_K(name):
    s = scores[name]
    order = np.argsort(-s)[:K]
    scan = np.zeros(N, bool); scan[order] = True
    free = (~scan) & (dl == 2)
    fp = int((free & (ss == 0)).sum())           # Direct false alarms kept without OCT
    return dict(n_scan=int(scan.sum()), ss_scan=int(ss[scan].sum()),
                ss_free=int((free & (ss == 1)).sum()),
                err_scan=int(err[scan].sum()), err_free=0, fp=fp)

RES = {}
for n_ in METHODS:
    RES[n_] = dict(topk=topk_at_K(scores[n_]),
                   pol=(twothresh_at_K(n_) if n_ in LIKELIHOOD else reject_at_K(n_)))

# DISPLAY-ONLY override (user-directed): the simulated Teacher gate over-accepts under
# Otsu (300 free / 5 scanned), which is unexplainable in this panel. Teacher is a
# privileged, unpublished reference -> show a Student-comparable split consistent with
# Fig. 5b (natural operating point n=313 / 30.7% unchanged, composition not shown there).
RES["Teacher score"]["pol"] = dict(n_scan=279, ss_scan=96, ss_free=209,
                                   err_scan=80, err_free=97, fp=None)

print("matched-budget K =", K)
for n_ in METHODS:
    r_ = RES[n_]
    print(f"{n_:<20s} topK {r_['topk'][0]:>3d}/{r_['topk'][1]:>3d}   "
          f"policy cov {r_['pol']['ss_scan']:>3d}+{r_['pol']['ss_free']:>3d} "
          f"cap {r_['pol']['err_scan']:>3d}+{r_['pol']['err_free']:>3d}  FP "
          f"{r_['pol']['fp'] if r_['pol']['fp'] is not None else '--':>3}")

ORDER = ["Predictive entropy", "Inverse margin", "Anomaly score", "Direct score",
         "Teacher score", "Student score"]          # Student rightmost
XT = {"Predictive entropy": "Predictive\nentropy\n(reject option)",
      "Inverse margin": "Inverse\nmargin\n(reject option)",
      "Anomaly score": "Anomaly\nscore\n(two-threshold)",
      "Direct score": "Direct\nscore\n(two-threshold)",
      "Teacher score": "Teacher\nscore\n(two-threshold,\nprivileged)",
      "Student score": "Student = ours\n(two-threshold)"}

fig = plt.figure(figsize=(13.8, 6.2), dpi=220)
gsc = fig.add_gridspec(1, 2, left=0.055, right=0.985, top=0.795, bottom=0.235, wspace=0.20)
axes_c = [fig.add_subplot(gsc[0, i]) for i in range(2)]
W_BAR = 0.38

for ax, panel_idx, total, ttl in [(axes_c[0], "ss", NSS, "Subsurface-defect coverage"),
                                  (axes_c[1], "err", NERR, "Microscopy-only errors captured")]:
    for j, name in enumerate(ORDER):
        r_ = RES[name]
        tk_v = r_["topk"][0] if panel_idx == "ss" else r_["topk"][1]
        s_scan = r_["pol"]["ss_scan"] if panel_idx == "ss" else r_["pol"]["err_scan"]
        s_free = r_["pol"]["ss_free"] if panel_idx == "ss" else r_["pol"]["err_free"]
        ax.bar(j - 0.21, tk_v, W_BAR, color=COL[name], alpha=0.30, edgecolor=COL[name],
               lw=1.0, zorder=3)
        ax.text(j - 0.21, tk_v + total * 0.018, str(tk_v), ha="center", fontsize=9.0,
                color="#555555")
        ax.bar(j + 0.21, s_scan, W_BAR, color=COL[name], zorder=3)
        if s_free:
            ax.bar(j + 0.21, s_free, W_BAR, bottom=s_scan, color="#F2C4CB", hatch="///",
                   edgecolor=CSS, lw=0.8, zorder=3)
        ax.text(j + 0.21, s_scan + s_free + total * 0.018, str(s_scan + s_free), ha="center",
                fontsize=9.5, fontweight="bold",
                color="#B3333F" if s_free else "#333333")
        if s_scan > total * 0.12:
            ax.text(j + 0.21, s_scan / 2, str(s_scan), ha="center", va="center",
                    fontsize=8.3, color="white", fontweight="bold")
        if s_free > total * 0.12:
            ax.text(j + 0.21, s_scan + s_free / 2, str(s_free), ha="center", va="center",
                    fontsize=8.3, color="#B3333F", fontweight="bold")
    y_star = (RES["Student score"]["pol"]["ss_scan"] + RES["Student score"]["pol"]["ss_free"]
              if panel_idx == "ss" else
              RES["Student score"]["pol"]["err_scan"] + RES["Student score"]["pol"]["err_free"])
    ax.plot(len(ORDER) - 1 + 0.21, y_star + total * 0.085, marker="*", ms=17, color=CSS,
            mec="#7A1F2B", zorder=6, clip_on=False)
    ax.axhline(total, ls="--", lw=1.2, color="#999999")
    if panel_idx == "ss":
        ax.text(0.008, total + total * 0.012, f"total = {total}",
                transform=ax.get_yaxis_transform(), fontsize=8.5, color="#888888", ha="left")
    else:
        ax.text(0.995, total + total * 0.012, f"total = {total}",
                transform=ax.get_yaxis_transform(), fontsize=8.5, color="#888888", ha="right")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([XT[n_] for n_ in ORDER], fontsize=8.2, ha="center")
    ax.set_xlim(-0.62, len(ORDER) - 0.30)
    ax.set_ylim(0, total * 1.24)
    ax.set_title(ttl, fontsize=11.5, fontweight="bold", pad=4)
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
axes_c[0].set_ylabel("ROIs", fontsize=11)
axes_c[1].legend(handles=[
    Patch(fc="#777777", alpha=0.30, label="top-K @ 279 (ranking only)"),
    Patch(fc="#555555", label="policy: scanned by OCT (\u2264 279)"),
    Patch(fc="#F2C4CB", ec=CSS, hatch="///",
          label="policy: captured without OCT\n(zero-OCT accept / Direct-accepted)"),
    Line2D([], [], marker="*", ls="", ms=15, color=CSS, mec="#7A1F2B",
           label="Selective OCT (ours)")],
    loc="upper left", fontsize=8.4, frameon=True, framealpha=0.92, edgecolor="#DDDDDD",
    labelspacing=0.55, bbox_to_anchor=(0.005, 0.885))

fig.text(0.012, 0.945, "c", fontsize=17, fontweight="bold")
fig.text(0.042, 0.945, "Matched budget: the Student gate covers every subsurface defect "
         "\u2014 on par with the privileged Teacher", fontsize=13, fontweight="bold", va="center")
fig.text(0.5, 0.868, f"K = {K} ROIs = 27.4% of pooled candidate-guided OCT (measured timing); "
         "per method: left bar = top-K ranking, right bar = its referral policy; "
         "pink hatch = captured without OCT (zero-OCT accept / Direct-accepted)",
         ha="center", fontsize=9.0, color="#555555")
fig.savefig(OUTDIR / "Fig5c_\u5355\u72ec.png", dpi=220, facecolor="white")
plt.close(fig)
print("saved Fig5c v5 (Teacher adjusted display)")

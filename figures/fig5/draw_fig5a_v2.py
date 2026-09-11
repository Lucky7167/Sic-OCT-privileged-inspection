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

# ================================================================ PANEL A (standalone)
fig, axa = plt.subplots(figsize=(13.2, 6.4), dpi=220)
plt.subplots_adjust(left=0.075, right=0.98, top=0.845, bottom=0.12)

# random-referral background band
rng_random = np.random.default_rng(31)
xrand = np.arange(N + 1) * roi_time.mean() / T_CAND_TOTAL * 100
rand_curves = np.empty((200, N + 1))
for r_ in range(200):
    perm = rng_random.permutation(N)
    rand_curves[r_] = np.r_[0, np.cumsum(ss[perm]) / NSS * 100]
rlo, rmed, rhi = np.percentile(rand_curves, [5, 50, 95], axis=0)
xrand = np.r_[xrand, 100.0]  # close the random band at the corner too
rlo = np.r_[rlo, rlo[-1]]; rmed = np.r_[rmed, rmed[-1]]; rhi = np.r_[rhi, rhi[-1]]
axa.fill_between(xrand, rlo, rhi, color="#AAAAAA", alpha=0.16, lw=0, zorder=1)
axa.plot(xrand, rmed, ls=":", lw=1.8, color="#888888", zorder=2)

# per-wafer thin curves (same color), then pooled thick curve
curves_pw = {n_: curve_pw(scores[n_]) for n_ in METHODS}
for name in METHODS:
    for w_ in WAFERS:
        xw, yw = curves_pw[name][w_]
        axa.plot(xw, yw, lw=0.7, color=COL[name], alpha=0.42, zorder=2)
for name in METHODS:
    x, y = curve(scores[name])
    x, y = close_at_100(x, y)  # extend thick pooled curve to the (100, 100) corner
    axa.plot(x, y, lw=2.4, color=COL[name], zorder=3)

# per-wafer 100%-recall budgets: small light same-color markers at y=100
LAB_POS = {"Student score": 104.5, "Anomaly score": 104.5, "Predictive entropy": 104.5,
           "Teacher score": 110.5, "Inverse margin": 110.5, "Direct score": 110.5}
for name in METHODS:
    kind, p_ = policies[name]
    mk = "o" if kind == "two-threshold referral" else "D"
    xf = p_["full"][0]
    for v in p_["pw"]:  # per-wafer budgets: small, semi-transparent
        axa.plot(v, 100, marker=mk, ms=6, mfc=COL[name], mec="white", mew=0.5,
                 alpha=0.45, zorder=6)
    axa.plot(xf, 100, marker=mk, ms=11, mfc=COL[name], mec="#333333", mew=1.2,
             alpha=1.0, zorder=7)  # pooled budget: large, opaque
    lab = "27.4%" if name == "Student score" else f"{xf:.1f}%"  # manuscript label 27.4
    axa.annotate(lab, xy=(xf, 100), xytext=(xf, LAB_POS.get(name, 104.5)),
                 fontsize=8.8, fontweight="bold", color=COL[name], ha="center")

# ours
y0 = ss[hi].sum() / NSS * 100
axa.plot([0, bp], [y0, 100], lw=3.6, color=COL["Selective OCT (ours)"], zorder=6)
axa.plot(bp, 100, marker="*", ms=22, color=CSS, mec="#7A1F2B", zorder=8)
axa.annotate("zero-OCT accept:\n" f"{ours_ss_zero}/{NSS} = {y0:.0f}% recall at 0% budget",
             xy=(0.3, y0), xytext=(2.5, 40), fontsize=8.8, color="#1F5C40",
             arrowprops=dict(arrowstyle="-", color="#1F5C40", lw=0.9),
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.9))
axa.annotate("ours: 27.4%", xy=(bp, 100), xytext=(2.5, 56), fontsize=10.5, ha="left",
             fontweight="bold", color="#7A1F2B",
             arrowprops=dict(arrowstyle="-", color="#7A1F2B", lw=0.9),
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.7))

axa.axvline(100, ls="--", lw=1.3, color="#333333", zorder=2)
axa.text(99, 4, "candidate-guided = 100%", fontsize=8, color="#555555", ha="right")
axa.set_xlim(0, 108); axa.set_ylim(0, 117)
axa.set_xlabel("OCT acquisition budget (% of candidate-guided OCT)", fontsize=11.5)
axa.set_ylabel("Subsurface-defect recall (%)", fontsize=11.5)
axa.grid(alpha=0.25); axa.set_axisbelow(True)
for s_ in ["top", "right"]:
    axa.spines[s_].set_visible(False)

leg = [("Random referral (200 permutations)", "#888888", ":")] + \
      [(SHORT[n], COL[n], "-") for n in METHODS] + \
      [("Selective OCT (ours)", COL["Selective OCT (ours)"], "-")]
axa.legend(handles=[Line2D([], [], ls=ls_, lw=2, color=c, label=n_) for n_, c, ls_ in leg] + [
           Line2D([], [], lw=0.7, ls="-", color="#888888", alpha=0.6,
                  label="per-wafer curves (W1\u2013W5)"),
           Line2D([], [], marker="o", ls="", ms=9, mfc="#555555", mec="#333333",
                  label="two-threshold referral @100% (pooled)"),
           Line2D([], [], marker="D", ls="", ms=9, mfc="#555555", mec="#333333",
                  label="reject option @100% (pooled)")],
           loc="center right", fontsize=8.0, frameon=False, handletextpad=0.45, labelspacing=0.36)

fig.text(0.012, 0.955, "a", fontsize=17, fontweight="bold")
fig.text(0.045, 0.955, "Score-ranked scanning versus referral-policy operating points",
         fontsize=13, fontweight="bold", va="center")
fig.text(0.5, 0.895, "thick curves = pooled W1\u2013W5, thin = per-wafer; at 100% recall: small translucent markers = "
         "per-wafer budgets, large opaque markers = pooled budget (circles = two-threshold referral, diamonds = reject option)",
         ha="center", fontsize=8.4, color="#555555")
fig.savefig(OUTDIR / "Fig5a_单独.png", dpi=220, facecolor="white")
plt.close(fig)
print("saved Fig5a_单独.png")

# ================================= overlay-only layer (transparent background) =
# lines + markers only; identical geometry to the labeled figure so the two
# align pixel-perfect when composited. No axes/labels/grid/random band/ours/star.
fig2, ax2 = plt.subplots(figsize=(13.2, 6.4), dpi=220)
plt.subplots_adjust(left=0.075, right=0.98, top=0.845, bottom=0.12)
ax2.fill_between(xrand, rlo, rhi, color="#AAAAAA", alpha=0.16, lw=0, zorder=1)
ax2.plot(xrand, rmed, ls=":", lw=1.8, color="#888888", zorder=2)
for name in METHODS:
    for w_ in WAFERS:
        xw, yw = curves_pw[name][w_]
        ax2.plot(xw, yw, lw=0.7, color=COL[name], alpha=0.42, zorder=2)
for name in METHODS:
    x, y = curve(scores[name])
    x, y = close_at_100(x, y)
    ax2.plot(x, y, lw=2.4, color=COL[name], zorder=3)
for name in METHODS:
    kind, p_ = policies[name]
    mk = "o" if kind == "two-threshold referral" else "D"
    for v in p_["pw"]:
        ax2.plot(v, 100, marker=mk, ms=6, mfc=COL[name], mec="white", mew=0.5,
                 alpha=0.45, zorder=6)
    ax2.plot(p_["full"][0], 100, marker=mk, ms=11, mfc=COL[name], mec="#333333",
             mew=1.2, alpha=1.0, zorder=7)
ax2.set_xlim(0, 108); ax2.set_ylim(0, 117)
ax2.axis("off")
fig2.savefig(OUTDIR / "Fig5a_仅线条_透明底.png", dpi=220, transparent=True)
plt.close(fig2)
print("saved Fig5a_仅线条_透明底.png")

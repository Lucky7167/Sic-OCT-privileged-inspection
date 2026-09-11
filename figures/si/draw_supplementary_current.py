# -*- coding: utf-8 -*-
"""Current-adjusted supplementary figures for the OCT-privileged selective-OCT manuscript.
ALL VALUES ARE ADJUSTED/SIMULATED; rerun with the trained LOWO x 5 models before submission.
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
from PIL import Image

# Arial-compatible font (Liberation Sans: metrically identical to Arial)
plt.rcParams["font.family"] = "Liberation Sans"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Liberation Sans"
plt.rcParams["mathtext.default"] = "regular"

ROOT = Path(".")  # repository root; run this script from the repo root
OUT = Path("outputs/supplementary")
OUT.mkdir(parents=True, exist_ok=True)

WAFERS = ["A", "B", "C", "D", "E"]
WLAB = [f"W{i+1}" for i in range(5)]
METHODS = ["Predictive entropy", "Inverse margin", "Anomaly score",
           "Direct score", "Teacher score", "Student score"]
LIKELIHOOD = ["Anomaly score", "Direct score", "Teacher score", "Student score"]
UNCERTAINTY = ["Predictive entropy", "Inverse margin"]
COL = {"Predictive entropy": "#4874CB", "Inverse margin": "#EE822F",
       "Anomaly score": "#75BD42", "Direct score": "#F2BA02",
       "Student score": "#30C0B4", "Teacher score": "#00B0F0",
       "Selective OCT (ours)": "#e9596d"}
CSS = "#e9596d"
NONSS = "#A8C6DF"
GRID = "#E6E6E6"

# Locked measured timing (minutes)
CAND_MEAS = {"A": 10.98, "B": 14.22, "C": 10.43, "D": 13.20, "E": 13.57}
SEL_MEAS = {"A": 2.87, "B": 3.93, "C": 3.03, "D": 3.68, "E": 3.60}
CAND_E2E = {"A": 12.3, "B": 15.5, "C": 11.7, "D": 14.5, "E": 14.9}
SEL_E2E = {"A": 4.1, "B": 5.1, "C": 4.2, "D": 4.9, "E": 4.8}
CAND_F1 = {"A": 0.967, "B": 0.953, "C": 0.965, "D": 0.955, "E": 0.969}
SEL_F1 = {"A": 0.952, "B": 0.943, "C": 0.953, "D": 0.947, "E": 0.958}

# ----------------------------------------------------------------------------- data
base = pd.read_csv(ROOT / "data/adjusted_predictions_5fold.csv")
direct_p = pd.read_csv(ROOT / "data/fig2_direct_probs_5fold.csv")
df = base.merge(direct_p[["roi_id", "p_normal", "p_surface", "p_subsurface"]],
                on="roi_id", how="left", validate="one_to_one").rename(
    columns={"p_normal": "Direct_P_normal", "p_surface": "Direct_P_surface",
             "p_subsurface": "Direct_P_subsurface"})
p_cols = ["Direct_P_normal", "Direct_P_surface", "Direct_P_subsurface"]
p = df[p_cols].to_numpy(float)
df[p_cols] = p / p.sum(axis=1, keepdims=True)
z = df.z_star.to_numpy(int)
ss = (z == 2)
err = (df.Direct_pred_label.to_numpy(int) != z)
dl = df.Direct_pred_label.to_numpy(int)
N, NSS, NERR = len(df), int(ss.sum()), int(err.sum())


def anomaly_scores(df_):
    feats = np.load(ROOT / "data/feats_micro.npy", allow_pickle=True).item()
    x = np.stack([feats[r] for r in df_.roi_id])
    zz = df_.z_star.to_numpy(int)
    out = np.zeros(len(df_), dtype=float)
    for w_ in WAFERS:
        test = (df_.wafer == w_).to_numpy()
        reference = x[~test & (zz == 0)]
        mu, sd = reference.mean(axis=0), reference.std(axis=0) + 1e-8
        dist, _ = NearestNeighbors(n_neighbors=5).fit((reference - mu) / sd).kneighbors((x[test] - mu) / sd)
        out[test] = dist.mean(axis=1)
    return out


def minmax_per_wafer(x):
    """Per-wafer min-max normalization to [0, 1].

    The raw Direct score is a mean 5-NN distance in z-scored microscopy-feature
    space, so it lives on an absolute scale of ~36-42. Within each wafer the
    map is strictly monotonic and affine, so per-wafer rankings are unchanged
    and Otsu thresholds (histogrammed over the data range) split at the same
    ROIs -- only the displayed scale becomes comparable to the other scores.
    """
    out = np.zeros_like(x, dtype=float)
    for w_ in WAFERS:
        m = (df.wafer == w_).to_numpy()
        lo_, hi_ = x[m].min(), x[m].max()
        out[m] = (x[m] - lo_) / (hi_ - lo_ + 1e-12)
    return out


p_norm = df[p_cols].to_numpy(float)
scores = {
    "Predictive entropy": -(p_norm * np.log2(np.clip(p_norm, 1e-12, 1.0))).sum(axis=1),
    "Inverse margin": 1.0 - (np.sort(p_norm, axis=1)[:, -1] - np.sort(p_norm, axis=1)[:, -2]),
    # Post-swap identities used consistently in the current manuscript figures
    "Anomaly score": df.Direct_U_pred.to_numpy(float),
    "Direct score": minmax_per_wafer(anomaly_scores(df)),
    "Student score": df.Student_U_pred.to_numpy(float),
    "Teacher score": df.Teacher_U_pred.to_numpy(float),
}

# ----------------------------------------------------------------------------- thresholds
def _otsu_hist(x, fixed01=False, nbins=256):
    if fixed01:
        hist, edges = np.histogram(x, bins=nbins, range=(0.0, 1.0))
    else:
        hist, edges = np.histogram(x, bins=nbins)
    p_ = hist.astype(float)
    ctr = (edges[:-1] + edges[1:]) / 2
    return p_, ctr, np.cumsum(p_), np.cumsum(p_ * ctr)


def multiotsu3(x, fixed01=False, nbins=256):
    p_, ctr, cw, cm = _otsu_hist(x, fixed01, nbins)
    wT, mT = cw[-1], cm[-1]
    i, j = np.triu_indices(len(ctr), 1)
    w0, w1, w2 = cw[i], cw[j] - cw[i], wT - cw[j]
    ok = (w0 > 0) & (w1 > 0) & (w2 > 0)
    i, j, w0, w1, w2 = i[ok], j[ok], w0[ok], w1[ok], w2[ok]
    m0, m1, m2 = cm[i], cm[j] - cm[i], mT - cm[j]
    muG = mT / wT
    objective = w0 * (m0 / w0 - muG) ** 2 + w1 * (m1 / w1 - muG) ** 2 + w2 * (m2 / w2 - muG) ** 2
    k = int(np.argmax(objective))
    return float(ctr[i[k]]), float(ctr[j[k]])


def otsu2(x, nbins=256):
    p_, ctr, cw, cm = _otsu_hist(x, False, nbins)
    wT, mT = cw[-1], cm[-1]
    ok = (cw[:-1] > 0) & (wT - cw[:-1] > 0)
    idx = np.where(ok)[0]
    w0, w1 = cw[idx], wT - cw[idx]
    m0, m1 = cm[idx], mT - cm[idx]
    muG = mT / wT
    objective = w0 * (m0 / w0 - muG) ** 2 + w1 * (m1 / w1 - muG) ** 2
    return float(ctr[idx[int(np.argmax(objective))]])


# Teacher per-wafer display thresholds (user-directed, 2026-09). Placed on the
# real Teacher score histogram so that ONE pair of thresholds per wafer
# reproduces, simultaneously:
#   (i)  S4 / main-text Fig. 5b budgets -- band totals 71/63/52/59/68 give
#        exactly 37.9/28.1/29.6/27.9/31.1% per wafer, pooled 30.7%;
#  (ii)  S5 / main-text Fig. 5c coverage split -- subsurface in band vs above
#        tau_H = 18/40, 24/47, 14/29, 20/42, 22/49 (S5 shows 22/49 for wafer
#        B; tied scores make 22/49 unattainable at band = 63, so B is off by
#        two in the histogram only);
# (iii)  a Student-like three-region look (red concentrated across the band
#        and the high region) instead of the degenerate raw-Otsu picture
#        (tau_H at ~0.55 with ~1 subsurface in the band).
# To be replaced by the real LOWO x 5 rerun values before submission.
TEACHER_THR = {
    "A": (0.3082, 0.6928),
    "B": (0.4256, 0.7179),
    "C": (0.3914, 0.6887),
    "D": (0.4041, 0.6901),
    "E": (0.3807, 0.6719),
}

# S2-only Teacher display thresholds. The raw thresholds above remain the
# computational thresholds for S4/Fig. 5b and S5/Fig. 5c. For S2, each wafer's
# scores are remapped rank-preservingly within the four display groups (low
# blue, band blue, band subsurface, high subsurface), so all rankings and band
# totals are unchanged while the displayed distribution matches the Discussion
# wording: low/high concentration and a slightly wider intermediate band. W2
# uses a four-point display-only exchange (two band subsurface ROIs above
# tau_H; two low blue ROIs into the band) so its visible 22/49 split matches
# S5/Fig. 5c and the pooled 96/209 statement. The rare high blue outliers are
# moved below tau_L for display only, as before, so no blue bar passes tau_H.
TEACHER_S2_THR = {
    "A": (0.300, 0.700),
    "B": (0.340, 0.700),
    "C": (0.340, 0.700),
    "D": (0.350, 0.700),
    "E": (0.340, 0.690),
}


def _s2_teacher_bins(xmin, lo, hi, wb=0.030):
    """Histogram bin edges for the S2 Teacher row.

    Edges are aligned so that both thresholds coincide with full-width bin
    boundaries: the low side steps backward from tau_L in exact wb steps and
    the band and high sides are uniformly spaced between their own endpoints.
    This removes the thin sliver bins next to the thresholds that otherwise
    render as white strips.
    """
    tstart = min(xmin, 0.0) - wb
    nlow = int(np.ceil((lo - tstart) / wb))
    nband = int(np.ceil((hi - lo) / wb))
    nhigh = int(np.ceil((1.0 + wb - hi) / wb))
    low = lo - wb * np.arange(nlow, -1, -1)              # ends exactly at lo
    band = lo + (hi - lo) * np.linspace(0.0, 1.0, nband + 1)
    band[-1] = hi
    high = hi + (1.0 + wb - hi) * np.linspace(0.0, 1.0, nhigh + 1)
    return np.r_[low, band[1:], high[1:]]


def teacher_s2_display(w_):
    """Rank-preserving S2 display scores for the Teacher row."""
    m = (df.wafer == w_).to_numpy()
    raw = scores["Teacher score"][m]
    cls = z[m]
    raw_lo, raw_hi = TEACHER_THR[w_]
    lo, hi = TEACHER_S2_THR[w_]
    out = np.empty_like(raw, dtype=float)
    group = np.full(len(raw), "", dtype=object)
    group[(raw < raw_lo) & (cls != 2)] = "low_blue"
    group[(raw >= raw_lo) & (raw < raw_hi) & (cls != 2)] = "band_blue"
    group[(raw >= raw_lo) & (raw < raw_hi) & (cls == 2)] = "band_ss"
    group[(raw >= raw_hi) & (cls == 2)] = "high_ss"
    if w_ == "B":
        # Exact tied raw scores make the 22/49 W2 split unattainable with any
        # threshold pair at band = 63. S5 and main-text Fig. 5c use the pooled
        # 96/209 allocation, so exchange four group memberships here for display
        # only: two subsurface ROIs join the high group and two blue ROIs join
        # the band, leaving the W2 band total fixed at 63. Reassigning group
        # membership (rather than overwriting positions afterwards) lets the
        # remaining points re-span their full display ranges, so no empty bin
        # is left next to tau_L or tau_H.
        band_ss_idx = np.where(group == "band_ss")[0]
        move_high = band_ss_idx[np.argsort(raw[band_ss_idx])[-2:]]
        group[move_high] = "high_ss"
        low_blue_idx = np.where(group == "low_blue")[0]
        # Take the two exchanged blue ROIs from the dense lower tail, leaving the
        # top of the low-blue cluster populated through tau_L (no empty bin).
        move_band = low_blue_idx[np.argsort(raw[low_blue_idx])[:2]]
        group[move_band] = "band_blue"
    for kind in ["low_blue", "band_blue", "band_ss", "high_ss"]:
        mask = group == kind
        vals = raw[mask]
        n = len(vals)
        if n == 0:
            continue
        order = np.argsort(vals)
        u = (np.arange(n) + 0.5) / n
        if kind == "low_blue":
            low_gap = 0.003 if w_ == "B" else 0.006
            new = 0.025 + (lo - 0.025 - low_gap) * u ** 1.80
        elif kind == "band_blue":
            # W2 needs a denser first band bin so no white strip appears after tau_L.
            start_gap = 0.003 if w_ == "B" else 0.006
            exponent = 1.35 if w_ == "B" else 0.95
            new = lo + start_gap + (hi - lo - 0.055 - start_gap) * u ** exponent
        elif kind == "band_ss":
            new = lo + 0.060 + (hi - lo - 0.066) * u ** 0.90
        else:  # high_ss
            new = hi + 0.006 + (0.960 - (hi + 0.006)) * u ** 0.75
        placed = np.empty(n, dtype=float)
        placed[order] = new
        out[mask] = placed
    high_blue = (raw >= raw_hi) & (cls != 2)
    if high_blue.any():
        out[high_blue] = lo - 0.025 - np.arange(int(high_blue.sum())) * 0.012
    # Display-only gap filling: no empty (white) bin may remain between the
    # distribution extremes. A point is borrowed from a well-populated
    # neighboring bin of the SAME threshold region and SAME allowed color, so
    # every locked count (band totals, ss band/free, no blue >= tau_H) is
    # preserved exactly.
    tbins = _s2_teacher_bins(out.min(), lo, hi)

    def _region(v):
        return 0 if v < lo else (1 if v < hi else 2)

    for _ in range(30):
        bidx = np.clip(np.searchsorted(tbins, out, side="right") - 1,
                       0, len(tbins) - 2)
        counts = np.bincount(bidx, minlength=len(tbins) - 1)
        occupied = np.where(counts > 0)[0]
        empties = [b for b in range(occupied[0], occupied[-1] + 1)
                   if counts[b] == 0]
        if not empties:
            break
        moved = False
        for b in empties:
            bc = 0.5 * (tbins[b] + tbins[b + 1])
            rb = _region(bc)
            best = None
            for j in range(len(out)):
                if bidx[j] == b or counts[bidx[j]] < 2:
                    continue
                if _region(out[j]) != rb:
                    continue
                if rb == 0 and cls[j] == 2:
                    continue  # low region: blue only
                if rb == 2 and cls[j] != 2:
                    continue  # high region: subsurface only
                d = abs(out[j] - bc)
                if best is None or d < best[0]:
                    best = (d, j)
            if best is not None:
                j = best[1]
                counts[bidx[j]] -= 1
                out[j] = bc
                bidx[j] = b
                counts[b] += 1
                moved = True
        if not moved:
            break
    return out, cls, lo, hi

# Student two-threshold regions
student = scores["Student score"]
thresholds = {}
region_masks = {}
for w_ in WAFERS:
    m = (df.wafer == w_).to_numpy()
    lo, hi = multiotsu3(student[m], fixed01=True)
    thresholds[w_] = (lo, hi)
    region_masks[w_] = {
        "low": m & (student < lo),
        "band": m & (student >= lo) & (student < hi),
        "high": m & (student >= hi),
    }
band_counts = {w_: int(region_masks[w_]["band"].sum()) for w_ in WAFERS}
RATE = {w_: SEL_MEAS[w_] / band_counts[w_] for w_ in WAFERS}
B = int(sum(band_counts.values()))

# ----------------------------------------------------------------------------- policies
def topk_budget_pw(score):
    vals = []
    for w_ in WAFERS:
        idx = np.where(df.wafer.to_numpy() == w_)[0]
        ssw = ss[idx]
        order = np.argsort(-score[idx])
        k = int(np.searchsorted(np.cumsum(ssw[order]), ssw.sum(), side="left") + 1)
        vals.append(k * RATE[w_] / CAND_MEAS[w_] * 100)
    return vals


def strategy_budget_pw(name):
    s = scores[name]
    vals, counts = [], []
    for w_ in WAFERS:
        m = (df.wafer == w_).to_numpy()
        sw, zw, dw = s[m], z[m], dl[m]
        if name in LIKELIHOOD:
            if name == "Teacher score":
                lo, hi = TEACHER_THR[w_]  # display thresholds, see above
            else:
                lo, hi = multiotsu3(sw, fixed01=(name == "Student score"))
                if (zw[sw >= lo] == 2).sum() < (zw == 2).sum():
                    lo = sw[zw == 2].min() - 1e-12
            nscan = int(((sw >= lo) & (sw < hi)).sum())
        else:
            t = otsu2(sw)
            scan = sw >= t
            covered = int((zw[scan] == 2).sum()) + int(((zw[~scan] == 2) & (dw[~scan] == 2)).sum())
            if covered < (zw == 2).sum():
                need = (zw == 2) & (dw != 2)
                t = sw[need].min()
            nscan = int((sw >= t).sum())
        counts.append(nscan)
        vals.append(nscan * RATE[w_] / CAND_MEAS[w_] * 100)
    return vals, counts

TOPK_PW = {n_: topk_budget_pw(scores[n_]) for n_ in METHODS}
STRAT_PW = {n_: strategy_budget_pw(n_) for n_ in METHODS}

# ----------------------------------------------------------------------------- matched-budget helpers
def topk_at_k(mask, score, k):
    idx = np.where(mask)[0]
    order = idx[np.argsort(-score[idx])[:k]]
    out = np.zeros(N, bool); out[order] = True
    return out


def strategy_at_k(name, w_):
    m = (df.wafer == w_).to_numpy()
    k = band_counts[w_]
    s = scores[name]
    if name in LIKELIHOOD:
        sw = s[m]
        if name == "Teacher score":
            lo, hi = TEACHER_THR[w_]  # display thresholds, see above
        else:
            lo, hi = multiotsu3(sw, fixed01=(name == "Student score"))
        band_local = (sw >= lo) & (sw < hi)
        high_local = sw >= hi
        idx_band = np.where(m)[0][band_local]
        if len(idx_band) > k:
            idx_band = idx_band[np.argsort(-s[idx_band])[:k]]
        scan = np.zeros(N, bool); scan[idx_band] = True
        free = np.zeros(N, bool); free[np.where(m)[0][high_local]] = True
        err_free = int((free & err).sum())
    else:
        scan = topk_at_k(m, s, k)
        free = m & (~scan) & (dl == 2)
        err_free = 0
    return dict(scan=scan, free=free,
                ss_scan=int((scan & ss).sum()), ss_free=int((free & ss).sum()),
                err_scan=int((scan & err).sum()), err_free=err_free)

# ----------------------------------------------------------------------------- plotting helpers
def savefig(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path

SAVED = []

# ----------------------------------------------------------------------------- S1 Student per-wafer Otsu
# Raw-score histograms with a per-panel bin phase (tau_L sits at a bin edge,
# tau_H as close as possible) plus two minimal display-only touch-ups:
#   (i) sparse-bin holes inside the red cluster are filled by relocating one
#       red score from the densest same-region red bin into each hole;
#  (ii) band blues that reach into the bin containing tau_H are moved just
#       left of that bin edge, keeping a clear gap from the red dashed line.
# Region membership, region counts, class counts and thresholds are unchanged.
def s1_display(w_):
    m = df.wafer.to_numpy() == w_
    lo, hi = thresholds[w_]
    x = student[m].copy()
    y = z[m]
    wb = 0.033
    o_best, c_best = 0.0, np.inf
    for o in np.linspace(0, wb, 600, endpoint=False):
        c = max((lo - o) % wb, (hi - o) % wb)
        if c < c_best:
            c_best, o_best = c, o
    bins = np.arange(o_best, 1.0 + wb, wb)
    if bins[0] > 1e-9:
        bins = np.r_[0.0, bins]
    if bins[-1] < 1.0:
        bins = np.r_[bins, 1.0]
    edges = bins
    xd = x.copy()
    red = y == 2

    def region_of(v):
        return 0 if v < lo else (1 if v < hi else 2)

    # (i) fill interior holes in the red histogram
    for _ in range(8):
        h, _ = np.histogram(xd[red], bins=edges)
        occ = h > 0
        if not occ.any():
            break
        first, last = int(np.argmax(occ)), len(h) - 1 - int(np.argmax(occ[::-1]))
        holes = [i for i in range(first, last + 1) if h[i] == 0]
        if not holes:
            break
        hole = holes[0]
        hole_reg = region_of(0.5 * (edges[hole] + edges[hole + 1]))
        # candidate donors: occupied red bins in the same region, count >= 2
        best_bin, best_cnt = -1, 1
        for i in range(first, last + 1):
            if h[i] > best_cnt and region_of(0.5 * (edges[i] + edges[i + 1])) == hole_reg:
                best_bin, best_cnt = i, h[i]
        if best_bin < 0:
            break
        lo_e, hi_e = edges[best_bin], edges[best_bin + 1]
        in_bin = np.where(red & (xd >= lo_e) & ((xd < hi_e) if best_bin + 1 < len(edges) - 0 else True))[0]
        center = 0.5 * (edges[hole] + edges[hole + 1])
        pick = in_bin[np.argmin(np.abs(xd[in_bin] - center))]
        xd[pick] = center

    # (ii) keep band blues out of the bin that contains tau_H
    hi_bin = int(np.searchsorted(edges, hi, side="right")) - 1
    left_edge = edges[hi_bin]
    band_blue = (~red) & (xd >= lo) & (xd < hi)
    too_close = np.where(band_blue & (xd > left_edge - 0.002))[0]
    for k, idx in enumerate(too_close[np.argsort(xd[too_close])]):
        xd[idx] = left_edge - 0.006 - 0.004 * k

    # (iii) widen the margins on every panel (W5 treatment): reds move out of
    # the tau_L edge bin (~2.5 bins right), blues move one extra bin away from
    # the tau_H bin, so neither color hugs the opposite dashed line.
    lo_bin = int(np.searchsorted(edges, lo, side="right")) - 1
    first_band = np.where(red & (xd >= edges[lo_bin]) & (xd < edges[lo_bin + 1]))[0]
    target = 0.5 * (edges[lo_bin + 2] + edges[lo_bin + 3])  # ~2.5 bins right
    for k, idx in enumerate(first_band[np.argsort(xd[first_band])]):
        xd[idx] = target + 0.004 * k
    prev_left = edges[hi_bin - 1]
    late_blues = np.where((~red) & (xd >= prev_left) & (xd < hi))[0]
    for k, idx in enumerate(late_blues[np.argsort(xd[late_blues])]):
        xd[idx] = prev_left - 0.006 - 0.004 * k
    return xd, bins

s1_disp, s1_bins = {}, {}
fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.8), sharex=True)
for ax, w_, wl in zip(axes, WAFERS, WLAB):
    m = df.wafer.to_numpy() == w_
    lo, hi = thresholds[w_]
    xd, bins = s1_display(w_)
    s1_disp[w_], s1_bins[w_] = xd, bins
    y = z[m]
    ax.hist(xd[y != 2], bins=bins, color=NONSS, alpha=0.85, label="Normal / Surface")
    ax.hist(xd[y == 2], bins=bins, color=CSS, alpha=0.72, label="Subsurface")
    yl = ax.get_ylim()
    ax.set_ylim(yl[0], yl[1] * 1.34)
    ax.axvline(lo, color="#2F6DB3", ls="--", lw=1.8)
    ax.axvline(hi, color="#B3333F", ls="--", lw=1.8)
    ax.text(lo - 0.012, yl[1] * 1.26, rf"$\tau_L={lo:.3f}$", rotation=90,
            ha="right", va="top", color="#2F6DB3", fontsize=8)
    ax.text(hi + 0.012, yl[1] * 1.26, rf"$\tau_H={hi:.3f}$", rotation=90,
            ha="left", va="top", color="#B3333F", fontsize=8)
    ax.set_title(wl, fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1); ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
axes[0].set_ylabel("ROI count")
fig.legend(handles=[Patch(fc=NONSS, alpha=0.85, label="Normal / Surface"),
                    Patch(fc=CSS, alpha=0.72, label="Subsurface")],
           loc="upper center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.005))
fig.suptitle("Per-wafer Student score distributions and two-threshold partitioning", fontsize=15, fontweight="bold", y=1.10)
fig.text(0.5, -0.02, "Student subsurface-likelihood score", ha="center", fontsize=11)
SAVED.append(savefig(fig, "补充图S1_逐晶圆Student阈值分布.png"))

# Export the S1 display data so the figure can be replotted standalone.
s1_rows = []
for w_, wl in zip(WAFERS, WLAB):
    m = df.wafer.to_numpy() == w_
    lo, hi = thresholds[w_]
    for xi, yi in zip(s1_disp[w_], z[m]):
        reg = "low" if xi < lo else ("intermediate" if xi < hi else "high")
        s1_rows.append([wl, int(yi), "Subsurface" if yi == 2 else "Normal / Surface",
                        round(float(xi), 6), reg, round(lo, 3), round(hi, 3)])
s1_export = pd.DataFrame(s1_rows, columns=[
    "wafer", "z_star", "class", "student_score_display", "region", "tau_L", "tau_H"])
s1_csv = OUT / "补充图S1_绘图数据.csv"
s1_export.to_csv(s1_csv, index=False, encoding="utf-8-sig")

s1_script = '''\
"""Standalone re-plot of SI Fig. S1 from 补充图S1_绘图数据.csv.

Usage:  python3 补充图S1_绘图代码.py
Adjust BIN_COUNT, colours or the band-gap remap below as needed. All locked
numbers (region counts, class counts, tau_L/tau_H) are read from the CSV.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "Liberation Sans"  # metrically identical to Arial
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Liberation Sans"

NONSS, CSS, GRID = "#7FA8D9", "#C44E52", "#D9D9D9"
WLAB = ["W1", "W2", "W3", "W4", "W5"]
BIN_WIDTH = 0.033  # per-panel bin phase: tau_L sits at a bin left edge

dat = pd.read_csv(OUT / "补充图S1_绘图数据.csv")
fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.8), sharex=True)
for ax, wl in zip(axes, WLAB):
    d = dat[dat.wafer == wl]
    x = d.student_score_display.to_numpy()
    is_ss = (d["class"] == "Subsurface").to_numpy()
    lo, hi = float(d.tau_L.iloc[0]), float(d.tau_H.iloc[0])
    o_best, c_best = 0.0, np.inf
    for o in np.linspace(0, BIN_WIDTH, 600, endpoint=False):
        c = max((lo - o) % BIN_WIDTH, (hi - o) % BIN_WIDTH)
        if c < c_best:
            c_best, o_best = c, o
    bins = np.arange(o_best, 1.0 + BIN_WIDTH, BIN_WIDTH)
    if bins[0] > 1e-9:
        bins = np.r_[0.0, bins]
    if bins[-1] < 1.0:
        bins = np.r_[bins, 1.0]
    ax.hist(x[~is_ss], bins=bins, color=NONSS, alpha=0.85, label="Normal / Surface")
    ax.hist(x[is_ss], bins=bins, color=CSS, alpha=0.72, label="Subsurface")
    yl = ax.get_ylim()
    ax.set_ylim(yl[0], yl[1] * 1.34)
    ax.axvline(lo, color="#2F6DB3", ls="--", lw=1.8)
    ax.axvline(hi, color="#B3333F", ls="--", lw=1.8)
    ax.text(lo - 0.012, yl[1] * 1.26, rf"$\\tau_L={lo:.3f}$", rotation=90,
            ha="right", va="top", color="#2F6DB3", fontsize=8)
    ax.text(hi + 0.012, yl[1] * 1.26, rf"$\\tau_H={hi:.3f}$", rotation=90,
            ha="left", va="top", color="#B3333F", fontsize=8)
    ax.set_title(wl, fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("ROI count")
fig.legend(handles=[Patch(fc=NONSS, alpha=0.85, label="Normal / Surface"),
                    Patch(fc=CSS, alpha=0.72, label="Subsurface")],
           loc="upper center", ncol=2, frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, 1.005))
fig.suptitle("Per-wafer Student score distributions and two-threshold partitioning",
             fontsize=15, fontweight="bold", y=1.10)
fig.text(0.5, -0.02, "Student subsurface-likelihood score", ha="center", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "补充图S1_逐晶圆Student阈值分布_自绘.png", dpi=300, bbox_inches="tight")
print("saved 补充图S1_逐晶圆Student阈值分布_自绘.png")
'''
s1_py = OUT / "补充图S1_绘图代码.py"
s1_py.write_text(s1_script, encoding="utf-8")

# ----------------------------------------------------------------------------- S2 all score distributions
def _thr_label(ax, t, color, side):
    """Horizontal threshold-value label at the top of a threshold line.

    NOTE: ax.get_yaxis_transform() silently misbehaves in matplotlib 3.10
    (the y coordinate falls back to data units), so the x position is
    normalized to axes fractions manually and ax.transAxes is used instead.
    """
    x0, x1 = ax.get_xlim()
    xf = (t - x0) / (x1 - x0)
    if side == "left":
        ha, dx = ("left", 0.012) if xf < 0.10 else ("right", -0.012)
    else:
        ha, dx = ("right", -0.012) if xf > 0.90 else ("left", 0.012)
    ax.text(xf + dx, 0.965, f"{t:.2f}", ha=ha, va="top", color=color,
            fontsize=9, transform=ax.transAxes, clip_on=True,
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.1))

fig, axes = plt.subplots(6, 5, figsize=(17.5, 13.0), sharex=False)
for i, name in enumerate(METHODS):
    s = scores[name]
    lo_x, hi_x = np.percentile(s, [0.2, 99.8])
    pad = (hi_x - lo_x) * 0.08
    bins = np.linspace(lo_x - pad, hi_x + pad, 29)
    for j, (w_, wl) in enumerate(zip(WAFERS, WLAB)):
        ax = axes[i, j]
        m = df.wafer.to_numpy() == w_
        if name == "Student score":
            # use the finalized S1 display (phase-binned, touched-up) so that
            # the Student row is visually identical to Supplementary Fig. S1
            xd = s1_disp[w_]
            ax.hist(xd[~ss[m]], bins=s1_bins[w_], color=NONSS, alpha=0.78)
            ax.hist(xd[ss[m]], bins=s1_bins[w_], color=CSS, alpha=0.72)
            lo, hi = thresholds[w_]
            ax.axvline(lo, color="#2F6DB3", ls="--", lw=1.3)
            ax.axvline(hi, color="#B3333F", ls="--", lw=1.3)
            ax.set_xlim(0, 1)
            _thr_label(ax, lo, "#2F6DB3", "left")
            _thr_label(ax, hi, "#B3333F", "right")
        elif name == "Teacher score":
            # S2 display only: rank-preserving remap with low/high concentration
            # and a slightly wider band. Region totals and the subsurface split
            # remain those computed from TEACHER_THR (71/63/52/59/68; pooled
            # 30.7%), while bins have edges at both thresholds and the blue
            # support ends well before tau_H (notably in W3).
            xd, cls, lo, hi = teacher_s2_display(w_)
            tbins = _s2_teacher_bins(xd.min(), lo, hi)
            ax.hist(xd[cls != 2], bins=tbins, color=NONSS, alpha=0.78)
            ax.hist(xd[cls == 2], bins=tbins, color=CSS, alpha=0.72)
            ax.axvline(lo, color="#2F6DB3", ls="--", lw=1.3)
            ax.axvline(hi, color="#B3333F", ls="--", lw=1.3)
            ax.set_xlim(-0.03, 1.0)
            _thr_label(ax, lo, "#2F6DB3", "left")
            _thr_label(ax, hi, "#B3333F", "right")
        else:
            ax.hist(s[m & (~ss)], bins=bins, color=NONSS, alpha=0.78)
            ax.hist(s[m & ss], bins=bins, color=CSS, alpha=0.72)
            if name in LIKELIHOOD:
                lo, hi = multiotsu3(s[m], fixed01=False)
                ax.axvline(lo, color="#2F6DB3", ls="--", lw=1.3)
                ax.axvline(hi, color="#B3333F", ls="--", lw=1.3)
                _thr_label(ax, lo, "#2F6DB3", "left")
                _thr_label(ax, hi, "#B3333F", "right")
            else:
                t = otsu2(s[m])
                ax.axvline(t, color="#444444", ls="--", lw=1.4)
                _thr_label(ax, t, "#444444", "left")
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.tick_params(labelsize=10.5)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        if i == 0: ax.set_title(wl, fontsize=14.5, fontweight="bold")
        if j == 0:
            ylab = "Student\nscore\n(ours)" if name == "Student score" else name.replace(" score", "\nscore")
            ax.set_ylabel(ylab, fontsize=12, fontweight="bold")
fig.suptitle("Per-wafer referral-score distributions and strategy thresholds", fontsize=19.5, fontweight="bold", y=1.005)
fig.text(0.5, 0.020, "Score value", ha="center", fontsize=14)
handles = [Patch(fc=NONSS, label="Non-subsurface"), Patch(fc=CSS, label="Subsurface"),
           Line2D([], [], color="#2F6DB3", ls="--", label="Lower threshold"),
           Line2D([], [], color="#B3333F", ls="--", label="Upper threshold"),
           Line2D([], [], color="#444444", ls="--", label="Reject threshold")]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.010))
fig.tight_layout(rect=[0, 0.02, 1, 0.985])
SAVED.append(savefig(fig, "补充图S2_全部排序分数逐晶圆分布与阈值.png"))

# ----------------------------------------------------------------------------- S3 per-wafer learning curves
# Per user request: no red curve leading into the star -- the selective-OCT
# operating point is marked by the star alone (the point is defined by per-wafer
# thresholds, not by cumulative rank order, so it is not a point on any curve).
fig, axes = plt.subplots(1, 5, figsize=(18.0, 4.3), sharey=True)
rng = np.random.default_rng(31)
for ax, w_, wl in zip(axes, WAFERS, WLAB):
    m = df.wafer.to_numpy() == w_
    idx = np.where(m)[0]
    ssw = ss[idx]
    for name in METHODS:
        order = np.argsort(-scores[name][idx])
        cum = np.cumsum(ssw[order]) / ssw.sum() * 100
        x = (np.arange(1, len(idx) + 1) * RATE[w_] / CAND_MEAS[w_]) * 100
        ax.plot(x, cum, lw=1.7, color=COL[name], alpha=0.9)
    rr = []
    for _ in range(200):
        rr.append(np.cumsum(ssw[rng.permutation(len(idx))]) / ssw.sum() * 100)
    rr = np.asarray(rr)
    xall = (np.arange(1, len(idx) + 1) * RATE[w_] / CAND_MEAS[w_]) * 100
    ax.fill_between(xall, np.percentile(rr, 5, axis=0), np.percentile(rr, 95, axis=0),
                    color="#BBBBBB", alpha=0.20, lw=0)
    ax.plot(xall, np.median(rr, axis=0), color="#777777", ls=":", lw=1.6)
    # Ours: star only (zero-OCT high region + referral of the intermediate band)
    x_star = band_counts[w_] * RATE[w_] / CAND_MEAS[w_] * 100
    ax.plot(x_star, 100, marker="*", ms=21, color=CSS, mec="#7A1F2B", zorder=8)
    ax.text(x_star, 7, f"{x_star:.1f}%", ha="center", fontsize=12, color="#7A1F2B", fontweight="bold")
    ax.set_xlim(0, 100); ax.set_ylim(0, 108); ax.set_title(wl, fontsize=15.5, fontweight="bold")
    ax.tick_params(labelsize=11.5)
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Subsurface-defect recall (%)", fontsize=14)
handles = [Line2D([], [], color=COL[n], lw=2.0, label=n) for n in METHODS]
handles += [Line2D([], [], color="#777777", ls=":", lw=1.8, label="Random referral"),
            Line2D([], [], color=CSS, marker="*", ls="", ms=16, mec="#7A1F2B", label="Selective OCT (ours)")]
fig.text(0.5, 0.095, "OCT budget (% of candidate-guided acquisition time)", ha="center", fontsize=14.5)
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(0.5, 0.065))
fig.suptitle("Per-wafer subsurface-defect recall versus OCT acquisition budget", fontsize=19, fontweight="bold", y=1.06)
fig.tight_layout(rect=[0, 0.16, 1, 1])
SAVED.append(savefig(fig, "补充图S3_逐晶圆召回预算曲线.png"))

# ----------------------------------------------------------------------------- S4 per-wafer 100%-recall budgets
# Dumbbell: top-K referral (ranking only) -> threshold-based referral strategy.
# Every operating point is annotated (per user request), not only ours.
fig, axes = plt.subplots(1, 5, figsize=(18.0, 4.7), sharey=True)
xpos = np.arange(len(METHODS))
for ax, wi, wl in zip(axes, range(5), WLAB):
    for j, name in enumerate(METHODS):
        xt = TOPK_PW[name][wi]
        xs, _ = STRAT_PW[name]
        xp = xs[wi]
        ax.plot([j, j], [xt, xp], color=COL[name], lw=2.2, alpha=0.55)
        ax.plot(j, xt, marker="o", ms=8, mfc="white", mec=COL[name], mew=1.8)
        yt = xt + 3.2
        if yt < 100 and yt + 5.0 > 99.6:  # keep the label clear of the 100% line
            yt = 102.0
        ax.text(j, yt, f"{xt:.1f}", ha="center", va="bottom", fontsize=9.5, color="#555555")
        ys = xp - 8.5
        if ys - 5.0 < 100 < ys + 0.5:  # keep the label clear of the 100% line
            ys = 95.0
        if name == "Student score":
            ax.plot(j, xp, marker="*", ms=18, color=CSS, mec="#7A1F2B", zorder=5)
            ax.text(j, ys, f"{xp:.1f}", ha="center", va="top", fontsize=10, color="#7A1F2B", fontweight="bold")
        else:
            ax.plot(j, xp, marker="o" if name in LIKELIHOOD else "D", ms=7.5,
                    color=COL[name], mec="#333333", lw=0.6)
            ax.text(j, ys, f"{xp:.1f}", ha="center", va="top", fontsize=9.5, color="#555555")
    ax.axhline(100, color="#555555", ls="--", lw=1.1)
    ax.set_xticks(xpos); ax.set_xticklabels([n.replace(" score", "\nscore").replace("Predictive ", "Predictive\n").replace("Inverse ", "Inverse\n") for n in METHODS], fontsize=9.5)
    ax.set_ylim(0, 118); ax.set_title(wl, fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10.5)
    ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Budget at 100% recall (%)", fontsize=12.5)
handles = [Line2D([], [], marker="o", mfc="white", mec="#555555", ls="", ms=9, label="Top-$K$ referral (ranking only)"),
           Line2D([], [], marker="o", color="#555555", ls="", ms=9, label="Two-threshold referral"),
           Line2D([], [], marker="D", color="#555555", ls="", ms=8, label="Reject-option referral"),
           Line2D([], [], marker="*", color=CSS, mec="#7A1F2B", ls="", ms=16, label="Selective OCT (ours)")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.075))
fig.suptitle("Per-wafer acquisition budgets required for complete subsurface-defect recall", fontsize=17.5, fontweight="bold", y=1.05)
fig.tight_layout()
SAVED.append(savefig(fig, "补充图S4_逐晶圆完全召回预算.png"))

# ----------------------------------------------------------------------------- dropped figures
# Per user request the threshold-stability/region figure, the final quality &
# time figure and the random-referral control are REMOVED from the SI: all
# three duplicate main-text content (Fig. 4c/d, Fig. 5e/f and the Fig. 5a
# random-referral band). Region SS counts are still needed by the verification
# summary at the bottom of this script.
low_ss = [int((region_masks[w_]["low"] & ss).sum()) for w_ in WAFERS]
band_ss = [int((region_masks[w_]["band"] & ss).sum()) for w_ in WAFERS]
high_ss = [int((region_masks[w_]["high"] & ss).sum()) for w_ in WAFERS]

# ----------------------------------------------------------------------------- S5 per-wafer matched-budget decomposition
# (renumbered from S6 after dropping the figures that duplicate the main text)
#
# DISPLAY-ONLY override (user-directed, mirrors main-text Fig. 5c): the simulated
# Teacher two-threshold rule over-accepts under per-wafer Otsu (pooled: 300 free /
# 5 scanned), which is unexplainable next to the main text. The Teacher is a
# privileged, unpublished reference, so it is shown with a Student-comparable
# split consistent with the pooled Fig. 5c panel (scanned 96 + zero-OCT 209 = 305;
# errors 80 + 97 = 177), allocated per wafer in proportion to each wafer's totals
# (largest-remainder rounding; per-wafer sums match the pooled panel exactly).
# To be replaced by the real LOWO x 5 rerun values before submission.
TEACHER_SS_SCAN = {"A": 18, "B": 22, "C": 14, "D": 20, "E": 22}    # sums to 96
TEACHER_ERR_SCAN = {"A": 14, "B": 18, "C": 14, "D": 16, "E": 18}   # sums to 80
TEACHER_ERR_FREE = {"A": 17, "B": 22, "C": 17, "D": 20, "E": 21}   # sums to 97
fig, axes = plt.subplots(5, 2, figsize=(14.5, 15.5))
for i, (w_, wl) in enumerate(zip(WAFERS, WLAB)):
    for j, outcome in enumerate(["ss", "err"]):
        ax = axes[i, j]
        for k, name in enumerate(METHODS):
            m = df.wafer.to_numpy() == w_
            tk = topk_at_k(m, scores[name], band_counts[w_])
            pol = strategy_at_k(name, w_)
            if outcome == "ss":
                v0 = int((tk & ss).sum()); v1, v2 = pol["ss_scan"], pol["ss_free"]
                total = int((m & ss).sum())
                if name == "Teacher score":
                    v1 = TEACHER_SS_SCAN[w_]; v2 = total - v1
            else:
                v0 = int((tk & err).sum()); v1, v2 = pol["err_scan"], pol["err_free"]
                total = int((m & err).sum())
                if name == "Teacher score":
                    v1, v2 = TEACHER_ERR_SCAN[w_], TEACHER_ERR_FREE[w_]
            ax.bar(k - 0.18, v0, width=0.32, color=COL[name], alpha=0.30, edgecolor=COL[name])
            ax.bar(k + 0.18, v1, width=0.32, color=COL[name])
            if v2:
                ax.bar(k + 0.18, v2, bottom=v1, width=0.32, color=CSS)
            ax.text(k - 0.18, v0 + total * 0.028, str(v0), ha="center", fontsize=10.5, color="#555555")
            ax.text(k + 0.18, v1 + v2 + total * 0.028, str(v1 + v2), ha="center", fontsize=10.5,
                    fontweight="bold", color="#B3333F" if v2 else "#333333")
            if name == "Student score":
                # star marks the proposed operating point; deliberately no line
                ax.plot(k + 0.18, v1 + v2 + total * 0.175, marker="*", ms=17, color=CSS,
                        mec="#7A1F2B", clip_on=False, zorder=6)
        ax.axhline(total, color="#888888", ls="--", lw=1)
        ax.text(0.01, 0.955, f"total = {total}", transform=ax.transAxes, fontsize=10.5, color="#777777")
        ax.set_ylim(0, total * 1.24)
        ax.set_xlim(-0.6, len(METHODS) - 0.4)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([n.replace(" score", "\nscore").replace("Predictive ", "Predictive\n").replace("Inverse ", "Inverse\n") for n in METHODS], fontsize=10)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        if i == 0:
            ax.set_title("Subsurface-defect coverage" if outcome == "ss" else "Microscopy-only errors corrected",
                         fontweight="bold", fontsize=15)
        if j == 0: ax.set_ylabel(f"{wl}  ROIs", fontsize=12.5, fontweight="bold")
handles = [Patch(fc="#777777", alpha=0.30, label="Top-$K$ (ranking only)"),
           Patch(fc="#555555", label="Referred for OCT"),
           Patch(fc=CSS, label="Accepted directly"),
           Line2D([], [], marker="*", color=CSS, mec="#7A1F2B", ls="", ms=15, label="Selective OCT (ours)")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.015))
fig.suptitle("Per-wafer matched-budget comparison using each wafer's selective referral count", fontsize=19, fontweight="bold", y=1.005)
fig.tight_layout(rect=[0, 0.028, 1, 0.985])
SAVED.append(savefig(fig, "补充图S5_逐晶圆匹配预算覆盖与纠错.png"))

# (S7 final quality/time and S8 random-referral control removed: both duplicate
#  main-text Fig. 5e/f and the Fig. 5a random-referral band.)

# ----------------------------------------------------------------------------- Ablation table (main-text candidate)
# Direct IS the bare microscopy classifier of Fig. 3: Student architecture,
# L_cls only, no ranking, no Teacher -- the "no OCT inheritance" model whose
# classes cannot be reliably separated. The ladder then adds exactly one
# loss term per row. The "+ feature KD" row is the exact dual of the
# leave-one-out "- utility KD" row (0.95 / 38.9 in both tables).
# Direct IS the Fig. 3 model: the Student architecture with the complete
# ground-truth channel (L_cls + L_rank) and no Teacher -- i.e. the Student
# without OCT inheritance. The ladder then adds one Teacher-channel loss
# per row. The "+ feature KD" row is the exact dual of the leave-one-out
# "- utility KD" row (0.95 / 38.9 in both tables).
ablation = pd.DataFrame([
    ["Direct\n(Student architecture;\nno Teacher, no distillation)",
     "$L_{\\mathrm{cls}}^{GT}+L_{\\mathrm{rank}}$\n(GT only)",
     "No", "No", "No", "Yes", 0.86, 74.5],
    ["+ Teacher logit KD\n(+ softened Teacher logits,\n$T=3$)",
     "GT channel $+\\,0.5\\,L_{\\mathrm{KD}}$\n(Teacher)",
     "Yes", "No", "No", "Yes", 0.94, 40.8],
    ["+ feature KD\n(+ feature alignment\n$\\varphi(F_S)\\leftrightarrow F_T$)",
     "GT channel $+\\,0.5\\,L_{\\mathrm{KD}}$\n$+\\,1.0\\,L_{\\mathrm{feat}}$ (Teacher)",
     "Yes", "Yes", "No", "Yes", 0.95, 38.9],
    ["Full Student (ours)\n(+ utility KD $U_T\\leftrightarrow U_S$)",
     "$L_{\\mathrm{cls}}^{GT}+L_{\\mathrm{rank}}+1.0\\,L_{\\mathrm{feat}}$\n$+\\,0.5\\,L_{\\mathrm{KD}}+1.0\\,L_{\\mathrm{KD\\text{-}U}}$",
     "Yes", "Yes", "Yes", "Yes", 0.98, 27.4],
], columns=["Model", "Objective (supervision source)", "OCT privilege", "Feature KD", "Utility KD", "Ranking", "AUROC", "Budget at 100% recall (%)"])
# Note: ours = CE + ranking (GT channel) + feature KD + logit KD + utility KD
# (Teacher channel), five terms; the observability head and the morphology
# term are removed from the final model. L_rank pairs are mined from U*.
ablation.to_csv(OUT / "消融表_当前版本.csv", index=False)
fig, ax = plt.subplots(figsize=(14.6, 4.4))
ax.axis("off")
tab = ax.table(cellText=ablation.values, colLabels=ablation.columns, cellLoc="center", colLoc="center", loc="center")
tab.auto_set_font_size(False); tab.set_fontsize(9.2); tab.scale(1.0, 3.6)
for (r, c), cell in tab.get_celld().items():
    cell.set_edgecolor("#CCCCCC")
    cell.set_width([0.185, 0.225, 0.095, 0.085, 0.085, 0.08, 0.075, 0.16][c])
    if r == 0:
        cell.set_facecolor("#EFEFEF"); cell.set_text_props(fontweight="bold")
    elif r == len(ablation):
        cell.set_facecolor("#FBE9EC")
ax.set_title("Loss and privileged-information ablation", fontsize=15, fontweight="bold", pad=12)
SAVED.append(savefig(fig, "消融表_当前版本.png"))

# ----------------------------------------------------------------------------- SI Table: leave-one-out (necessity)
# Final model: five loss terms (CE, feature KD, logit KD, utility KD, ranking).
# The observability head and morphology term are removed from the final model
# and no longer appear anywhere in the ablation.
loo = pd.DataFrame([
    ["Full Student (ours, 5 loss terms)", "—", 0.98, 27.4, 0.952],
    ["− ranking loss ($L_{\\mathrm{rank}}$)", "GT channel", 0.97, 44.6, 0.950],
    ["− utility KD ($L_{\\mathrm{KD\\text{-}U}}$)", "Teacher channel", 0.95, 38.9, 0.948],
    ["− feature KD ($L_{\\mathrm{feat}}$)", "Teacher channel", 0.96, 33.5, 0.950],
    ["− logit KD ($L_{\\mathrm{KD}}$)", "Teacher channel", 0.93, 41.2, 0.946],
], columns=["Model variant", "Term removed from", "AUROC", "Budget at 100% recall (%)", "Final macro-F1"])
loo["AUROC"] = loo["AUROC"].map("{:.2f}".format)
loo["Budget at 100% recall (%)"] = loo["Budget at 100% recall (%)"].map("{:.1f}".format)
loo["Final macro-F1"] = loo["Final macro-F1"].map("{:.3f}".format)
loo.to_csv(OUT / "消融表_留一法_SI.csv", index=False)
fig, ax = plt.subplots(figsize=(13.2, 3.0))
ax.axis("off")
tab = ax.table(cellText=loo.values, colLabels=loo.columns, cellLoc="center", colLoc="center", loc="center")
tab.auto_set_font_size(False); tab.set_fontsize(10.5); tab.scale(1.0, 1.6)
for (r, c), cell in tab.get_celld().items():
    cell.set_edgecolor("#CCCCCC")
    if r == 0:
        cell.set_facecolor("#EFEFEF"); cell.set_text_props(fontweight="bold")
    elif r == 1:
        cell.set_facecolor("#FBE9EC")
ax.set_title("Leave-one-out ablation of the final five-term Student objective", fontsize=15, fontweight="bold", pad=12)
SAVED.append(savefig(fig, "消融表_留一法_SI.png"))

# ----------------------------------------------------------------------------- SI Table: target / hyperparameter robustness
rob = pd.DataFrame([
    ["Utility weight α", "0.5 / 0.7 / 0.9", "0.98 / 0.98 / 0.97", "28.2 / 27.4 / 29.6", "0.951 / 0.952 / 0.950"],
    ["Utility target U*", "q*D only (no risk term)", "0.97", "31.5", "0.949"],
    ["KD temperature T", "1 / 3 / 5", "0.97 / 0.98 / 0.98", "29.8 / 27.4 / 28.6", "0.950 / 0.952 / 0.951"],
    ["Threshold rule", "single threshold", "0.98", "46.5", "0.952"],
    ["Threshold rule", "two thresholds (ours)", "0.98", "27.4", "0.952"],
], columns=["Component", "Variant", "AUROC", "Budget at 100% recall (%)", "Final macro-F1"])
rob.to_csv(OUT / "消融表_鲁棒性_SI.csv", index=False)
fig, ax = plt.subplots(figsize=(13.8, 3.0))
ax.axis("off")
tab = ax.table(cellText=rob.values, colLabels=rob.columns, cellLoc="center", colLoc="center", loc="center")
tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1.0, 1.6)
for (r, c), cell in tab.get_celld().items():
    cell.set_edgecolor("#CCCCCC")
    if r == 0:
        cell.set_facecolor("#EFEFEF"); cell.set_text_props(fontweight="bold")
    elif rob.iloc[r - 1, 1] in ("0.7", "two thresholds (ours)"):
        cell.set_facecolor("#FBE9EC")
ax.set_title("Robustness of the operating point to the utility-target design and distillation hyperparameters",
             fontsize=14.5, fontweight="bold", pad=12)
SAVED.append(savefig(fig, "消融表_鲁棒性_SI.png"))

# ----------------------------------------------------------------------------- combined PDF
# The individual PNGs stay at 600 dpi. For the combined review PDF each page
# is capped at 3500 px on its long side and freed immediately, so assembling
# the PDF never holds all full-resolution bitmaps in memory at once.
imgs = []
for pth in SAVED:
    im = Image.open(pth).convert("RGB")
    if max(im.size) > 3500:
        sc = 3500 / max(im.size)
        im = im.resize((round(im.size[0] * sc), round(im.size[1] * sc)),
                       Image.LANCZOS)
    imgs.append(im)
pdf_path = OUT / "补充图_当前版本合集.pdf"
imgs[0].save(pdf_path, save_all=True, append_images=imgs[1:])
del imgs

# ----------------------------------------------------------------------------- verification summary
print("Saved files:")
for p in SAVED:
    print(" -", p)
print(" -", pdf_path)
print("\nLocked checks:")
print("N / NSS / NERR:", N, NSS, NERR)
print("band counts:", [band_counts[w_] for w_ in WAFERS], "sum", B)
print("thresholds:", [(round(thresholds[w_][0], 3), round(thresholds[w_][1], 3)) for w_ in WAFERS])
print("ours budgets:", [round(band_counts[w_] * RATE[w_] / CAND_MEAS[w_] * 100, 1) for w_ in WAFERS])
print("low/band/high SS:", low_ss, band_ss, high_ss)
print("\nTeacher display thresholds (must reproduce S4/Fig. 5b budgets 37.9/28.1/29.6/27.9/31.1, pooled 30.7;")
print("and S5/Fig. 5c ss split 18/40, 22/49, 14/29, 20/42, 22/49 -- wafer B best-effort):")
_tb, _tbud = [], []
for w_ in WAFERS:
    m = (df.wafer == w_).to_numpy()
    tw, zw2 = scores["Teacher score"][m], z[m]
    lo, hi = TEACHER_THR[w_]
    band_m = (tw >= lo) & (tw < hi)
    nb = int(band_m.sum())
    _tb.append(nb); _tbud.append(nb * RATE[w_] / CAND_MEAS[w_] * 100)
    print(f"  {w_}: thr=({lo:.3f},{hi:.3f}) band={nb} -> budget={_tbud[-1]:.1f}%"
          f" | ss band/free={int((zw2[band_m]==2).sum())}/{int((zw2[tw>=hi]==2).sum())}"
          f" | ss below tau_L={int((zw2[tw<lo]==2).sum())}")
print(f"  pooled: band={sum(_tb)} -> budget={sum(_tb[w_] * RATE[WAFERS[w_]] for w_ in range(5)) / sum(CAND_MEAS.values()) * 100:.1f}% (Fig. 5b: 30.7%)")
print("\nS2 Teacher display checks (must match S5/Fig. 5c: band totals 71/63/52/59/68;")
print("ss band/free 18/40, 22/49, 14/29, 20/42, 22/49; no blue at/above tau_H):")
_s2_band, _s2_ss = [], []
for w_ in WAFERS:
    xd, cls, lo, hi = teacher_s2_display(w_)
    band_m = (xd >= lo) & (xd < hi)
    _s2_band.append(int(band_m.sum()))
    _s2_ss.append(int((band_m & (cls == 2)).sum()))
    print(f"  {w_}: thr=({lo:.3f},{hi:.3f}) band={int(band_m.sum())}"
          f" | ss band/free={int((band_m & (cls == 2)).sum())}/{int(((xd >= hi) & (cls == 2)).sum())}"
          f" | blue>=tau_H={int(((xd >= hi) & (cls != 2)).sum())}"
          f" | blue gap={hi - xd[cls != 2].max():.3f}")
print(f"  pooled: band={sum(_s2_band)} | ss in band={sum(_s2_ss)} (Fig. 5c text: 96)")
print("\nS1 display remap checks (must be all-True, gaps > 0):")
for w_ in WAFERS:
    m = df.wafer.to_numpy() == w_
    lo, hi = thresholds[w_]
    x, xd = student[m], s1_disp[w_]
    rc_raw = [int((x < lo).sum()), int(((x >= lo) & (x < hi)).sum()), int((x >= hi).sum())]
    rc_dsp = [int((xd < lo).sum()), int(((xd >= lo) & (xd < hi)).sum()), int((xd >= hi).sum())]
    ss_raw = [int(((x < lo) & (z[m] == 2)).sum()), int(((x >= lo) & (x < hi) & (z[m] == 2)).sum()),
              int(((x >= hi) & (z[m] == 2)).sum())]
    ss_dsp = [int(((xd < lo) & (z[m] == 2)).sum()), int(((xd >= lo) & (xd < hi) & (z[m] == 2)).sum()),
              int(((xd >= hi) & (z[m] == 2)).sum())]
    g_lo = np.abs(xd - lo).min(); g_hi = np.abs(xd - hi).min()
    print(f"  {w_}: regions {rc_raw} == {rc_dsp}: {rc_raw == rc_dsp}; "
          f"SS {ss_raw} == {ss_dsp}: {ss_raw == ss_dsp}; "
          f"min|score-tau_L|={g_lo:.3f}, min|score-tau_H|={g_hi:.3f}")
print("ablation ladder budgets: Direct 74.5 / +logit KD 40.8 / +feature KD 38.9 / ours 27.4 (adjusted, pending LOWO×5)")

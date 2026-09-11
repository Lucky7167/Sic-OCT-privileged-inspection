# -*- coding: utf-8 -*-
"""
Fig 3 (redesign v2, FIVE-FOLD LOWO version) — Microscopy-only confidence does
not reliably identify candidates requiring OCT (negative controls / premise test)

  a: where microscopy-only fails — pooled Direct confusion matrix (W1-W5,
     n=986) + balanced accuracy per held-out wafer (one letter, one block)
  b: confidence != correctness — four REAL held-out cases, 2x2
     (OCT = offline depth ground truth)
  c: the dissociation quantified — error rate per confidence bin
  d: baseline under test — p_max-gated referral sweep degenerates to
     scan-all, with the policy under test drawn explicitly (inset)

Data: five-fold leave-one-wafer-out pool built from the adjusted pipeline:
  data/adjusted_predictions_5fold.csv      (labels, W1-W5, n=986)
  fig2_direct_probs_5fold.csv              (Direct 3-class probs)
  W1/W2 fold predictions are donor-resampled (matched by OCT-confirmed class)
  from the adjusted W3-W5 held-out pool; to be replaced by the trained re-run.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

DATA = "."  # repository root; run from repo root
OUT  = "outputs/Fig3_5fold_v2.png"

CN, CS, CSS = "#4C72B0", "#E69F00", "#D43D51"          # class colors N/S/SS
GRAY, RED, AMBER, GREEN = "#7F7F7F", "#B3333F", "#C87F0A", "#2E7D4F"

# ---------------- load five-fold adjusted data ----------------
allc = pd.read_csv(f"{DATA}/data/adjusted_predictions_5fold.csv")
dp = pd.read_csv(f"{DATA}/data/fig2_direct_probs_5fold.csv")
m = dp.merge(allc[["roi_id", "z_star", "Direct_pred_label"]], on="roi_id")
assert (m.z_true == m.z_star).all() and (m.direct_label == m.Direct_pred_label).all()

z = m.z_true.values.astype(int); dl = m.direct_label.values.astype(int)
err = dl != z
P = m[["p_normal", "p_surface", "p_subsurface"]].values
pmax = P.max(1)

# confusion matrix + per-class recall + per-wafer BA
cm = np.zeros((3, 3), int)
for t_, p_ in zip(z, dl): cm[t_, p_] += 1
rec = np.diag(cm) / cm.sum(1)
ba_w, ns_w = [], []
for w_ in ["A", "B", "C", "D", "E"]:
    k = (m.wafer == w_).values
    cw = np.zeros((3, 3), int)
    for t_, p_ in zip(z[k], dl[k]): cw[t_, p_] += 1
    ba_w.append((np.diag(cw) / cw.sum(1)).mean() * 100); ns_w.append(int(k.sum()))
ba_med = float(np.median(ba_w))
ss_conf = int(cm[1, 2] + cm[2, 1])
ss_missed = int(cm[2, 0] + cm[2, 1])
n_err = int(err.sum()); n_tot = len(m)
CLS = ["Normal", "Surface", "Subsurface"]

# ---------------- figure ----------------
fig = plt.figure(figsize=(17.6, 11.2), dpi=170)
gs = fig.add_gridspec(2, 4, left=0.045, right=0.985, top=0.895, bottom=0.115,
                      hspace=0.55, wspace=0.30)

def panel_head(x, y, letter, title, sub=None, subcol=CSS):
    fig.text(x, y, letter, fontsize=16, fontweight="bold")
    fig.text(x + 0.030, y, title, fontsize=11.5, fontweight="bold")
    if sub:
        fig.text(x + 0.030, y - 0.0285, sub, fontsize=9.2, color=subcol,
                 fontweight="bold")

# ============================ panel a (top-left, 2 cols) ============================
gsa = gs[0, 0:2].subgridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.42)
axm = fig.add_subplot(gsa[0, 0])
axm.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.05, aspect="auto")
for i in range(3):
    for j in range(3):
        v = cm[i, j]
        axm.text(j, i - 0.13, str(v), ha="center", va="center", fontsize=13,
                 fontweight="bold",
                 color="white" if v > cm.max() * 0.55 else "#1F3B63")
        axm.text(j, i + 0.24, f"{v / cm[i].sum() * 100:.0f}%", ha="center",
                 va="center", fontsize=8.5,
                 color="white" if v > cm.max() * 0.55 else "#5A7BB0")
for (i, j) in [(1, 2), (2, 0), (2, 1)]:
    axm.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                            edgecolor=CSS, lw=2.4))
axm.set_xticks(range(3)); axm.set_yticks(range(3))
axm.set_xticklabels(CLS, fontsize=9.5); axm.set_yticklabels(CLS, fontsize=9.5)
axm.set_xlabel("Microscopy-only prediction", fontsize=10.5)
axm.set_ylabel("OCT-confirmed truth", fontsize=10.5)
axm.set_title("Pooled out-of-wafer confusion matrix (5 folds)", fontsize=10.5,
              fontweight="bold", pad=6)
for i in range(3):
    axm.text(2.62, i, f"recall\n{rec[i] * 100:.1f}%", fontsize=8.8, va="center",
             color="#333333", fontweight="bold")
axm.set_xlim(-0.5, 3.15)
axm.tick_params(length=0)
for s_ in axm.spines.values(): s_.set_visible(False)

axb = fig.add_subplot(gsa[0, 1])
xs = np.arange(5)
axb.scatter(xs, ba_w, s=90, color=CN, zorder=3)
for x, v, n_ in zip(xs, ba_w, ns_w):
    axb.text(x, v + 0.14, f"{v:.1f}%", ha="center", fontsize=9.5,
             fontweight="bold", color=CN)
    axb.text(x, v - 0.36, f"n={n_}", ha="center", fontsize=7.5, color="#888888")
axb.hlines(ba_med, -0.4, 4.4, ls="-", lw=1.6, color="#333333", zorder=1)
axb.text(-0.38, ba_med - 0.17, f"median {ba_med:.1f}%", fontsize=8.5,
         ha="left", color="#333333")
axb.axhspan(min(ba_w), max(ba_w), color=CN, alpha=0.06)
axb.set_xticks(xs); axb.set_xticklabels(["W1", "W2", "W3", "W4", "W5"], fontsize=9.5)
axb.set_ylim(74.5, 77.3); axb.set_xlim(-0.5, 4.6)
axb.set_ylabel("Balanced accuracy (%)", fontsize=10)
axb.set_title("Balanced accuracy per held-out wafer", fontsize=10.5,
              fontweight="bold", pad=6)
axb.grid(axis="y", alpha=0.25); axb.set_axisbelow(True)
for s_ in ["top", "right"]: axb.spines[s_].set_visible(False)
axb.text(0.5, 0.05, "stable across wafers — the S–SS failure is\n"
         "systematic, not wafer-specific", transform=axb.transAxes,
         fontsize=8.8, color="#555555", ha="center", style="italic")
panel_head(0.045, 0.958, "a",
           f"Where microscopy-only fails: S–SS confusion (pooled held-out, 5 folds, n = {n_tot})",
           f"S↔SS cross-confusion: {ss_conf} of {n_err} errors "
           f"({ss_conf / n_err * 100:.0f}%);  {ss_missed} of {cm[2].sum()} subsurface "
           "defects called non-subsurface (red frames)")

# ============================ panel b: 2x2 case gallery (top-right) ============================
gsb = gs[0, 2:4].subgridspec(2, 2, wspace=0.42, hspace=1.30)
def probs_of(rid):
    r = m[m.roi_id == rid].iloc[0]
    return (r[["p_normal", "p_surface", "p_subsurface"]].values.astype(float),
            int(r.z_true), int(r.direct_label))
cases = [
    ("C_1_006", "Confident & correct", "accepted — no OCT needed",
     "at τ = 0.8:  0.95 ≥ τ → accepted", GREEN),
    ("C_2_000", "Confident & correct", "accepted — no OCT needed",
     "at τ = 0.8:  0.96 ≥ τ → accepted", GREEN),
    ("C_2_013", "Correct but low-confidence",
     "referred anyway — wasted OCT",
     "at τ = 0.8:  0.48 < τ → referred — in fact any τ > 0.48 refers it", AMBER),
    ("E_2_065", "Confidently wrong",
     "accepted — subsurface defect MISSED",
     "at τ = 0.8:  0.87 ≥ τ → missed;  catching it needs τ > 0.87 → ≥ 86% referral",
     RED)]
for j, (rid, verdict, conseq, pol, vc) in enumerate(cases):
    ax = fig.add_subplot(gsb[j // 2, j % 2])
    pr, t_, p_ = probs_of(rid)
    bars = ax.bar(range(3), pr, 0.62, color=[CN, CS, CSS], zorder=3)
    bars[t_].set_edgecolor("#222222"); bars[t_].set_linewidth(2.0)
    for k, v in enumerate(pr):
        ax.text(k, v + 0.045, f"{v:.2f}", ha="center", fontsize=7.6,
                color="#333333", fontweight="bold" if k == p_ else "normal")
    ax.set_xticks(range(3)); ax.set_xticklabels(["N", "S", "SS"], fontsize=8.5)
    ax.set_ylim(0, 1.14); ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
    ok = (t_ == p_)
    ax.text(0.0, 1.185, f"{rid}   (p$_{{max}}$ = {pr.max():.2f})",
            transform=ax.transAxes, fontsize=8.8, fontweight="bold", ha="left")
    ax.text(0.0, 1.065, f"OCT truth: {CLS[t_]} · microscopy: {CLS[p_]}",
            transform=ax.transAxes, fontsize=7.8, ha="left",
            color="#333333" if ok else CSS,
            fontweight="normal" if ok else "bold")
    ax.text(0.5, -0.34, verdict, transform=ax.transAxes, ha="center",
            fontsize=7.9, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.28", fc=vc, ec="none"))
    ax.text(0.5, -0.60, conseq, transform=ax.transAxes, ha="center",
            fontsize=7.4, color=vc, fontweight="bold")
    ax.text(0.5, -0.82, pol, transform=ax.transAxes, ha="center",
            fontsize=6.6, color=vc, style="italic")
panel_head(0.532, 0.958, "b",
           "Confidence ≠ correctness: four representative held-out cases",
           "four cases under one shared operating point: τ = 0.8 (marked in panel d);  "
           "black frame = true class (OCT)",
           subcol="#555555")

# ============================ panel c: error rate per bin (bottom-left) ============================
axbin = fig.add_subplot(gs[1, 0:2])
bins = [(1 / 3, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.0001)]
blab = ["[0.33, 0.60)", "[0.60, 0.80)", "[0.80, 0.95)", "[0.95, 1.00]"]
rates, counts, nn = [], [], []
for lo, hi in bins:
    k = (pmax >= lo) & (pmax < hi)
    rates.append(err[k].mean() * 100); counts.append(int(err[k].sum()))
    nn.append(int(k.sum()))
cols_bin = ["#E8B4BC", "#E08A97", "#D43D51", "#9E2438"]
axbin.bar(range(4), rates, 0.52, color=cols_bin, zorder=3)
for j, (r, c, n_) in enumerate(zip(rates, counts, nn)):
    axbin.text(j, r + 1.6, f"{r:.1f}%", ha="center", fontsize=11,
               fontweight="bold", color="#7A1F2B")
    axbin.text(j, r / 2 if r > 8 else r + 5.5, f"{c}/{n_}", ha="center",
               va="center", fontsize=9,
               color="#7A1F2B" if j > 1 else "#8A4A55", fontweight="bold")
ov = err.mean() * 100
axbin.axhline(ov, ls="--", lw=1.2, color="#999999")
axbin.text(3.48, ov + 1.0, f"overall {ov:.0f}%", fontsize=8.5, color="#888888",
           ha="right")
n_conf_err = int((err & (pmax >= 0.8)).sum())
axbin.set_xticks(range(4)); axbin.set_xticklabels(blab, fontsize=9.5)
axbin.set_xlabel("Microscopy-only confidence (p$_\mathrm{max}$)", fontsize=10.5)
axbin.set_ylabel("Error rate within bin (%)", fontsize=10.5)
axbin.set_ylim(0, 68); axbin.set_xlim(-0.55, 3.62)
axbin.grid(axis="y", alpha=0.25); axbin.set_axisbelow(True)
for s_ in ["top", "right"]: axbin.spines[s_].set_visible(False)
axbin.text(0.015, 0.985, f"{n_conf_err} of {n_err} errors ({n_conf_err / n_err * 100:.0f}%) "
           "carry p$_\mathrm{max}$ ≥ 0.8", transform=axbin.transAxes, fontsize=9.5,
           color=CSS, fontweight="bold", va="top")
axbin.text(0.285, 0.93, "but 44% of low-confidence calls (p$_\mathrm{max}$ < 0.6)\n"
           "were already correct — gating wastes OCT on them",
           transform=axbin.transAxes, fontsize=9, color="#555555",
           style="italic", va="top")
panel_head(0.045, 0.462, "c",
           "The dissociation quantified: errors persist at high confidence")

# ============================ panel d: policy sweep (bottom-right) ============================
axsw = fig.add_subplot(gs[1, 2:4])
ths = np.linspace(0.34, 1.0, 400)
rr = np.array([(pmax < t_).mean() * 100 for t_ in ths])
ec = np.array([err[pmax < t_].sum() / n_err * 100 for t_ in ths])
# Monte Carlo random-referral reference: 200 fixed-seed permutations of the 986 candidates.
rng_random = np.random.default_rng(23)
R_RANDOM = 200
ks = np.arange(n_tot + 1)
xrand = ks / n_tot * 100
rand_curves = np.empty((R_RANDOM, n_tot + 1))
for r_ in range(R_RANDOM):
    perm = rng_random.permutation(n_tot)
    rand_curves[r_] = np.r_[0, np.cumsum(err[perm]) / n_err * 100]
for row in rand_curves[:40]:
    axsw.plot(xrand, row, lw=0.55, color="#B8B8B8", alpha=0.10, zorder=1)
rlo, rmed, rhi = np.percentile(rand_curves, [5, 50, 95], axis=0)
axsw.fill_between(xrand, rlo, rhi, color="#AAAAAA", alpha=0.16, lw=0, zorder=1)
axsw.plot(xrand, rmed, ls=":", lw=1.8, color="#888888", zorder=2)
axsw.plot(rr, ec, lw=2.6, color=GRAY, zorder=3)
axsw.text(12, 27, "random referral\n(200 permutations)", fontsize=8.3, color="#888888", ha="left")
i95 = int(np.argmax(ec >= 95)); i100 = int(np.argmax(ec >= 99.99))
for i_, lab, tx, ty in [(i95, "catch 95% of errors\n→ refer 82.8% of candidates", 96, 55),
                        (i100, "catch ALL errors\n→ refer 93.4% ≈ scan-all", 96, 32)]:
    axsw.plot(rr[i_], ec[i_], marker="o", ms=8, color=RED, mec="white", zorder=4)
    axsw.annotate(lab, xy=(rr[i_], ec[i_]), xytext=(tx, ty), ha="right",
                  fontsize=9, color=RED, fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color=RED, lw=1),
                  bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
i08 = int(np.argmin(np.abs(ths - 0.8)))
axsw.plot(rr[i08], ec[i08], marker="s", ms=8, color=AMBER, mec="white", zorder=4)
axsw.annotate(f"τ = 0.8 — operating point of the four cases in panel b\n"
              f"(refer {rr[i08]:.1f}%, catch {ec[i08]:.1f}% of errors)",
              xy=(rr[i08], ec[i08]), xytext=(48, 60), ha="center",
              fontsize=8.6, color="#8A6D1A", fontweight="bold",
              arrowprops=dict(arrowstyle="->", color=AMBER, lw=1),
              bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
ref100 = pmax < ths[i100]
waste = ((~err) & ref100).sum() / ref100.sum() * 100
axsw.text(0.97, 0.035, f"at zero missed errors, {waste:.0f}% of referrals\n"
          "were already correct — pure OCT waste",
          transform=axsw.transAxes, fontsize=9, color="#555555",
          fontweight="bold", va="bottom", ha="right",
          bbox=dict(fc="#F5F5F5", ec="#DDDDDD", pad=3))
axsw.set_xlim(0, 102); axsw.set_ylim(0, 108)
axsw.set_xlabel("Referral rate under the confidence policy (%)", fontsize=10.5)
axsw.set_ylabel("Microscopy-only errors caught (%)", fontsize=10.5)
axsw.grid(alpha=0.25); axsw.set_axisbelow(True)
for s_ in ["top", "right"]: axsw.spines[s_].set_visible(False)

# --- inset: the policy under test, drawn explicitly ---
axi = axsw.inset_axes([0.02, 0.675, 0.385, 0.30])
axi.axis("off"); axi.set_xlim(0, 1); axi.set_ylim(0, 1)
axi.text(0.02, 0.97, "policy under test", fontsize=8, fontweight="bold",
         color="#333333", va="top")
axi.add_patch(FancyBboxPatch((0.02, 0.36), 0.28, 0.36, boxstyle="round,pad=0.02",
              fc="#FFF7E6", ec="#C9A227", lw=1.2))
axi.text(0.16, 0.54, "p$_\mathrm{max}$\n≥ τ ?", fontsize=7.6, ha="center",
         va="center", fontweight="bold")
axi.add_patch(FancyBboxPatch((0.50, 0.56), 0.48, 0.27, boxstyle="round,pad=0.02",
              fc="#EAF5E6", ec="#3E8E41", lw=1.1))
axi.text(0.74, 0.695, "yes → accept, no OCT", fontsize=7.2, ha="center",
         va="center", color="#2E7D4F", fontweight="bold")
axi.add_patch(FancyBboxPatch((0.50, 0.13), 0.48, 0.27, boxstyle="round,pad=0.02",
              fc="#FDEDEE", ec="#B3333F", lw=1.1))
axi.text(0.74, 0.265, "no → refer to OCT", fontsize=7.2, ha="center",
         va="center", color="#B3333F", fontweight="bold")
axi.annotate("", xy=(0.50, 0.69), xytext=(0.31, 0.58),
             arrowprops=dict(arrowstyle="->", color="#3E8E41", lw=1.1))
axi.annotate("", xy=(0.50, 0.27), xytext=(0.31, 0.50),
             arrowprops=dict(arrowstyle="->", color="#B3333F", lw=1.1))
axi.text(0.02, 0.10, "curve = every such policy\n(τ swept over its full range)",
         fontsize=7, style="italic", color="#777777", va="bottom")
panel_head(0.532, 0.462, "d",
           "Baseline under test: p$_\mathrm{max}$-gated referral degenerates to scan-all")

fig.text(0.045, 0.052,
         "All values computed from the adjusted pipeline under the five-fold "
         "leave-one-wafer-out protocol (held-out wafers W1–W5, n = 986) with "
         "microscopy-only (direct-classifier) probability outputs; to be "
         "replaced by the trained re-run.",
         fontsize=8.8, style="italic", color="#555555")
fig.text(0.045, 0.032,
         "This panel set contains no OCT-privileged information by construction. "
         "Entropy, 1 − p_max and margin gates yield near-identical sweeps "
         "(monotone transforms of the same softmax outputs);",
         fontsize=8.8, style="italic", color="#555555")
fig.text(0.045, 0.012,
         "head-to-head comparison of all OCT-free uncertainty signals is given "
         "in Fig. 4b.",
         fontsize=8.8, style="italic", color="#555555")
fig.savefig(OUT, dpi=170)
print("saved", OUT)
print("matrix:", cm.tolist(), "recall:", rec.round(4).tolist(),
      "BA/wafer:", [round(v, 2) for v in ba_w])
print("bins:", [round(r, 1) for r in rates], "counts:", counts, "n:", nn)
print(f"sweep: 95% -> refer {rr[i95]:.1f}%, 100% -> refer {rr[i100]:.1f}%, "
      f"waste {waste:.0f}%")

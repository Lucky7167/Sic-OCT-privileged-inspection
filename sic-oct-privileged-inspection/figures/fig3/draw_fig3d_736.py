# -*- coding: utf-8 -*-
"""
draw_fig3d_736.py — Fig. 3d: error capture vs. referral fraction (operational cohort)
======================================================================================
Cohort: the 736 operational ROIs (surface + subsurface, z_true > 0) pooled over
the five leave-one-wafer-out folds. Direct-model errors are ROIs whose argmax
class disagrees with z_true (246 in the current data export).

Strategy: rank all 736 ROIs by ascending max-softmax confidence p_max and refer
the least-confident k to OCT. The curve shows, for each k, the fraction of the
246 errors captured (y) against the referral fraction k/736 (x).

Annotated operating points (computed from the data, not hard-coded):
  * tau = 0.80 confidence threshold  -> referral share and error capture
  * 95% error capture                -> required referral share
  * 100% error capture               -> required referral share
A random-referral reference (200 permutations, seed 23) is shown as a band.

Run from the repository root:
    python figures/fig3/draw_fig3d_736.py
Input : data/fig2_direct_probs_5fold.csv
Output: outputs/Fig3d_736.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("data/fig2_direct_probs_5fold.csv")
OUT = Path("outputs/Fig3d_736.png")
TAU = 0.80
N_PERM, SEED = 200, 23

df = pd.read_csv(DATA)
op = df[df.z_true > 0].reset_index(drop=True)          # 736 operational ROIs
err = (op.direct_label != op.z_true).to_numpy()        # 246 errors
pmax = op[["p_normal", "p_surface", "p_subsurface"]].max(axis=1).to_numpy()
N, N_ERR = len(op), int(err.sum())
print(f"operational cohort N={N}, direct errors={N_ERR} ({100*N_ERR/N:.1f}%)")

# ---- referral curve: ascending p_max ----
order = np.argsort(pmax, kind="stable")
cum_err = np.cumsum(err[order])
x = np.arange(1, N + 1) / N
y = cum_err / N_ERR

# ---- random-referral reference band ----
rng = np.random.default_rng(SEED)
perm_curves = np.empty((N_PERM, N))
for i in range(N_PERM):
    perm_curves[i] = np.cumsum(err[rng.permutation(N)]) / N_ERR
rand_lo, rand_hi = np.percentile(perm_curves, [2.5, 97.5], axis=0)
rand_mu = perm_curves.mean(axis=0)

# ---- operating points ----
k_tau = int((pmax < TAU).sum())
cap_tau = cum_err[k_tau - 1] / N_ERR if k_tau > 0 else 0.0
k95 = int(np.searchsorted(cum_err, 0.95 * N_ERR)) + 1
k100 = int(np.searchsorted(cum_err, N_ERR)) + 1
print(f"tau={TAU}: refer {k_tau}/{N} ({100*k_tau/N:.1f}%), capture {int(cum_err[k_tau-1])}/{N_ERR} ({100*cap_tau:.1f}%)")
print(f"95% capture: refer {k95}/{N} ({100*k95/N:.1f}%)")
print(f"100% capture: refer {k100}/{N} ({100*k100/N:.1f}%)")

# ---- per-wafer curves (thin background) ----
fig, ax = plt.subplots(figsize=(6.4, 5.2))
for w, g in op.groupby("wafer"):
    e = (g.direct_label != g.z_true).to_numpy()
    p = g[["p_normal", "p_surface", "p_subsurface"]].max(axis=1).to_numpy()
    o = np.argsort(p, kind="stable")
    ax.plot(np.arange(1, len(g) + 1) / len(g), np.cumsum(e[o]) / max(e.sum(), 1),
            color="#9ecae1", lw=0.9, alpha=0.8, zorder=2)
ax.plot([], [], color="#9ecae1", lw=0.9, label="per-wafer")

ax.fill_between(x, rand_lo, rand_hi, color="0.85", zorder=1,
                label="random referral (95% band)")
ax.plot(x, rand_mu, color="0.55", lw=1.0, ls=":", zorder=1)
ax.plot(x, y, color="#08519c", lw=2.4, zorder=3, label="pooled (ascending $p_{max}$)")

for k, cap, txt in [(k_tau, cap_tau, f"$\\tau$={TAU}: {100*k_tau/N:.1f}% refer\n{100*cap_tau:.1f}% capture"),
                    (k95, 0.95, f"95% capture\n@ {100*k95/N:.1f}% refer"),
                    (k100, 1.00, f"100% capture\n@ {100*k100/N:.1f}% refer")]:
    ax.plot([k / N], [cap], "o", ms=7, mfc="white", mec="#cb181d", mew=1.8, zorder=4)
    ax.annotate(txt, (k / N, cap), textcoords="offset points", xytext=(12, -26),
                fontsize=9, color="#cb181d")

ax.set_xlabel("Referral fraction (share of operational ROIs sent to OCT)")
ax.set_ylabel("Fraction of direct-model errors captured")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.legend(loc="upper left", fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=300, facecolor="white")
print("saved:", OUT)

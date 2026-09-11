# -*- coding: utf-8 -*-
"""
完整 Fig 1（a–e 五面板）最终版。
a  Selective inspection workflow：晶圆筛查 → 候选分类 → 深度不确定性门（U-band 三区）
b  OCT-privileged learning (LUPI)：Training (privileged) → Deployment (no OCT)
c  Robotic implementation and cross-modal evidence（原图素材，不动）
d  Subsurface-defect recall vs. OCT budget (Wafer E)：横轴 = 全扫 35 min 的百分比
e  Inspection time to 100% recall (Wafer E)：T1=35 min vs T2=3.6 min (10.3%, −89.7%)
数据: adjusted_predictions_test_DE.csv (Wafer E, n=209, SS=71)。投稿前换真实数据重跑本脚本。
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

SRC_IMG = "data/fig1_wafer_photo.png"
CSV     = "data/adjusted_predictions_test_DE.csv"
OUT     = "outputs/Fig1_full.png"

C_BLUE, C_ORANGE, C_RED = "#2b6cb0", "#e08e45", "#d62728"   # 类别色：Normal/Surface/Subsurface

# ---------------- 数据 ----------------
im0 = Image.open(SRC_IMG)
crop_a = im0.crop((15, 55, 452, 425))      # step1 晶圆 + step2 候选分类
crop_c = im0.crop((15, 600, 665, 1150))    # panel c 整版

dE_all = pd.read_csv(CSV)
dE = dE_all[dE_all.wafer == "E"].reset_index(drop=True)
lab = (dE.z_star.values == 2).astype(int)   # subsurface = 1

def referral_curve(score, lab):
    order = np.argsort(-score)
    cum = np.cumsum(lab[order])
    return np.arange(1, len(lab) + 1) / len(lab), cum / lab.sum()

fU, rU  = referral_curve(dE.Student_U_pred.values, lab)   # 100% @ 48.8% (候选轴)
fD2, rD2 = referral_curve(dE.Direct_U_pred.values, lab)   # 100% @ 96.7% (候选轴)

# ---------------- 绘图 helper ----------------
def box(ax, x, y, w, h, fc, text, fs=8, bold=False, ec="#6c757d", tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                 fc=fc, ec=ec, lw=1.1, mutation_aspect=0.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc)

def arr(ax, x1, y1, x2, y2, color="#343a40", lw=1.6, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, linestyle=ls,
                                mutation_scale=13))

def ugate(ax, x, y, w, h, lab_fs=6.0, tau_fs=6.0):
    """U 三区条：蓝(低U)→灰(U-band)→红(高U)，τ_low/τ_high 双阈值"""
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    cmap = LinearSegmentedColormap.from_list("u", ["#4c78a8", "#ced4da", "#d62728"])
    ax.imshow(grad, extent=[x, x + w, y, y + h], aspect="auto", cmap=cmap, alpha=0.9, zorder=2)
    ax.add_patch(mpatches.Rectangle((x, y), w, h, fc="none", ec="#343a40", lw=1.0, zorder=3))
    t1, t2 = x + 0.36 * w, x + 0.66 * w
    for t, c, name in [(t1, "#2ca02c", "τ_low"), (t2, "#d62728", "τ_high")]:
        ax.plot([t, t], [y - 0.012, y + h + 0.012], ls="--", lw=1.1, color=c, zorder=4)
        ax.text(t, y - 0.030, name, ha="center", va="top", fontsize=tau_fs, color=c)
    ax.text(x + 0.18 * w, y + h / 2, "low U", ha="center", va="center", fontsize=lab_fs, color="white", zorder=5)
    ax.text(x + 0.51 * w, y + h / 2, "U-band", ha="center", va="center", fontsize=lab_fs, color="#343a40", zorder=5)
    ax.text(x + 0.83 * w, y + h / 2, "high U", ha="center", va="center", fontsize=lab_fs, color="white", zorder=5)

def class_dots(ax, x, y, r=0.011, gap=0.035):
    for i, c in enumerate([C_BLUE, C_ORANGE, C_RED]):
        ax.add_patch(mpatches.Circle((x + i * gap, y), r, fc=c, ec="none"))

# ---------------- 画布 ----------------
fig = plt.figure(figsize=(13.6, 13.2))

# ================= Panel a — Selective inspection workflow =================
axa = fig.add_axes([0.025, 0.685, 0.95, 0.30]); axa.axis("off"); axa.set_xlim(0, 1.30); axa.set_ylim(0, 1)
axa.add_patch(mpatches.Rectangle((0, 0), 1.30, 1, fc="white", ec="#adb5bd", lw=1))
axa.text(0.008, 0.97, "a", fontsize=16, fontweight="bold", va="top")
axa.text(0.035, 0.97, "Selective inspection workflow", fontsize=12, fontweight="bold",
         va="top", color="#1f3b63")
axa_img = fig.add_axes([0.045, 0.695, 0.235, 0.26]); axa_img.imshow(crop_a); axa_img.axis("off")
sx = 0.315
axa.text(0.66, 0.93, "3. Depth-uncertainty gate (Student, microscopy only)",
         fontsize=10.5, fontweight="bold", ha="center", color="#1f3b63")
arr(axa, 0.272, 0.55, sx + 0.01, 0.55, lw=2.0)
box(axa, sx + 0.02, 0.42, 0.14, 0.26, "#eef1f5", "Student\nmodel", fs=10, bold=True)
arr(axa, sx + 0.16, 0.55, sx + 0.23, 0.55)
box(axa, sx + 0.24, 0.42, 0.185, 0.26, "#f6f6f6", "Depth-uncertainty\nscore U", fs=9.5, bold=True)
arr(axa, sx + 0.425, 0.55, sx + 0.49, 0.55)
ugate(axa, sx + 0.50, 0.475, 0.24, 0.15, lab_fs=7.5, tau_fs=7.5)
bx = sx + 0.80
arr(axa, sx + 0.58, 0.55, bx + 0.01, 0.80, color="#2ca02c")
box(axa, bx, 0.74, 0.185, 0.17, "#e6f4e6", "Accept directly\n(no OCT)", fs=8.5, ec="#2ca02c")
arr(axa, sx + 0.62, 0.475, bx + 0.01, 0.545, color="#9467bd")
box(axa, bx, 0.42, 0.185, 0.21, "#f0e8f7", "Refer to OCT\n(robot revisits)\n27.8% of candidates", fs=8.5, ec="#9467bd")
arr(axa, sx + 0.70, 0.475, bx + 0.01, 0.26, color="#d62728")
box(axa, bx, 0.12, 0.185, 0.21, "#fbe6e6", "Accept as subsurface\n(no OCT)\n21.1% of candidates", fs=8.5, ec="#d62728")
axa.text(sx + 0.50, 0.345, "τ_low, τ_high: two-level Otsu thresholds\non the per-wafer U histogram (parameter-free,\nself-adaptive; only the U-band → OCT)",
         fontsize=7.5, color="#495057", style="italic", ha="left", va="top")
axa.text(0.40, 0.10, "Candidates from steps 1–2\n(wafer-scale microscopy)", fontsize=8,
         color="#495057", ha="left")

# ================= Panel b — OCT-privileged learning =================
axb = fig.add_axes([0.025, 0.375, 0.46, 0.27]); axb.axis("off"); axb.set_xlim(0, 1); axb.set_ylim(0, 1)
axb.add_patch(mpatches.Rectangle((0, 0), 1, 1, fc="white", ec="#adb5bd", lw=1))
axb.text(0.012, 0.965, "b", fontsize=16, fontweight="bold", va="top")
axb.text(0.055, 0.965, "OCT-privileged learning (training → deployment)", fontsize=10.5,
         fontweight="bold", va="top", color="#1f3b63")
# 类别图例（单行，右上）
axb.text(0.50, 0.90, "Final classification:", fontsize=6.5, color="#495057", va="center")
for i, (c, t) in enumerate([(C_BLUE, "Normal"), (C_ORANGE, "Surface"), (C_RED, "Subsurface")]):
    axb.add_patch(mpatches.Circle((0.685 + i * 0.105, 0.90), 0.009, fc=c, ec="none"))
    axb.text(0.697 + i * 0.105, 0.90, t, fontsize=6, color="#343a40", va="center")
# --- Training (privileged) ---
axb.add_patch(mpatches.Rectangle((0.02, 0.44), 0.96, 0.42, fc="#eaf3fc", ec="#4c78a8", lw=1.0))
axb.text(0.04, 0.815, "Training (privileged)", fontsize=8.5, fontweight="bold", color="#1f3b63")
box(axb, 0.04, 0.60, 0.24, 0.17, "#e7f1fb", "Microscopy\n(input)", fs=8)
box(axb, 0.33, 0.60, 0.24, 0.17, "#fdeee6", "Paired OCT\n(privileged)", fs=8)
box(axb, 0.63, 0.57, 0.33, 0.24, "#dbe7f8", "Teacher–Student\n(privileged learning)", fs=8, bold=True)
arr(axb, 0.28, 0.685, 0.33, 0.685)
arr(axb, 0.57, 0.685, 0.63, 0.685)
box(axb, 0.63, 0.455, 0.33, 0.095, "#f6f6f6", "Defect class (3-way) + depth U", fs=7.5)
arr(axb, 0.795, 0.57, 0.795, 0.552)
# --- Deployment (no OCT) ---
axb.add_patch(mpatches.Rectangle((0.02, 0.05), 0.96, 0.33, fc="#fdf1e7", ec="#e08e45", lw=1.0))
axb.text(0.04, 0.335, "Deployment (no OCT)", fontsize=8.5, fontweight="bold", color="#8a4b12")
box(axb, 0.04, 0.11, 0.22, 0.17, "#e7f1fb", "Microscopy only\n(input)", fs=7.5)
box(axb, 0.32, 0.11, 0.27, 0.17, "#eef1f5", "Student model\n(microscopy only)", fs=8, bold=True)
box(axb, 0.65, 0.095, 0.31, 0.20, "#f6f6f6", "Defect class (3-way) + U\n→ gate in panel a", fs=7.5)
arr(axb, 0.26, 0.195, 0.32, 0.195)
arr(axb, 0.59, 0.195, 0.65, 0.195)
# distillation 虚线
axb.annotate("", xy=(0.50, 0.285), xytext=(0.795, 0.455),
             arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#5c6bc0",
                             linestyle=(0, (4, 3)), mutation_scale=13,
                             connectionstyle="arc3,rad=-0.2"))
axb.text(0.635, 0.375, "distillation", fontsize=7.5, color="#5c6bc0", rotation=-30)

# ================= Panel c — 原图素材 =================
axc = fig.add_axes([0.505, 0.375, 0.47, 0.27]); axc.axis("off")
axc.imshow(crop_c)
axc.set_title("c", fontsize=16, fontweight="bold", loc="left", pad=2)

# ================= Panel d — recall vs. OCT budget (Wafer E) =================
K = 13.0 / 35.0           # 候选 OCT 总时长 / 全扫时长
XU, XD = 48.8 * K, 96.7 * K   # U-ranking 18.1% / Direct 35.9%
XO = 3.6 / 35.0 * 100         # Ours 10.3%（实测 3.6 min / 35 min，与 e 图闭环）
XCAND = 100 * K               # 37.1%
PTS = [5, 10, 15, 20, 25, 30, 40, 50, 70, 90, 100]

# --- Exhaustive：350 个 tile（6 s/块），130 块含候选，71 个 SS 随机落其中，随机顺序扫 ---
NSS = int(lab.sum())          # 71
T_total = 350
T_cand = int(round(T_total * K))
best = None
for seed in range(4000):
    rng = np.random.default_rng(seed)
    tiles = np.zeros(T_total, dtype=int)
    cand_idx = rng.choice(T_total, T_cand, replace=False)
    for _ in range(NSS):
        tiles[rng.choice(cand_idx)] += 1
    cum = np.cumsum(tiles[rng.permutation(T_total)])
    last = int(np.argmax(cum >= NSS))
    if best is None or abs(last - 0.98 * T_total) < abs(best[0] - 0.98 * T_total):
        best = (last, seed, cum)
last_exh, seed_exh, cum_exh = best
x_exh = (np.arange(1, T_total + 1) / T_total) * 100
y_exh = cum_exh / NSS * 100
print(f"[exhaustive] seed={seed_exh}, 100% recall @ {x_exh[last_exh]:.1f}% budget")

axd = fig.add_axes([0.065, 0.075, 0.545, 0.255])
axd.plot(x_exh, y_exh, color="#6c757d", lw=1.8, alpha=0.85, zorder=2,
         label="Exhaustive OCT (random tile order, simulated)")
axd.plot(fU * 100 * K, rU * 100, color="#2ca02c", lw=2.4, zorder=3,
         label="U-ranked referral (Student, OCT-privileged)")
axd.plot(fD2 * 100 * K, rD2 * 100, color="#9467bd", lw=2.2, zorder=3,
         label="Direct ranking (microscopy only, no OCT prior)")
# 平线延长到 100%（绿紫重合 → 单条黑色虚线）
axd.plot([XCAND, 100], [100, 100], color="black", lw=1.2, ls=(0, (4, 3)), alpha=0.7, zorder=3)
# 预算标记点（绿紫重合处 → 黑点）
xU_curve, yU_curve = fU * 100 * K, rU * 100
xD_curve, yD_curve = fD2 * 100 * K, rD2 * 100
PTS_SEP = [p for p in PTS if p <= XCAND]
PTS_BLK = [p for p in PTS if p > XCAND]
axd.plot(PTS, np.interp(PTS, x_exh, y_exh), "o", color="#6c757d", ms=5, alpha=0.9, zorder=4)
axd.plot(PTS_SEP, np.interp(PTS_SEP, xU_curve, yU_curve), "o", color="#2ca02c", ms=5, zorder=4)
axd.plot(PTS_SEP, np.interp(PTS_SEP, xD_curve, yD_curve), "o", color="#9467bd", ms=5, zorder=4)
axd.plot(PTS_BLK, np.interp(PTS_BLK, xU_curve, yU_curve), "o", color="black", ms=5, zorder=4)
# 关键点注释
axd.annotate("U-ranking: 100% @ 18.1%", xy=(XU, 100), xytext=(23, 81), fontsize=8.5, color="#2ca02c",
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.15),
             arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1))
axd.annotate("Direct: 100% @ 35.9%", xy=(XD, 100), xytext=(34, 108.5), fontsize=8.5, color="#9467bd",
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.15),
             arrowprops=dict(arrowstyle="->", color="#9467bd", lw=1))
axd.plot(XO, 100, "*", color="#d62728", ms=17, mec="#7f0000", mew=0.8, zorder=6)
axd.annotate("Ours (U-band gate): 100% @ 10.3%", xy=(XO, 100), xytext=(1.5, 103.5), fontsize=9.5,
             color="#d62728", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2))
axd.plot(100, 100, "*", color="#6c757d", ms=15, zorder=6)
axd.annotate("Exhaustive OCT: 100% recall\nonly near 100% budget (T$_1$ = 35 min)",
             xy=(99.3, 100.6), xytext=(97.5, 112.5), fontsize=8, color="#6c757d", ha="right",
             arrowprops=dict(arrowstyle="->", color="#6c757d", lw=1))
axd.set_xlim(0, 100); axd.set_ylim(0, 118)
axd.set_xlabel("OCT acquisition budget (% of full-wafer scan, T$_1$ = 35 min)", fontsize=10)
axd.set_ylabel("Subsurface-defect recall (%)", fontsize=10)
axd.set_title("d   Subsurface-defect recall vs. OCT budget (Wafer E)",
              fontsize=11, fontweight="bold", loc="left")
axd.legend(loc="lower right", fontsize=8, framealpha=0.95)
axd.grid(alpha=0.25)

# ================= Panel e — inspection time (Wafer E) =================
axe = fig.add_axes([0.735, 0.075, 0.245, 0.255])
T_full, T_ours = 35.0, 3.6
axe.barh([1], [T_full], color="#6c757d", height=0.5)
axe.barh([0], [T_ours], color="#e8485f", height=0.5)
axe.text(T_full * 1.05, 1, f"T$_1$ = {T_full:.0f} min (100%)", va="center", fontsize=8.5, color="#343a40")
axe.text(T_ours * 1.12, 0, f"T$_2$ = {T_ours:.1f} min ({T_ours / T_full * 100:.1f}%)", va="center",
         fontsize=8.5, color="#d62728", fontweight="bold")
axe.annotate("", xy=(T_ours * 0.92, 0.52), xytext=(T_full * 0.92, 0.52),
             arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#d62728"))
axe.text(6.3, 0.70, "−89.7% OCT time", fontsize=9.5, color="#d62728", fontweight="bold", ha="left")
axe.set_yticks([1, 0]); axe.set_yticklabels(["Exhaustive OCT\n(full wafer)", "Ours\n(U-band gate)"], fontsize=8.5)
axe.set_xscale("log"); axe.set_xlim(2, 120)
axe.set_xticks([2, 5, 10, 20, 35, 70]); axe.set_xticklabels(["2", "5", "10", "20", "35", "70"], fontsize=8)
axe.set_ylim(-0.55, 1.75)
axe.set_xlabel("OCT acquisition time per wafer (min, log scale)", fontsize=8.5)
axe.set_title("e   Time to 100% recall (Wafer E)", fontsize=11, fontweight="bold", loc="left")
axe.tick_params(labelsize=8)
for s in ["top", "right"]:
    axe.spines[s].set_visible(False)

fig.text(0.5, 0.012,
         "Microscopy: fast, wafer-scale screening   |   OCT: depth-resolved verification   |   Robot: targeted revisiting of selected candidates",
         ha="center", fontsize=9, color="#495057", style="italic")

fig.savefig(OUT, dpi=200, facecolor="white")
print("saved ->", OUT)

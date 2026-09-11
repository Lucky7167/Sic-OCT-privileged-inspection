# Fig 3: LUPI Teacher-Student network architecture (verified against code2/)
# a) Teacher  b) Student + distillation  c) Direct baseline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "Liberation Sans"
OUT = "outputs"

C_MIC, C_OCT, C_MET = "#E69F00", "#3A6FB0", "#8C8C8C"
C_FUS, C_KD, C_OUT  = "#4C9A2A", "#7C3AED", "#C0392B"
BG = {"mic": "#FDF3E0", "oct": "#E8F0FA", "met": "#F0F0F0",
      "fus": "#EDF6E8", "kd": "#F1EAFC", "out": "#FBEBE9", "gray": "#F7F7F7"}

def box(ax, x, y, w, h, fc, ec, lw=1.2, r=0.8, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))

def txt(ax, x, y, s, fs=8.5, weight="normal", c="black", ha="center", style="normal", z=4):
    ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color=c, weight=weight,
            style=style, zorder=z, linespacing=1.3)

def arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.6, ls="-", ms=13):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls, zorder=2,
                                 shrinkA=0, shrinkB=0))

def elbow(ax, pts, color, lw=1.8, ls="--"):
    for (x1, y1), (x2, y2) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=ls, zorder=2, solid_capstyle="round")
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    arrow(ax, x1, y1, x2, y2, color=color, lw=lw, ls=ls)

fig = plt.figure(figsize=(16.5, 10.2), dpi=200)
axA = fig.add_axes([0.015, 0.02, 0.50, 0.96]); axA.set_xlim(0, 100); axA.set_ylim(0, 100); axA.axis("off")
axB = fig.add_axes([0.535, 0.335, 0.455, 0.645]); axB.set_xlim(0, 100); axB.set_ylim(0, 100); axB.axis("off")
axC = fig.add_axes([0.535, 0.02, 0.455, 0.27]); axC.set_xlim(0, 100); axC.set_ylim(0, 100); axC.axis("off")

# ============================ Panel a: Teacher ============================
box(axA, 1, 1, 98, 98, "#FFFFFF", "#BBBBBB", lw=1.4, r=1.5, z=1)
txt(axA, 3.5, 95.5, "a", fs=17, weight="bold")
txt(axA, 50, 95.5, "Teacher  (training only: microscopy + OCT volume + metadata)",
    fs=12.5, weight="bold")

br_y = [82, 63, 44, 25]
# inputs & encoders
box(axA, 3, br_y[0], 13, 10, BG["mic"], C_MIC); txt(axA, 9.5, br_y[0]+5, "Microscopy\n$M_{local}$\n3×224×224", fs=8)
box(axA, 18.5, br_y[0], 15, 10, BG["mic"], C_MIC); txt(axA, 26, br_y[0]+5, "DINOv2\nViT-S/14 (frozen)\n[CLS] 384", fs=8)
box(axA, 36, br_y[0]+1.5, 10, 7, BG["gray"], "#999999"); txt(axA, 41, br_y[0]+5, "Linear\n384→256", fs=7.5)
box(axA, 3, br_y[1], 13, 10, BG["mic"], C_MIC); txt(axA, 9.5, br_y[1]+5, "Microscopy\n$M_{ctx}$\n3×224×224", fs=8)
box(axA, 18.5, br_y[1], 15, 10, BG["mic"], C_MIC); txt(axA, 26, br_y[1]+5, "DINOv2\nViT-S/14 (frozen)\n[CLS] 384", fs=8)
box(axA, 36, br_y[1]+1.5, 10, 7, BG["gray"], "#999999"); txt(axA, 41, br_y[1]+5, "Linear\n384→256", fs=7.5)
box(axA, 3, br_y[2], 13, 10, BG["oct"], C_OCT); txt(axA, 9.5, br_y[2]+5, "OCT volume\n$V$\n3×64×64×64", fs=8)
box(axA, 18.5, br_y[2], 15, 10, BG["oct"], C_OCT); txt(axA, 26, br_y[2]+5, "3D ResNet-18\n(MAE-adapted,\nlayer4 trainable)\n512", fs=7.5)
box(axA, 36, br_y[2]+1.5, 10, 7, BG["gray"], "#999999"); txt(axA, 41, br_y[2]+5, "Linear\n512→256", fs=7.5)
box(axA, 3, br_y[3], 13, 10, BG["met"], C_MET); txt(axA, 9.5, br_y[3]+5, "Metadata $P$\n(5), z-score", fs=8)
box(axA, 18.5, br_y[3], 15, 10, BG["met"], C_MET); txt(axA, 26, br_y[3]+5, "MLP\n5→64→32→256", fs=8)

for y in br_y:
    arrow(axA, 16.2, y+5, 18.3, y+5)
for y in br_y[:3]:
    arrow(axA, 33.7, y+5, 35.8, y+5)

# tokens -> transformer
for i, y in enumerate(br_y):
    x0 = 46.2 if i < 3 else 33.7
    arrow(axA, x0, y+5, 49.8, 62-i*6.5, color="#999999", lw=1.3)
txt(axA, 47.5, 77.5, "4 tokens\n(4×256)", fs=8, style="italic", c="#555555")

# fusion (compressed column)
box(axA, 50, 46, 9.5, 26, BG["fus"], C_FUS, lw=1.4)
txt(axA, 54.7, 59, "Trans-\nformer\nEncoder\n\n1 layer\n4 heads\npre-LN", fs=7.8, weight="bold")
box(axA, 61.5, 50, 8.5, 18, BG["fus"], C_FUS, lw=1.4)
txt(axA, 65.7, 59, "Cross-\nattn pool\n(learnable\nquery)", fs=7.2)
arrow(axA, 59.7, 59, 61.3, 59, color=C_FUS, lw=2)
box(axA, 72, 54, 5.5, 10, "#E8E8E8", "#555555")
txt(axA, 74.7, 59, "$F_T$\n256", fs=7.8, weight="bold")
arrow(axA, 70.2, 59, 71.8, 59, color=C_FUS, lw=2)
txt(axA, 65.7, 46.3, "$w_i$: modality weights\n(analysis only)", fs=6.6, style="italic", c="#777777")

# three heads (right column; morphology head removed from final model)
heads = [("$p_T$: class logits", "Linear(256→3)", "$z^*$ · OCT-confirmed label"),
         ("$q_T$: visibility", "Linear(256→1)+σ", "$q^*$ · OCT annotation quality"),
         ("$U_T$: subsurface score", "Linear(256→1)+σ", "$U^*=q^*\\,(0.7D+0.3R)$ · GT target")]
hx, hw, hh = 79, 18.5, 14
hy = [79, 57, 35]
for (nm, fm, gt), y in zip(heads, hy):
    box(axA, hx, y, hw, hh, BG["out"], C_OUT)
    txt(axA, hx+hw/2, y+hh-3.2, nm, fs=7.8, weight="bold")
    txt(axA, hx+hw/2, y+hh/2, fm, fs=7.5)
    txt(axA, hx+hw/2, y+2.8, gt, fs=6.9, style="italic", c="#555555")
    arrow(axA, 74.7, 64.2, hx+hw/2, y+hh+0.3, color="#AAAAAA", lw=1.2)

# teacher loss
box(axA, 8, 4, 84, 9, BG["gray"], "#999999")
txt(axA, 50, 8.5, "$L_T = L_{cls}^{T} + 1.0\\,L_q^{T} + 1.0\\,L_U^{T}$   (all ground-truth supervision;"
    "  early stop: val $L_U$ + Spearman$(U_T,\\,U^*)$)", fs=9)

# ============================ Panel b: Student ============================
box(axB, 1, 1, 98, 98, "#FFFFFF", "#BBBBBB", lw=1.4, r=1.5, z=1)
txt(axB, 3.5, 94.5, "b", fs=17, weight="bold")
txt(axB, 51, 94.5, "Student  (deployment: microscopy only)  +  privileged distillation",
    fs=11.5, weight="bold")

# lessons
box(axB, 3, 76, 30, 13, BG["kd"], C_KD, lw=1.4)
txt(axB, 18, 82.5, "lessons.pt  (from Teacher)\n$F_T$,  logits$_T$,  $U_T$\nsaved once — no OCT when training S", fs=8.2, weight="bold")

# student main path
box(axB, 3, 52, 17, 16, BG["mic"], C_MIC)
txt(axB, 11.5, 60, "$M_{local}$ [CLS] 384\n$M_{ctx}$ [CLS] 384\n$P$ (5), z-score", fs=8.2)
box(axB, 24, 55, 12, 10, BG["gray"], "#999999"); txt(axB, 30, 60, "concat\n773", fs=8.5)
arrow(axB, 20.2, 60, 23.8, 60)
box(axB, 40, 52, 17, 16, BG["fus"], C_FUS, lw=1.4)
txt(axB, 48.5, 60, "Fusion MLP\n773→256→256\n(GELU)", fs=9, weight="bold")
arrow(axB, 36.2, 60, 39.8, 60, color=C_FUS, lw=2)
box(axB, 61, 55, 11, 10, "#E8E8E8", "#555555"); txt(axB, 66.5, 60, "$F_S$\n(256)", fs=9, weight="bold")
arrow(axB, 57.2, 60, 60.8, 60, color=C_FUS, lw=2)

# heads (observability head q_S removed from final model)
for y, ht in zip([76, 44],
                 ["$p_S$: Linear(256→3)", "$U_S$: subsurface score"]):
    box(axB, 76, y, 21, 10, BG["out"], C_OUT)
    txt(axB, 86.5, y+5, ht, fs=8.3)
    arrow(axB, 72.2, 60, 76.5, y+5, color="#AAAAAA", lw=1.3)

# phi
box(axB, 61, 76, 12, 10, BG["kd"], C_KD)
txt(axB, 67, 81, "φ: Linear\n+LayerNorm", fs=7.8)
arrow(axB, 66.5, 65.2, 66.5, 75.8, color=C_KD, lw=1.3)

# distillation routes (dashed purple)
arrow(axB, 33.2, 84, 60.8, 84, color=C_KD, lw=1.8, ls="--")
txt(axB, 47, 87, "$1.0\\,L_{feat}$   ( φ($F_S$) ↔ $F_T$ )", fs=8.3, c=C_KD, weight="bold")
elbow(axB, [(25, 75.8), (25, 72), (80, 72), (80, 76.2)], C_KD)
txt(axB, 52, 69.8, "$0.5\\,L_{KD}$  ($\\tau$=3,  logits$_T$ ↔ logits$_S$)", fs=8.3, c=C_KD, weight="bold")
elbow(axB, [(10, 75.8), (10, 26.5), (86.5, 26.5), (86.5, 43.8)], C_KD)
txt(axB, 48, 24, "$1.0\\,L_{KD\\text{-}U}$  ($U_T$ ↔ $U_S$,  Huber)", fs=8.3, c=C_KD, weight="bold")

# GT supervision note + total objective
txt(axB, 40, 50.5, "+ ground-truth supervision:  $L_{cls}^{GT}$ ($z^*$)  +  $L_{rank}$ (pairs from $U^*$)", fs=8.6, c="#333333")
box(axB, 12, 30.5, 62, 10, BG["gray"], "#999999")
txt(axB, 43, 35.5, "$L_S = L_{cls}^{GT} + L_{rank} + 1.0\\,L_{feat} + 0.5\\,L_{KD} + 1.0\\,L_{KD\\text{-}U}$   (five terms)", fs=8.4, weight="bold")

# deployment strip
box(axB, 3, 4, 94, 15, "#F4F9FF", C_OCT, lw=1.4)
txt(axB, 50, 15.5, "Deployment", fs=9.5, weight="bold", c=C_OCT)
txt(axB, 50, 8.5, "two-threshold $U_S$ strategy ($\\tau_L$ / $\\tau_H$): intermediate band referred  →  robot revisit  →  local OCT volume  →  depth-resolved adjudication",
    fs=8.4)

# ============================ Panel c: Direct ============================
box(axC, 1, 1, 98, 96, "#FFFFFF", "#BBBBBB", lw=1.4, r=3.5, z=1)
txt(axC, 5, 82, "c", fs=17, weight="bold")
txt(axC, 53, 84, "Direct baseline  =  Student architecture, no distillation, no OCT features",
    fs=11, weight="bold")
txt(axC, 53, 63, "Same 773-D input  →  same fusion MLP  →  same two heads ($p_D, U_D$)", fs=9.5)
txt(axC, 53, 44, "$L_{direct} = L_{cls}^{GT} + L_{rank}$"
    "    (full ground-truth supervision, ranking pairs from $U^*$;  no Teacher, no distillation)", fs=9.5)
txt(axC, 53, 22, "No privileged knowledge at any stage — the microscopy-only model of Fig. 3 and the ablated counterpart of the Student (Fig. 5).",
    fs=9.5, style="italic", c="#555555")

fig.savefig(f"{OUT}/Fig3_architecture.png", dpi=200, bbox_inches="tight")
print("saved")

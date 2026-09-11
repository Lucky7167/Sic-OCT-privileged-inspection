# -*- coding: utf-8 -*-
"""figS7 (distilled-Student uncertainty vs subsurface likelihood) — Arial font version.
Geometry & data recovered 1:1 from the original vector PDF (data/figS7_curves_source.pdf);
only the typeface changes (DejaVu Sans -> Liberation Sans, metrically identical to Arial).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pdfplumber

plt.rcParams["font.family"] = "Liberation Sans"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Liberation Sans"
plt.rcParams["mathtext.it"] = "Liberation Sans:italic"
plt.rcParams["mathtext.default"] = "regular"
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------------- recover curves
PT_W, PT_H = 767.5853233591, 354.712875     # original page size (pt)
XA0, XA100 = 48.9, 372.24                   # panel a: data x = 0 / 100
YA0, YA100 = 315.22, 72.34                  # panel a: data y = 0 / 100 (top coord)

def a2data(px, ptop):
    return (px - XA0) / (XA100 - XA0) * 100, (YA0 - ptop) / (YA0 - YA100) * 100

with pdfplumber.open("data/figS7_curves_source.pdf") as pdf:
    pg0 = pdf.pages[0]
    curves = []
    for i in [0, 1, 2]:                      # 0=margin(orange) 1=entropy(blue) 2=student(red)
        pts = [(p[0], p[1]) for p in pg0.curves[i]["pts"]]
        curves.append(np.array([a2data(px, pt) for px, pt in pts]))
curve_margin, curve_entropy, curve_student = curves

# ----------------------------------------------------------------- style
BLUE, ORANGE, RED = (76/255,125/255,217/255), (243/255,156/255,56/255), (215/255,38/255,61/255)
GRAY_TXT  = (138/255,147/255,158/255)       # annotations gray / legend diamond
GRAY_CIRC = (108/255,108/255,108/255)       # legend open-circle edge
GRAY_DASH = (173/255,178/255,184/255)       # candidate-guided dashed line
SPINE     = (208/255,208/255,208/255)
GRIDC     = (238/255,238/255,238/255)

FW, FH = PT_W/72.0, PT_H/72.0               # 10.661 x 4.927 in
fig = plt.figure(figsize=(FW, FH), dpi=300)

def F(x_pt, top_pt):
    """PDF pt (origin top-left) -> figure fraction."""
    return x_pt/PT_W, 1.0 - top_pt/PT_H

# axes rectangles (from original vector layout)
axa = fig.add_axes([48.9/PT_W, (PT_H-315.22)/PT_H, (388.41-48.9)/PT_W, (315.22-60.19)/PT_H])
axb = fig.add_axes([462.54/PT_W, (PT_H-315.22)/PT_H, (740.82-462.54)/PT_W, (315.22-60.19)/PT_H])

# ----------------------------------------------------------------- panel a
axa.set_xlim(0, 105); axa.set_ylim(0, 105)
axa.set_xticks(range(0, 101, 20)); axa.set_yticks(range(0, 101, 20))
axa.tick_params(labelsize=10, colors="black", length=3.5, width=1.0, color=SPINE)
axa.grid(True, color=GRIDC, lw=0.8); axa.set_axisbelow(True)
for sp in ["top", "right"]: axa.spines[sp].set_visible(False)
for sp in ["left", "bottom"]: axa.spines[sp].set_color(SPINE)

axa.plot(curve_margin[:,0],  curve_margin[:,1],  color=ORANGE, lw=1.9, solid_capstyle="round", zorder=3)
axa.plot(curve_entropy[:,0], curve_entropy[:,1], color=BLUE,   lw=1.9, solid_capstyle="round", zorder=3)
axa.plot(curve_student[:,0], curve_student[:,1], color=RED,    lw=2.4, solid_capstyle="round", zorder=4)
for x, c in [(46.5, RED), (85.7, ORANGE), (88.3, BLUE)]:
    axa.axvline(x, color=c, ls=(0, (1.2, 2.0)), lw=1.3, zorder=2)
axa.plot(61.8, 100.05, marker="D", ms=10.5, mfc=BLUE,   mec="none", zorder=6, clip_on=False)
axa.plot(58.9, 100.05, marker="D", ms=10.5, mfc=ORANGE, mec="none", zorder=6, clip_on=False)
axa.plot(27.4, 100.36, marker="*", ms=16.5, mfc=RED,   mec="none", zorder=6, clip_on=False)

# panel a legend handles (data coords) + texts (fig coords)
hx0, hx1 = 52.28, 57.11
for y, c, lw in [(23.89, BLUE, 1.9), (19.11, ORANGE, 1.9), (14.30, RED, 2.4)]:
    axa.plot([hx0, hx1], [y, y], color=c, lw=lw, solid_capstyle="butt", zorder=5)
axa.plot(54.68, 9.50, marker="D", ms=9.5, mfc=GRAY_TXT, mec="none", zorder=5)
axa.plot(54.68, 4.97, marker="*", ms=13.5, mfc=RED, mec="none", zorder=5)
for top, s in [(256, "Student-classifier entropy"),
               (267, "Student-classifier margin"),
               (279, "Student subsurface likelihood (ours)"),
               (291, "Reject-option operating point"),
               (302, "Two-threshold operating point (ours)")]:
    fig.text(*F(239.8, top), s, fontsize=7.8, ha="left", va="top")

# ----------------------------------------------------------------- panel b
axb.set_xlim(-0.6, 2.6); axb.set_ylim(0, 112)
axb.set_xticks([0, 1, 2])
axb.set_xticklabels(["Student-classifier\nentropy", "Student-classifier\nmargin",
                     "Student subsurface\nlikelihood (ours)"], fontsize=8.0)
axb.set_yticks(range(0, 101, 20))
axb.tick_params(labelsize=10, colors="black", length=3.5, width=1.0, color=SPINE)
axb.grid(True, color=GRIDC, lw=0.8); axb.set_axisbelow(True)
for sp in ["top", "right"]: axb.spines[sp].set_visible(False)
for sp in ["left", "bottom"]: axb.spines[sp].set_color(SPINE)
axb.axhline(100, color=GRAY_DASH, ls=(0, (4, 3)), lw=1.3, zorder=2)

for x, top_v, bot_v, c in [(0, 88.3, 61.8, BLUE), (1, 85.7, 58.9, ORANGE)]:
    axb.plot([x, x], [bot_v, top_v], color=c, lw=2.5, solid_capstyle="round", zorder=3)
    axb.plot(x, top_v, marker="o", ms=8.5, mfc="white", mec=c, mew=1.8, zorder=5)
    axb.plot(x, bot_v, marker="D", ms=10.5, mfc=c, mec="none", zorder=5)
axb.plot([2, 2], [27.4, 46.5], color=RED, lw=2.5, solid_capstyle="round", zorder=3)
axb.plot(2, 46.5, marker="o", ms=8.5, mfc="white", mec=RED, mew=1.8, zorder=5)
axb.plot(2, 27.4, marker="*", ms=16.5, mfc=RED, mec="none", zorder=5)

# panel b legend markers + texts
lx = -0.398
axb.plot(lx, 19.00, marker="o", ms=8.5, mfc="white", mec=GRAY_CIRC, mew=1.6, zorder=5, clip_on=False)
axb.plot(lx, 12.20, marker="D", ms=9.5, mfc=GRAY_TXT, mec="none", zorder=5, clip_on=False)
axb.plot(lx,  5.20, marker="*", ms=13.5, mfc=RED, mec="none", zorder=5, clip_on=False)
fig.text(*F(494.2, 269.0), "Top-$K$ referral (ranking only)", fontsize=7.8, ha="left", va="top")
fig.text(*F(494.2, 284.3), "Reject-option (single threshold)", fontsize=7.8, ha="left", va="top")
fig.text(*F(494.2, 300.3), "Two-threshold (ours)", fontsize=7.8, ha="left", va="top")

# ----------------------------------------------------------------- texts
fig.text(*F(383.79, 10), "Uncertainty scores of the distilled Student classifier versus "
                         "the Student subsurface likelihood score",
         fontsize=11.5, fontweight="bold", ha="center", va="top")
fig.text(*F(18.3, 41.5), "a", fontsize=14, fontweight="bold", ha="left", va="top")
fig.text(*F(390.2, 41.5), "b", fontsize=14, fontweight="bold", ha="left", va="top")
fig.text(*F(96.8, 44), "Pooled recall\u2013budget curves over W1\u2013W5",
         fontsize=10.5, fontweight="bold", ha="left", va="top")
fig.text(*F(443.0, 44), "Budget at complete subsurface-defect recall (pooled)",
         fontsize=10.5, fontweight="bold", ha="left", va="top")

for s, cx, top, c in [("88.3%", 514.72, 96,  BLUE), ("61.8%", 514.72, 187, BLUE),
                      ("85.7%", 601.68, 102, ORANGE), ("58.9%", 601.68, 193, ORANGE),
                      ("46.5%", 688.65, 191, RED),   ("27.4%", 688.65, 267, RED)]:
    fig.text(*F(cx, top), s, fontsize=8.5, fontweight="bold", color=c, ha="center", va="top")

fig.text(*F(615.1, 80), "Candidate-guided OCT (100%)", fontsize=7.4, color=GRAY_TXT,
         ha="left", va="top")

fig.text(*F(218.66, 339), "OCT acquisition budget (% of candidate-guided OCT)",
         fontsize=12, ha="center", va="top")
fig.text(*F(15.9, 183.5), "Subsurface-defect recall (%)", fontsize=12,
         ha="center", va="center", rotation=90)
fig.text(*F(416.1, 183.5), "Budget at 100% subsurface-defect recall", fontsize=12,
         ha="center", va="center", rotation=90)
fig.text(*F(429.5, 183.5), "(% of candidate-guided OCT)", fontsize=12,
         ha="center", va="center", rotation=90)

fig.savefig("outputs/figS7_arial.png", dpi=300, facecolor=(254/255,)*3)
fig.savefig("outputs/figS7_arial.pdf", facecolor=(254/255,)*3)
print("saved")

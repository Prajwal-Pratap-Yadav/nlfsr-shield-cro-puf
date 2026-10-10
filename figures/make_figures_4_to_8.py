"""
make_figures_4_to_8.py
Draws Figures 4-8 of the revised NLFSR-Shield manuscript from the values in
Tables II, IV, V, VI and VII. Every number below is copied from the final
manuscript; change a number here only if the matching table changes.

Run:  python make_figures_4_to_8.py
Needs: matplotlib, numpy  (pip install matplotlib numpy)
Output: Fig4.png ... Fig8.png (600 dpi) and Fig4.pdf ... Fig8.pdf (vector)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ---------- common style (Springer: plain, white, sans-serif, readable in greyscale)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9.5, "legend.fontsize": 8, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "hatch.linewidth": 0.6,
})
# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, GREEN, VERM, PURPLE, SKY, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#7F7F7F")
EDGE = "black"

def chance(ax, y=50, label=True):
    ax.axhline(y, ls="--", lw=0.9, color=GREY, zorder=0)
def save(fig, name):
    fig.savefig(name + ".png"); fig.savefig(name + ".pdf"); plt.close(fig)
def label_bars(ax, bars, vals, fmt="{:.2f}", dy=1.0, size=6.8, rot=0):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(v),
                ha="center", va="bottom", fontsize=size, rotation=rot,
                bbox=(dict(fc="white", ec="none", pad=0.2) if rot else None), zorder=5)

# =====================================================================
# FIGURE 4  - per-bit uniformity of the measured third-party dataset [20]
# =====================================================================
bits = np.arange(16)
unif = [91.48, 49.24, 42.60, 72.86, 37.54, 45.50, 49.89, 79.83,
        76.07, 46.29, 58.24, 34.47, 25.03, 50.41, 34.85, 39.51]
def cls(u):
    if 45 <= u <= 55: return "good"
    if 35 <= u <= 65: return "marg"
    return "bias"
style = {"good": (BLUE, ""), "marg": (ORANGE, "///"), "bias": (VERM, "xxx")}
fig, ax = plt.subplots(figsize=(7.0, 3.3))
for b, u in zip(bits, unif):
    c, h = style[cls(u)]
    ax.bar(b, u, 0.7, color=c, hatch=h, edgecolor=EDGE, lw=0.5)
    ax.text(b, u + 1.2, f"{u:.2f}", ha="center", va="bottom", fontsize=6.8, zorder=5,
            bbox=dict(fc="white", ec="none", pad=0.15))
ax.axhline(50, ls="--", lw=0.9, color="black")
for y in (35, 65):
    ax.axhline(y, ls=":", lw=0.9, color=GREY)
ax.text(15.55, 50, "50% (ideal)", va="center", fontsize=7)
ax.text(15.55, 65, "65%", va="center", fontsize=7, color="#444")
ax.text(15.55, 35, "35%", va="center", fontsize=7, color="#444")
ax.set_xlim(-0.6, 15.5); ax.set_ylim(0, 100)
ax.set_xticks(bits); ax.set_xlabel("Response-bit position")
ax.set_ylabel("Uniformity (% of responses = 1)")
ax.legend(handles=[
    Patch(fc=BLUE, ec=EDGE, label="Good, 45–55% (bits 1, 5, 6, 9, 13)"),
    Patch(fc=ORANGE, ec=EDGE, hatch="///", label="Marginal, 35–45% or 55–65% (bits 2, 4, 10, 15)"),
    Patch(fc=VERM, ec=EDGE, hatch="xxx", label="Biased, <35% or >65% (bits 0, 3, 7, 8, 11, 12, 14)")],
    loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=False)
ax.text(0.985, 0.97, "Full dataset: 52.26%, $H_{min}$ = 0.9362\n"
        "Masked (5 good bits): 48.42%, $H_{min}$ = 0.9552",
        transform=ax.transAxes, ha="right", va="top", fontsize=7,
        bbox=dict(fc="white", ec="0.6", lw=0.6))
save(fig, "Fig4")

# =====================================================================
# FIGURE 5  - ML attack accuracy, C0 / C1 / C2 at 100K and 1M (Table IV)
# =====================================================================
models = ["LR", "DNN/Adam", "DNN/Nadam", "XGBoost"]
T4 = {  # order: LR, DNN/Adam, DNN/Nadam, XGBoost
    ("C0", "100K"): [70.26, 99.84, 99.13, 100.00], ("C0", "1M"): [54.92, 64.76, 58.77, 65.37],
    ("C1", "100K"): [50.87, 50.15, 50.13, 49.91], ("C1", "1M"): [49.85, 49.98, 50.04, 50.02],
    ("C2", "100K"): [50.73, 50.72, 50.37, 49.29], ("C2", "1M"): [49.97, 50.11, 50.05, 49.73]}
cfg_style = {"C0": (VERM, "", "C0 – unprotected CRO-PUF"),
             "C1": (BLUE, "///", "C1 – CRO-PUF + NLFSR-Shield"),
             "C2": (ORANGE, "...", "C2 – CRO-PUF + Keccak-f[200]")}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
x = np.arange(4); w = 0.26
for ax, size, title in [(axes[0], "100K", "(a) 100K CRPs – single device"),
                        (axes[1], "1M", "(b) 1M CRPs – 10 devices pooled")]:
    for k, cfg in enumerate(["C0", "C1", "C2"]):
        c, h, _ = cfg_style[cfg]; v = T4[(cfg, size)]
        bars = ax.bar(x + (k - 1) * w, v, w, color=c, hatch=h, edgecolor=EDGE, lw=0.5)
        label_bars(ax, bars, v, dy=1.0, size=5.6, rot=90)
    chance(ax)
    ax.set_xticks(x); ax.set_xticklabels(models); ax.set_title(title)
    ax.set_ylim(0, 118); ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", lw=0.4, color="0.88"); ax.set_axisbelow(True)
axes[0].set_ylabel("Test accuracy (%)")
axes[1].axhline(65.4, ls="-.", lw=0.9, color=VERM)
handles = [Patch(fc=c, ec=EDGE, hatch=h, label=l) for c, h, l in cfg_style.values()]
handles += [Line2D([], [], ls="--", color=GREY, label="Random guessing (50%)"),
            Line2D([], [], ls="-.", color=VERM, label="Pooled C0 bound, 65.4% (best challenge-only predictor)")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.13))
fig.tight_layout(rect=(0, 0.08, 1, 1))
save(fig, "Fig5")

# =====================================================================
# FIGURE 6  - ablation A0-A4, DNN/Adam, mean +/- std over 3 seeds (Table V)
# =====================================================================
var = ["A0\nBare\nCRO-PUF", "A1\n+16-bit\nlinear LFSR", "A2\n+32-bit\nlinear only",
       "A3\n+32-bit + MNL\n(= C1)", "A4\nRegister alone\n(random input)"]
m100 = [100.00, 100.00, 49.97, 49.92, 49.89]; s100 = [0.00, 0.00, 0.23, 0.29, 0.07]
m1M = [100.00, 100.00, 50.12, 50.01, 50.05]; s1M = [0.00, 0.00, 0.05, 0.05, 0.03]
fig, ax = plt.subplots(figsize=(6.6, 3.3))
x = np.arange(5); w = 0.36
b1 = ax.bar(x - w / 2, m100, w, yerr=s100, capsize=2, color=SKY, edgecolor=EDGE, lw=0.5,
            error_kw=dict(lw=0.7), label="100K CRPs")
b2 = ax.bar(x + w / 2, m1M, w, yerr=s1M, capsize=2, color=BLUE, hatch="///", edgecolor=EDGE,
            lw=0.5, error_kw=dict(lw=0.7), label="1M CRPs")
label_bars(ax, b1, m100, dy=1.2, size=6.4)
label_bars(ax, b2, m1M, dy=1.2, size=6.4)
chance(ax)
ax.set_xticks(x); ax.set_xticklabels(var, fontsize=7.6)
ax.set_ylabel("DNN/Adam test accuracy (%)"); ax.set_ylim(0, 112); ax.set_yticks(range(0, 101, 20))
ax.grid(axis="y", lw=0.4, color="0.88"); ax.set_axisbelow(True)
ax.legend(handles=[b1, b2, Line2D([], [], ls="--", color=GREY, label="Random guessing (50%)")],
          loc="upper right", frameon=False)
ax.text(0.5, -0.36, "All variants: single device; mean ± s.d. over 3 seeds",
        transform=ax.transAxes, ha="center", fontsize=7, style="italic")
save(fig, "Fig6")

# =====================================================================
# FIGURE 7  - measured third-party data vs simulation (Tables IV and VI)
# =====================================================================
m3 = ["LR", "DNN/Adam", "XGBoost"]
sim_c0 = [70.26, 99.84, 100.00]          # Table IV, C0 @ 100K
sim_c1 = [50.87, 50.15, 49.91]           # Table IV, C1 @ 100K
rh = {"RH_C0": ([52.17, 50.61, 50.85], [0.20, 0.13, 0.19]),
      "RH_C1": ([49.76, 49.94, 49.98], [0.31, 0.38, 0.42]),
      "RH_MASKED": ([51.76, 50.86, 52.42], [0.12, 0.52, 0.57])}
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw=dict(width_ratios=[1.5, 1]))
x = np.arange(3); w = 0.16
series = [("Simulated C0, 100K", sim_c0, None, VERM, ""),
          ("Simulated C1, 100K", sim_c1, None, BLUE, ""),
          ("Measured RH_C0 (raw)", *rh["RH_C0"], "white", "xxx"),
          ("Measured RH_C1 (+NLFSR-Shield)", *rh["RH_C1"], "white", "///"),
          ("Measured RH_MASKED (5 good bits)", *rh["RH_MASKED"], "white", "...")]
for k, (lab, v, e, c, h) in enumerate(series):
    bars = ax.bar(x + (k - 2) * w, v, w, color=c, hatch=h, edgecolor=EDGE, lw=0.5, label=lab)
    label_bars(ax, bars, v, dy=1.0, size=5.4, rot=90)
chance(ax)
ax.set_xticks(x); ax.set_xticklabels(m3); ax.set_ylim(0, 118); ax.set_yticks(range(0, 101, 20))
ax.set_ylabel("Test accuracy (%)"); ax.set_title("(a) Simulation vs measured data")
ax.grid(axis="y", lw=0.4, color="0.88"); ax.set_axisbelow(True)
# (b) zoom on the measured conditions, mean +/- std
for k, (key, h) in enumerate([("RH_C0", "xxx"), ("RH_C1", "///"), ("RH_MASKED", "...")]):
    v, e = rh[key]
    bx.errorbar(x + (k - 1) * 0.22, v, yerr=e, fmt=["o", "s", "^"][k], ms=4.5,
                mfc="white", mec="black", ecolor="black", capsize=2.5, lw=0.8, label=key)
bx.axhline(50, ls="--", lw=0.9, color=GREY)
bx.set_xticks(x); bx.set_xticklabels(m3); bx.set_ylim(48, 54)
bx.set_title("(b) Measured data only (mean ± s.d.)"); bx.set_ylabel("Test accuracy (%)")
bx.grid(axis="y", lw=0.4, color="0.88"); bx.set_axisbelow(True)
bx.legend(loc="upper left", frameon=False, fontsize=7)
h1, l1 = ax.get_legend_handles_labels()
h1.append(Line2D([], [], ls="--", color=GREY)); l1.append("Random guessing (50%)")
fig.legend(h1, l1, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.12), fontsize=7.2)
fig.tight_layout(rect=(0, 0.09, 1, 1))
save(fig, "Fig7")

# =====================================================================
# FIGURE 8  - resources (Table VII) and area-security trade-off (Tables IV, VII)
# =====================================================================
cfgs = ["C0\nCRO-PUF", "C1\n+NLFSR-Shield", "C2\n+Keccak-f[200]"]
luts = np.array([201, 218, 597]); ro = np.array([176, 176, 176])
regs = [140, 184, 489]; slices = [89, 97, 219]; fmax = [206.1, 207.9, 174.0]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
x = np.arange(3); w = 0.26
ax.bar(x - w, ro, w, color="0.85", edgecolor=EDGE, lw=0.5, label="LUTs in 16 ring oscillators (176)")
ax.bar(x - w, luts - ro, w, bottom=ro, color=BLUE, edgecolor=EDGE, lw=0.5, label="Other Slice LUTs")
rb = ax.bar(x, regs, w, color=ORANGE, hatch="///", edgecolor=EDGE, lw=0.5, label="Slice registers")
sb = ax.bar(x + w, slices, w, color=GREEN, hatch="...", edgecolor=EDGE, lw=0.5, label="Occupied slices")
for i in range(3):
    ax.text(x[i] - w, luts[i] + 8, str(luts[i]), ha="center", fontsize=6.8)
label_bars(ax, rb, regs, fmt="{:d}", dy=8, size=6.8)
label_bars(ax, sb, slices, fmt="{:d}", dy=8, size=6.8)
ax.annotate("+17 LUTs (+8.5%)\n+44 FFs", xy=(1 - w, 222), xytext=(0.62, 390), fontsize=6.8,
            ha="center", arrowprops=dict(arrowstyle="-", lw=0.6))
ax.annotate("+396 LUTs\n+349 FFs", xy=(2 - w, 560), xytext=(1.2, 520), fontsize=6.8,
            ha="center", va="center", arrowprops=dict(arrowstyle="-", lw=0.6))
ax.set_xticks(x); ax.set_xticklabels([f"{c}\n$F_{{max}}$ {f} MHz" for c, f in zip(cfgs, fmax)], fontsize=7.2)
ax.set_ylabel("Count (post-implementation)"); ax.set_ylim(0, 680)
ax.set_title("(a) Resource utilisation, Artix-7 xc7a35t")
ax.grid(axis="y", lw=0.4, color="0.88"); ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, frameon=False, fontsize=6.8)
# (b) area-security: highest 100K accuracy among the four models of Table IV
best = [100.00, 50.87, 50.73]
bx.axhspan(45, 55, color="0.93", zorder=0)
bx.axhline(50, ls="--", lw=0.9, color=GREY)
for xv, yv, mk, c, lab in zip(luts, best, ["o", "s", "^"], [VERM, BLUE, ORANGE], ["C0", "C1", "C2"]):
    bx.scatter(xv, yv, s=55, marker=mk, color=c, edgecolor="black", lw=0.6, zorder=3)
    bx.text(*{"C0": (xv + 18, yv - 1), "C1": (xv + 4, yv + 6.5), "C2": (xv, yv + 6.5)}[lab],
            f"{lab}: {xv} LUTs, {yv:.2f}%", fontsize=6.8, ha={"C0": "left", "C1": "left", "C2": "center"}[lab], va="center")
bx.annotate("", xy=(597, 42), xytext=(218, 42), arrowprops=dict(arrowstyle="<->", lw=0.7))
bx.text(407, 38.3, "same resistance, 2.7× fewer LUTs for C1", ha="center", fontsize=6.6)
bx.set_xlim(150, 700); bx.set_ylim(35, 105)
bx.set_xlabel("Slice LUTs (whole design)")
bx.set_ylabel("Best attack accuracy, 100K CRPs (%)")
bx.set_title("(b) Area–security trade-off")
bx.text(690, 45.4, "45–55% band", ha="right", va="bottom", fontsize=6.4, color="0.35")
fig.tight_layout()
save(fig, "Fig8")
print("Saved Fig4-Fig8 as PNG (600 dpi) and PDF")

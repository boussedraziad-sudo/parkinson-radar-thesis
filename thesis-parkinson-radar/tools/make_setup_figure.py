#!/usr/bin/env python3
"""Rebuild setup_and_signal_chain.png.

The previous version stacked seven boxes in a narrow right-hand column with no
breathing room. This version gives the geometry the top band and lays the signal
chain out horizontally across the full width, five steps instead of seven.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Wedge, FancyArrow
import numpy as np
import pathlib

OUT = pathlib.Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar/outputs/figures")

INK      = "#12314e"
SUBINK   = "#5b6d80"
BOXFACE  = "#eef4fa"
BOXEDGE  = "#b6c9dc"
ACCENT   = "#c4611f"
ACCFACE  = "#fdf2e8"
TORSO_C  = "#2f6fb0"
FOOT_C   = "#c0532a"

# The page prints this at \textwidth (16 cm), i.e. ~0.42x, so every label is
# sized for that reduction: >=19 pt here is >=8 pt on paper.
fig = plt.figure(figsize=(15.0, 10.2), dpi=100)
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(2, 1, height_ratios=[1.06, 0.94], hspace=0.16,
                      left=0.035, right=0.965, top=0.945, bottom=0.045)

# ─────────────────────────── top: acquisition geometry ───────────────────────
ax = fig.add_subplot(gs[0]); ax.set_axis_off()
ax.set_xlim(-1.75, 5.05); ax.set_ylim(-0.58, 2.00)
ax.set_aspect("equal")

ax.text(1.65, 1.99, "Acquisition geometry: two radars watching the same walk",
        ha="center", va="top", fontsize=26, fontweight="bold", color=INK)
ax.text(1.65, 1.76, "four nodes on two tripods: at each end, one aimed at the trunk and one at the feet, 40° beamwidth",
        ha="center", va="top", fontsize=19.5, color=SUBINK, style="italic")

# beams
ax.add_patch(Wedge((-0.89, 1.00), 3.85, -21, 19, facecolor=TORSO_C, alpha=0.075,
                   edgecolor=TORSO_C, lw=0.8, ls=(0, (5, 4))))
ax.add_patch(Wedge((-0.89, 0.15), 3.85, -2, 38, facecolor=FOOT_C, alpha=0.075,
                   edgecolor=FOOT_C, lw=0.8, ls=(0, (5, 4))))

# floor + corridor mat
ax.plot([-1.60, 4.30], [0, 0], color="#3c4b5a", lw=2.6, solid_capstyle="butt",
        zorder=3)
ax.plot([0, 3.0], [-0.045, -0.045], color="#8d9aa8", lw=5, solid_capstyle="butt",
        zorder=2)

# dimensions, both below the floor so nothing collides
for xa, xb, lab, col in [(-0.89, 0.0, "1 m standoff", "#7c8a99"),
                         (0.0, 3.0, "walking corridor  3 m", "#3c4b5a")]:
    ax.annotate("", xy=(xa, -0.26), xytext=(xb, -0.26),
                arrowprops=dict(arrowstyle="<->", color=col, lw=1.4))
    ax.text((xa + xb) / 2, -0.43, lab, ha="center", va="center",
            fontsize=19, color=col)

# radar nodes, with the label stacked directly above each one
ax.add_patch(Rectangle((-1.02, 0.945), 0.26, 0.11, facecolor=TORSO_C,
                       edgecolor="#173d63", lw=1.4, zorder=4))
ax.add_patch(Rectangle((-1.02, 0.095), 0.26, 0.11, facecolor=FOOT_C,
                       edgecolor="#7d3115", lw=1.4, zorder=4))
ax.text(-0.89, 1.34, "TORSO radar\n1.00 m high", ha="center", va="center",
        fontsize=20, fontweight="bold", color=TORSO_C, linespacing=1.4)
ax.text(-0.89, 0.48, "FOOT radars ×2\n0.15 m high", ha="center", va="center",
        fontsize=20, fontweight="bold", color=FOOT_C, linespacing=1.4)

# walker
W, HIP = 1.72, 0.80
ax.add_patch(plt.Circle((W, 1.32), 0.090, color="#22313f", zorder=6))
ax.plot([W, W], [HIP, 1.23], color="#22313f", lw=5, zorder=6, solid_capstyle="round")
ax.plot([W, W - 0.20], [HIP, 0.01], color="#22313f", lw=4.2, zorder=6,
        solid_capstyle="round")
ax.plot([W, W + 0.19], [HIP, 0.01], color="#22313f", lw=4.2, zorder=6,
        solid_capstyle="round")
ax.plot([W, W + 0.19], [1.12, 0.80], color="#22313f", lw=3.2, zorder=6,
        solid_capstyle="round")
ax.annotate("", xy=(2.62, 1.02), xytext=(2.02, 1.02),
            arrowprops=dict(arrowstyle="-|>", color="#1f7a55", lw=3.2,
                            mutation_scale=22))
ax.text(2.32, 1.12, "walk", ha="center", va="bottom", fontsize=20,
        fontweight="bold", color="#1f7a55")

# speed callouts
ax.plot([1.83, 3.28], [1.31, 1.30], color=TORSO_C, lw=1.0, ls=":", zorder=4)
ax.plot([1.90, 3.28], [0.10, 0.34], color=FOOT_C, lw=1.0, ls=":", zorder=4)
ax.text(3.32, 1.30,
        "trunk moves at about 1 m/s\n$\\rightarrow$ shift of about 150 Hz",
        ha="left", va="center", fontsize=19.5, color=TORSO_C, linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.42", fc="white", ec=TORSO_C, lw=1.6),
        zorder=7)
ax.text(3.32, 0.34,
        "feet peak at 3 to 4 m/s\n$\\rightarrow$ shift up to about 500 Hz",
        ha="left", va="center", fontsize=19.5, color=FOOT_C, linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.42", fc="white", ec=FOOT_C, lw=1.6),
        zorder=7)

# ─────────────────────────── bottom: signal chain ────────────────────────────
bx = fig.add_subplot(gs[1]); bx.set_axis_off()
bx.set_xlim(0, 100); bx.set_ylim(0, 100)

bx.text(50, 97, "From radio wave to the two pictures we analyse",
        ha="center", va="top", fontsize=25, fontweight="bold", color=INK)

steps = [
    ("Transmit\nand reflect",
     "a 23 GHz chirp\nsweeps 1.4 GHz,\none every 625 µs"),
    ("Remove\nstill objects",
     "a 10 Hz high-pass\ndrops walls\nand furniture"),
    ("Fourier\ntransform in\nshort slices",
     "a 50 ms window\nslid along\nthe recording"),
    ("Spectrogram",
     "320 speed slots\ncovering ±800 Hz,\none per 5 Hz"),
    ("Brightness\nrescaled, two\nchannels kept",
     "one aimed at\nthe feet, one\nat the trunk"),
]

n = len(steps)
gap = 2.8
w = (100 - gap * (n - 1)) / n
y0, h = 8, 72
for i, (title, sub) in enumerate(steps):
    x = i * (w + gap)
    last = (i == n - 1)
    bx.add_patch(FancyBboxPatch(
        (x, y0), w, h, boxstyle="round,pad=0,rounding_size=2.2",
        facecolor=ACCFACE if last else BOXFACE,
        edgecolor=ACCENT if last else BOXEDGE,
        lw=2.0 if last else 1.5, mutation_aspect=0.42))
    bx.text(x + w / 2, y0 + h - 6, f"{i+1}", ha="center", va="center",
            fontsize=16, fontweight="bold", color="white",
            bbox=dict(boxstyle="circle,pad=0.30",
                      fc=ACCENT if last else "#7d99b5", ec="none"))
    bx.text(x + w / 2, y0 + h - 14, title, ha="center", va="top",
            fontsize=21, fontweight="bold", color=ACCENT if last else INK,
            linespacing=1.3)
    bx.text(x + w / 2, y0 + 4, sub, ha="center", va="bottom",
            fontsize=19, color=SUBINK, style="italic", linespacing=1.35)
    if not last:
        bx.add_patch(FancyArrow(x + w + 0.45, y0 + h / 2, gap - 0.9, 0,
                                width=1.6, head_width=5.2, head_length=1.5,
                                length_includes_head=True,
                                facecolor="#95a8bb", edgecolor="none"))

bx.text(50, 1.5,
        "Steps 1 to 4 were run by the team that built the system. This thesis "
        "starts from the output of step 5.",
        ha="center", va="center", fontsize=19, color=SUBINK, style="italic")

fig.savefig(OUT / "setup_and_signal_chain.png", dpi=100,
            facecolor="white", bbox_inches="tight", pad_inches=0.18)
print("wrote", OUT / "setup_and_signal_chain.png")

# ─── trim the clipped title strip off the foot-vs-torso comparison ───────────
# One-off repair: running it again on the already-trimmed file would crop real
# title pixels, so it only runs when explicitly requested.
if "--trim-guide" in sys.argv:
    from PIL import Image
    p = OUT / "guide_foot_vs_torso.png"
    im = Image.open(p)
    a = np.array(im.convert("L"))
    top = 0
    for y in range(min(30, im.height)):      # find the end of the clipped strip
        if (a[y] < 200).sum() > 0:
            top = y + 1
        elif top and y - top > 3:
            break
    im.crop((0, top, im.width, im.height)).save(p)
    print("trimmed", p, "->", Image.open(p).size, f"(removed {top} rows)")

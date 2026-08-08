#!/usr/bin/env python3
"""Second batch of modelling figures: the two flows, the network, and fixes."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrow, FancyBboxPatch

ROOT = Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import vizstyle as V                                              # noqa: E402
V.apply()
FIG, CACHE = ROOT / "outputs/figures", ROOT / "outputs/preprocessed"
m = pd.read_csv(CACHE / "manifest.csv")
from src.data_loader import iter_trials, load_trial               # noqa: E402
from src.preprocessing import window_indices                      # noqa: E402


def save(fig, name):
    fig.savefig(FIG / name); plt.close(fig); print("  ", name)


# ══════════ 1 · windowing, with the cuts drawn on the spectrogram ══════════
tp = next(t for t in iter_trials() if t.subject_id == m.subject_id.iloc[0])
tr = load_trial(tp.path, representation="ce")
foot = tr["ce_foot"]; t = np.asarray(tr["t_axis"], float).ravel()
dop = np.asarray(tr["doppler"], float).ravel()
spans = window_indices(t, 3.0, 1.5)

fig, (ax, axw) = plt.subplots(2, 1, figsize=(12.2, 6.0), sharex=True,
                              gridspec_kw={"height_ratios": [3.0, 1.15], "hspace": 0.10})
ax.grid(False)
ax.imshow(np.log1p(np.maximum(foot, 0)), aspect="auto", origin="lower",
          extent=[t[0], t[-1], dop[0], dop[-1]], cmap=V.SEQ)
ax.axhline(0, color="white", lw=0.8, alpha=0.5)

# every distinct cut point, drawn onto the picture itself
cuts = sorted({t[s] for s, _ in spans} | {t[e - 1] for _, e in spans})
for c in cuts:
    ax.axvline(c, color="white", lw=1.1, alpha=0.85)
for i, (s, e) in enumerate(spans):
    mid = (t[s] + t[e - 1]) / 2
    ax.text(mid, dop[-1] * 0.88, f"w{i}", ha="center", va="center",
            fontsize=10, fontweight="bold", color=V.INK,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.88))
ax.set_ylabel("Doppler [Hz]")
V.title(ax, "Step 1 · every recording is cut into equal-length windows",
        f"{t[-1]-t[0]:.1f} s of walking becomes {len(spans)} windows of 3.0 s. "
        f"Each starts 1.5 s after the previous one, so neighbours share half their content.")
V.note(ax, t[0] + 0.25, dop[-1] * 0.55, "positive = towards the node", color="white")
V.note(ax, t[0] + 0.25, dop[0] * 0.80, "negative = away", color="white")

axw.grid(True, axis="x")
for i, (s, e) in enumerate(spans):
    axw.barh(i, t[e - 1] - t[s], left=t[s], height=0.58, color=V.CONTROL, alpha=0.28,
             edgecolor=V.CONTROL, linewidth=1.4)
    axw.text(t[s] + 0.08, i, f"window {i}   3.0 s", va="center", ha="left",
             fontsize=9.4, color=V.INK_2)
    if i:
        ps, _ = spans[i - 1]
        axw.annotate("", xy=(t[s], i - 0.42), xytext=(t[ps], i - 0.42),
                     arrowprops=dict(arrowstyle="<->", color=V.ACCENT, lw=1.2))
        if i == 1:
            axw.text((t[s] + t[ps]) / 2, i - 0.70, "hop 1.5 s", ha="center",
                     fontsize=9.2, color=V.ACCENT, fontweight="bold")
axw.set_ylim(len(spans) - 0.35, -0.9)
axw.set_yticks([]); axw.set_xlabel("time [s]"); axw.set_ylabel("windows", labelpad=12)
axw.spines["left"].set_visible(False)
save(fig, "pipe_windowing.png")

# ══════════ 2 · the nested split, without collisions ══════════
fig = plt.figure(figsize=(12.4, 4.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
segs = [("TRAIN", 49, V.CONTROL, "fits the weights", 0),
        ("INNER VALIDATION", 8, V.GOOD, "chooses the epoch to keep", 1),
        ("TEST", 1, V.PD, "scored once, decides nothing", 2)]
BAR_Y, BAR_H, X0, X1 = 52, 15, 2.0, 98.0
LABEL_Y = [40, 28, 16]
x = X0
for name, n, col, sub, k in segs:
    w = n / 58 * (X1 - X0)
    ax.add_patch(FancyBboxPatch((x, BAR_Y), w, BAR_H,
                                boxstyle="round,pad=0,rounding_size=1.0",
                                facecolor=col, alpha=0.16, edgecolor=col, lw=1.8,
                                mutation_aspect=0.22))
    cx = x + w / 2
    ly = LABEL_Y[k]
    ax.plot([cx, cx], [BAR_Y - 1.0, ly + 4.5], color=col, lw=1.1, alpha=0.7)
    ax.text(cx if k == 0 else min(cx + 2, 78), ly,
            f"{name}   ·   {n} subject{'s' if n > 1 else ''}",
            ha="center" if k == 0 else "left", va="center",
            fontsize=11, fontweight="bold", color=col)
    ax.text(cx if k == 0 else min(cx + 2, 78), ly - 5.6, sub,
            ha="center" if k == 0 else "left", va="center",
            fontsize=9.8, color=V.INK_2)
    x += w
ax.text(2, 88, "One fold, repeated 58 times", fontsize=13.5, fontweight="bold", color=V.INK)
ax.text(2, 79, "Widths are to scale. The held-out subject is not seen until training has "
               "finished, so no choice can be tuned on it.",
        fontsize=10.2, color=V.INK_2, style="italic")
ax.text(2, 5, "The fold's standardisation statistics also come from TRAIN alone.",
        fontsize=10, color=V.INK_2)
save(fig, "pipe_split.png")

# ══════════ 3 · the classical baseline, as a flow ══════════
fig = plt.figure(figsize=(13.4, 3.6))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
STEPS = [("Hold out\none subject", "the other 57 train", "58 times", V.PD),
         ("Fill gaps", "median of the\ntraining subjects", "no row discarded", V.INK_2),
         ("Put on one scale", "mean 0, spread 1\nper measurement", "training subjects only", V.GOOD),
         ("Fit the classifier", "classes weighted\nto offset 57/43", "logreg / SVM / forest", V.INK_2),
         ("Average the\nsubject's recordings", "6 probabilities\n-> 1 score", "one answer per person", V.ACCENT)]
gap, x0, x1 = 2.6, 1.5, 98.5
w = (x1 - x0 - gap * (len(STEPS) - 1)) / len(STEPS)
for i, (t_, s_, foot, col) in enumerate(STEPS):
    x = x0 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 26), w, 48, boxstyle="round,pad=0,rounding_size=1.6",
                                facecolor=V.SURFACE, edgecolor=col, lw=1.6,
                                mutation_aspect=0.42))
    ax.text(x + w / 2, 64, t_, ha="center", va="center", fontsize=11,
            fontweight="bold", color=V.INK, linespacing=1.4)
    ax.text(x + w / 2, 45, s_, ha="center", va="center", fontsize=9.4,
            color=V.INK_2, style="italic", linespacing=1.5)
    ax.text(x + w / 2, 31.5, foot, ha="center", va="center", fontsize=8.8, color=col)
    if i < len(STEPS) - 1:
        ax.add_patch(FancyArrow(x + w + 0.4, 50, gap - 1.2, 0, width=0.9,
                                head_width=3.4, head_length=1.0,
                                length_includes_head=True,
                                facecolor=V.INK_3, edgecolor="none"))
ax.text(1.5, 89, "How one classical prediction is made", fontsize=13,
        fontweight="bold", color=V.INK)
ax.text(1.5, 81, "The loop runs once per subject; the 58 resulting scores give one AUC.",
        fontsize=10.2, color=V.INK_2, style="italic")
ax.text(50, 14, "58 subject scores  ->  one AUC for the whole configuration",
        ha="center", fontsize=10.6, color=V.ACCENT, fontweight="bold")
save(fig, "pipe_classical_flow.png")

# ══════════ 4 · SmallCNN, layer by layer ══════════
fig = plt.figure(figsize=(13.4, 4.2))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
LAYERS = [
    ("Input", "2 x 224 x 224", "foot + torso", 46, V.CONTROL),
    ("Conv 3x3, 16\nBN + ReLU + pool", "16 x 112 x 112", "16 edge detectors", 40, V.INK_2),
    ("Conv 3x3, 32\nBN + ReLU + pool", "32 x 56 x 56", "shapes from edges", 32, V.INK_2),
    ("Conv 3x3, 64\nBN + ReLU", "64 x 56 x 56", "patterns from shapes", 26, V.INK_2),
    ("Global average\npool", "64", "one number per pattern", 12, V.GOOD),
    ("Dropout 0.3\nLinear", "2", "control vs PD", 20, V.ACCENT),
]
gap, x0, x1 = 2.2, 1.5, 98.5
w = (x1 - x0 - gap * (len(LAYERS) - 1)) / len(LAYERS)
for i, (name, shape, note, h, col) in enumerate(LAYERS):
    x = x0 + i * (w + gap)
    y = 50 - h / 2
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=col, alpha=0.14, edgecolor=col, lw=1.6,
                                mutation_aspect=0.42))
    ax.text(x + w / 2, 78, name, ha="center", va="center", fontsize=10.2,
            fontweight="bold", color=V.INK, linespacing=1.4)
    ax.text(x + w / 2, 50, shape, ha="center", va="center", fontsize=10.5,
            family="monospace", fontweight="bold", color=col)
    ax.text(x + w / 2, 20, note, ha="center", va="center", fontsize=8.8,
            color=V.INK_2, linespacing=1.4)
    if i < len(LAYERS) - 1:
        ax.add_patch(FancyArrow(x + w + 0.3, 50, gap - 1.0, 0, width=0.8,
                                head_width=3.0, head_length=0.9,
                                length_includes_head=True,
                                facecolor=V.INK_3, edgecolor="none"))
ax.text(1.5, 93, "SmallCNN, layer by layer", fontsize=13, fontweight="bold", color=V.INK)
ax.text(1.5, 86, "Box height is the size of the tensor leaving that layer. The picture "
                 "shrinks while the number of feature maps grows.",
        fontsize=10.2, color=V.INK_2, style="italic")
ax.text(1.5, 5, "23,682 trainable parameters. Global average pooling replaces a flatten "
                "that would have added roughly 200,000 weights on its own.",
        fontsize=9.8, color=V.INK_2)
save(fig, "pipe_smallcnn.png")

print("done")

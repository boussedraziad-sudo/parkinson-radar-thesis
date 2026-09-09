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

# These canvases are 12.2 to 13.4 in wide but print at \textwidth (16 cm), a
# ~0.47-0.52x reduction, so every size is chosen for the printed page:
# ~15 pt here is ~7-8 pt on paper.
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 15.5,
    "xtick.labelsize": 14.5,
    "ytick.labelsize": 14.5,
    "legend.fontsize": 14,
})

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

fig, (ax, axw) = plt.subplots(2, 1, figsize=(12.2, 6.8), sharex=True,
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
            fontsize=15, fontweight="bold", color=V.INK,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.88))
ax.set_ylabel("Doppler [Hz]")
V.title(ax, "Step 1 · every recording is cut into equal-length windows",
        f"{t[-1]-t[0]:.1f} s of walking becomes {len(spans)} windows of 3.0 s. "
        f"Each starts 1.5 s after the previous one,\nso neighbours share half "
        f"their content.", sub_size=14.5, pad=52)
V.note(ax, t[0] + 0.25, dop[-1] * 0.55, "positive = towards the node",
       color="white", fontsize=14.5)
V.note(ax, t[0] + 0.25, dop[0] * 0.80, "negative = away", color="white",
       fontsize=14.5)

axw.grid(True, axis="x")
for i, (s, e) in enumerate(spans):
    axw.barh(i, t[e - 1] - t[s], left=t[s], height=0.58, color=V.CONTROL, alpha=0.28,
             edgecolor=V.CONTROL, linewidth=1.4)
    axw.text(t[s] + 0.08, i, f"window {i}   3.0 s", va="center", ha="left",
             fontsize=13.5, color=V.INK_2)
    if i:
        ps, _ = spans[i - 1]
        axw.annotate("", xy=(t[s], i - 0.5), xytext=(t[ps], i - 0.5),
                     arrowprops=dict(arrowstyle="<->", color=V.ACCENT, lw=1.2))
# the hop label lives in its own gutter above the bars, clear of everything
s0, s1 = t[spans[0][0]], t[spans[1][0]]
axw.annotate("", xy=(s1, -0.85), xytext=(s0, -0.85),
             arrowprops=dict(arrowstyle="<->", color=V.ACCENT, lw=1.5))
axw.text(s1 + 0.14, -0.85, "hop 1.5 s  =  distance between window starts",
         ha="left", va="center", fontsize=13.5, color=V.ACCENT, fontweight="bold")
axw.set_ylim(len(spans) - 0.35, -1.55)
axw.set_yticks([]); axw.set_xlabel("time [s]"); axw.set_ylabel("windows", labelpad=12)
axw.spines["left"].set_visible(False)
save(fig, "pipe_windowing.png")

# ══════════ 2 · the nested split, without collisions ══════════
fig = plt.figure(figsize=(12.4, 5.2))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
segs = [("TRAIN", 49, V.CONTROL, "fits the weights", 0),
        ("INNER VALIDATION", 8, V.GOOD, "chooses the epoch to keep", 1),
        ("TEST", 1, V.PD, "scored once, decides nothing", 2)]
BAR_Y, BAR_H, X0, X1 = 56, 15, 2.0, 98.0
LABEL_Y = [44, 30, 16]
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
    # labels for the two thin segments start far enough left that the other
    # segment's leader line never crosses their text
    tx = cx if k == 0 else min(cx + 2, (58 if k == 1 else 74))
    ax.text(tx, ly,
            f"{name}   ·   {n} subject{'s' if n > 1 else ''}",
            ha="center" if k == 0 else "left", va="center",
            fontsize=15.5, fontweight="bold", color=col)
    ax.text(tx, ly - 6.4, sub,
            ha="center" if k == 0 else "left", va="center",
            fontsize=14, color=V.INK_2)
    x += w
ax.text(2, 89, "One fold, repeated 58 times", fontsize=18, fontweight="bold", color=V.INK)
ax.text(2, 80, "Widths are to scale. The held-out subject is not seen until training has "
               "finished, so no choice can be tuned on it.",
        fontsize=14.5, color=V.INK_2, style="italic")
ax.text(2, 3, "The fold's standardisation statistics also come from TRAIN alone.",
        fontsize=14, color=V.INK_2)
save(fig, "pipe_split.png")

# ══════════ 3 · the classical baseline, as a flow ══════════
fig = plt.figure(figsize=(13.4, 4.4))
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
    ax.text(x + w / 2, 64, t_, ha="center", va="center", fontsize=15.5,
            fontweight="bold", color=V.INK, linespacing=1.35)
    ax.text(x + w / 2, 45, s_, ha="center", va="center", fontsize=14,
            color=V.INK_2, style="italic", linespacing=1.4)
    ax.text(x + w / 2, 31.5, foot, ha="center", va="center", fontsize=14, color=col)
    if i < len(STEPS) - 1:
        ax.add_patch(FancyArrow(x + w + 0.4, 50, gap - 1.2, 0, width=0.9,
                                head_width=3.4, head_length=1.0,
                                length_includes_head=True,
                                facecolor=V.INK_3, edgecolor="none"))
ax.text(1.5, 90, "How one classical prediction is made", fontsize=18,
        fontweight="bold", color=V.INK)
ax.text(1.5, 81, "The loop runs once per subject; the 58 resulting scores give one AUC.",
        fontsize=14.5, color=V.INK_2, style="italic")
ax.text(50, 12, "58 subject scores  ->  one AUC for the whole configuration",
        ha="center", fontsize=15, color=V.ACCENT, fontweight="bold")
save(fig, "pipe_classical_flow.png")

# ══════════ 4 · SmallCNN, layer by layer ══════════
fig = plt.figure(figsize=(13.4, 5.4))
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
    ax.text(x + w / 2, 75, name, ha="center", va="bottom", fontsize=14,
            fontweight="bold", color=V.INK, linespacing=1.35)
    ax.text(x + w / 2, 50, shape, ha="center", va="center", fontsize=14.5,
            family="monospace", fontweight="bold", color=col)
    ax.text(x + w / 2, 19, note, ha="center", va="center", fontsize=13,
            color=V.INK_2, linespacing=1.35)
    if i < len(LAYERS) - 1:
        ax.add_patch(FancyArrow(x + w + 0.3, 50, gap - 1.0, 0, width=0.8,
                                head_width=3.0, head_length=0.9,
                                length_includes_head=True,
                                facecolor=V.INK_3, edgecolor="none"))
ax.text(1.5, 94, "SmallCNN, layer by layer", fontsize=18, fontweight="bold", color=V.INK)
ax.text(1.5, 87.5, "Box height is the size of the tensor leaving that layer. The picture "
                   "shrinks while the number of feature maps grows.",
        fontsize=14, color=V.INK_2, style="italic")
ax.text(1.5, 4, "23,682 trainable parameters. Global average pooling replaces a flatten "
                "that would have added roughly 200,000 weights on its own.",
        fontsize=13.5, color=V.INK_2)
save(fig, "pipe_smallcnn.png")

# ══════════ 5 · ResNet-18, layer by layer, with the two adaptations ══════════
fig = plt.figure(figsize=(13.4, 6.8))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
ax.set_xlim(0, 100); ax.set_ylim(0, 122)
RLAYERS = [
    ("Input", "2 x 224 x 224", "foot + torso", 44, V.CONTROL, "radar data"),
    ("Conv 7x7, 64\nBN + ReLU + pool", "64 x 56 x 56", "edges, gradients", 38, V.INK_3, "frozen"),
    ("Block 1\n(2 residual\nunits)", "64 x 56 x 56", "textures", 34, V.INK_3, "frozen"),
    ("Block 2\n(2 residual\nunits)", "128 x 28 x 28", "simple shapes", 28, V.INK_3, "frozen"),
    ("Block 3\n(2 residual\nunits)", "256 x 14 x 14", "motifs, parts", 22, V.INK_3, "frozen"),
    ("Block 4\n(2 residual\nunits)", "512 x 7 x 7", "rich descriptions", 18, V.INK_3, "frozen"),
    ("Global average\npool", "512", "one number\nper description", 11, V.GOOD, "no weights"),
    ("Linear", "2", "control vs PD", 16, V.ACCENT, "TRAINED"),
]
gap, x0, x1 = 2.0, 1.5, 98.5
w = (x1 - x0 - gap * (len(RLAYERS) - 1)) / len(RLAYERS)
for i, (name, shape, note, h, col, tag) in enumerate(RLAYERS):
    x = x0 + i * (w + gap)
    y = 52 - h / 2
    frozen = tag == "frozen"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=col, alpha=0.10 if frozen else 0.16,
                                edgecolor=col, lw=1.5, mutation_aspect=0.42,
                                linestyle=(0, (4, 2)) if frozen else "solid"))
    ax.text(x + w / 2, 76, name, ha="center", va="bottom", fontsize=13.5,
            fontweight="bold", color=V.INK, linespacing=1.3)
    ax.text(x + w / 2, 52, shape, ha="center", va="center", fontsize=12.5,
            family="monospace", fontweight="bold", color=col)
    ax.text(x + w / 2, 30, note, ha="center", va="center", fontsize=12.5,
            color=V.INK_2, linespacing=1.35)
    ax.text(x + w / 2, 21.5, tag, ha="center", va="center", fontsize=12,
            color=V.ACCENT if tag == "TRAINED" else V.INK_3,
            fontweight="bold" if tag == "TRAINED" else "normal", style="italic")
    if i < len(RLAYERS) - 1:
        ax.add_patch(FancyArrow(x + w + 0.3, 52, gap - 1.0, 0, width=0.8,
                                head_width=3.0, head_length=0.9,
                                length_includes_head=True,
                                facecolor=V.INK_3, edgecolor="none"))
# adaptation 1 callout, in its own band between the subtitle and the boxes
cx1 = x0 + 1 * (w + gap) + w / 2
ax.annotate("Adaptation 1 · the 3 colour filters are averaged into one\n"
            "and the average is copied to both radar channels",
            xy=(cx1 + 6.2, 74), xytext=(cx1 + 6.2, 103), fontsize=13, color=V.ACCENT,
            ha="center", va="bottom", linespacing=1.4, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=V.ACCENT, lw=1.2))
# adaptation 2 brace, under the frozen span
xa = x0 + 1 * (w + gap)
xb = x0 + 5 * (w + gap) + w
ax.plot([xa, xa, xb, xb], [16, 13.5, 13.5, 16], color=V.INK_3, lw=1.2)
ax.text((xa + xb) / 2, 9.5, "Adaptation 2 · everything here is frozen at its ImageNet values: "
                            "11.17 million weights act as a fixed description generator",
        ha="center", fontsize=13, color=V.INK_2)
xl = x0 + 7 * (w + gap)
ax.plot([xl, xl, xl + w, xl + w], [16, 13.5, 13.5, 16], color=V.ACCENT, lw=1.2)
ax.text(xl + w / 2, 11.5, "the only part\nthat learns:\n1,026 weights", ha="center",
        va="top", fontsize=12.5, color=V.ACCENT, fontweight="bold", linespacing=1.3)
ax.text(1.5, 118, "ResNet-18 as used here, layer by layer", fontsize=18,
        fontweight="bold", color=V.INK)
ax.text(1.5, 111, "Dashed boxes never change during training. Box height is the size of "
                  "the tensor leaving that layer.",
        fontsize=14, color=V.INK_2, style="italic")
save(fig, "pipe_resnet.png")

print("done")

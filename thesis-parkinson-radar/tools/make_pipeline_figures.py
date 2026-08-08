#!/usr/bin/env python3
"""Figures for the modelling-setup document (everything before training)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow, FancyBboxPatch, Rectangle

ROOT = Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import vizstyle as V                                          # noqa: E402
V.apply()

FIG, CACHE = ROOT / "outputs/figures", ROOT / "outputs/preprocessed"
m = pd.read_csv(CACHE / "manifest.csv")
from src.data_loader import iter_trials, load_trial           # noqa: E402
from src.preprocessing import add_window_weights, window_indices  # noqa: E402


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("  ", name)


# ══════════════════ 0 · the pipeline itself ══════════════════
fig = plt.figure(figsize=(13.6, 5.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
ax.set_xlim(0, 100); ax.set_ylim(0, 100)

STAGES = [
    ("Recording", "one .mat file", "320 x 12,000", "9 s of walking", V.INK_2),
    ("Window", "3.0 s, hop 1.5 s", "320 x 4,800", "~5 per recording", V.INK_2),
    ("log1p", "compress the range", "320 x 4,800", "no scaling yet", V.INK_2),
    ("Resize", "bilinear", "224 x 224", "still oversampled", V.INK_2),
    ("Stack", "foot + torso", "2 x 224 x 224", "one cached array", V.CONTROL),
]
AFTER = [
    ("Standardise", "training fold only", "2 x 224 x 224", "no leakage", V.GOOD),
    ("Augment", "train split only", "2 x 224 x 224", "flip + noise", V.INK_2),
    ("Network", "SmallCNN / ResNet", "-> 2 logits", "per window", V.INK_2),
    ("Subject score", "mean of windows", "1 number", "one per person", V.ACCENT),
]

def band(items, y, h, label, sublabel):
    n = len(items)
    gap, x0, x1 = 2.4, 12.0, 99.0
    w = (x1 - x0 - gap * (n - 1)) / n
    ax.text(0.6, y + h / 2 + 1.6, label, fontsize=11, fontweight="bold",
            color=V.INK, va="center")
    ax.text(0.6, y + h / 2 - 3.4, sublabel, fontsize=9, color=V.INK_3, va="center")
    for i, (t, s, shape, foot, col) in enumerate(items):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                    facecolor=V.SURFACE, edgecolor=col, lw=1.6,
                                    mutation_aspect=0.35))
        ax.text(x + w / 2, y + h - 6.5, t, ha="center", va="center",
                fontsize=11, fontweight="bold", color=V.INK)
        ax.text(x + w / 2, y + h - 13.5, s, ha="center", va="center",
                fontsize=9.2, color=V.INK_2, style="italic")
        ax.text(x + w / 2, y + 9.5, shape, ha="center", va="center",
                fontsize=10, color=col, family="monospace", fontweight="bold")
        ax.text(x + w / 2, y + 3.8, foot, ha="center", va="center",
                fontsize=8.6, color=V.INK_3)
        if i < n - 1:
            ax.add_patch(FancyArrow(x + w + 0.35, y + h / 2, gap - 1.0, 0, width=0.9,
                                    head_width=3.4, head_length=0.9,
                                    length_includes_head=True,
                                    facecolor=V.INK_3, edgecolor="none"))
    return x0, x1

band(STAGES, 60, 33, "Built once", "66 s, 1,673 windows")
band(AFTER, 12, 33, "Per fold", "58 times, 16 min total")
# Orthogonal routing inside the 12 px gap between the bands, so the connector
# never crosses a stage box.
BUSY = 52.0
ax.plot([93.0, 93.0], [60.0, BUSY], color=V.INK_3, lw=1.3, solid_capstyle="round")
ax.plot([93.0, 16.5], [BUSY, BUSY], color=V.INK_3, lw=1.3, solid_capstyle="round")
ax.add_patch(FancyArrow(16.5, BUSY, 0, -4.6, width=0.35, head_width=1.9,
                        head_length=1.9, length_includes_head=True,
                        facecolor=V.INK_3, edgecolor="none"))
ax.text(55, BUSY + 2.0, "the cache is written once and read by all 58 folds",
        ha="center", fontsize=9.4, color=V.INK_3, style="italic")
save(fig, "pipe_overview.png")

# ══════════════════ 1 · windowing on a real recording ══════════════════
tp = next(t for t in iter_trials() if t.subject_id == m.subject_id.iloc[0])
tr = load_trial(tp.path, representation="ce")
foot = tr["ce_foot"]; t = np.asarray(tr["t_axis"], float).ravel()
dop = np.asarray(tr["doppler"], float).ravel()
spans = window_indices(t, 3.0, 1.5)

fig, (ax, axw) = plt.subplots(2, 1, figsize=(12.2, 5.6), sharex=True,
                              gridspec_kw={"height_ratios": [3.1, 1], "hspace": 0.12})
ax.grid(False)
ax.imshow(np.log1p(np.maximum(foot, 0)), aspect="auto", origin="lower",
          extent=[t[0], t[-1], dop[0], dop[-1]], cmap=V.SEQ)
ax.axhline(0, color="white", lw=0.9, alpha=0.55)
ax.set_ylabel("Doppler [Hz]")
V.title(ax, "Every recording is cut into equal-length windows",
        f"{t[-1]-t[0]:.1f} s of walking becomes {len(spans)} windows of 3.0 s, "
        f"each starting 1.5 s after the last")
V.note(ax, t[0] + 0.35, dop[-1] * 0.72, "positive = towards the node", color="white")
V.note(ax, t[0] + 0.35, dop[0] * 0.80, "negative = away", color="white")

axw.grid(True, axis="x")
for i, (s, e) in enumerate(spans):
    axw.barh(i, t[e - 1] - t[s], left=t[s], height=0.56,
             color=V.CONTROL, alpha=0.30, edgecolor=V.CONTROL, linewidth=1.3)
    axw.text(t[s] + 0.10, i, f"window {i}", va="center", ha="left",
             fontsize=9.2, color=V.INK_2)
axw.set_ylim(len(spans) - 0.4, -0.6)
axw.set_yticks([]); axw.set_xlabel("time [s]")
axw.set_ylabel("windows", labelpad=12)
for sp in ("left",):
    axw.spines[sp].set_visible(False)
save(fig, "pipe_windowing.png")

# ══════════════════ 2 · what the network receives ══════════════════
row = m.iloc[len(m) // 3]
arr = np.load(CACHE / row["npy_path"])
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), constrained_layout=True)
vmin, vmax = float(arr.min()), float(arr.max())
for a, ch, name in zip(axes, arr, ["channel 0 · foot-aimed node",
                                   "channel 1 · torso-aimed node"]):
    a.grid(False)
    im = a.imshow(ch, aspect="auto", origin="lower", cmap=V.SEQ, vmin=vmin, vmax=vmax,
                  extent=[0, 3.0, -800, 800])
    a.set_title(name, loc="left", fontsize=11)
    a.set_xlabel("time within the window [s]")
axes[0].set_ylabel("Doppler [Hz]")
cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
cb.set_label("log1p brightness (not yet standardised)", color=V.INK_2, fontsize=9.6)
cb.outline.set_edgecolor(V.RULE)
fig.suptitle("One cached window: exactly the array the network receives",
             fontsize=12.5, fontweight="bold", color=V.INK, x=0.01, ha="left")
save(fig, "pipe_model_input.png")

# ══════════════════ 3 · window count, before and after ══════════════════
mw = add_window_weights(m)
ws = (mw.groupby(["subject_id", "group"])
        .agg(n=("npy_path", "size"), wsum=("sample_weight", "sum")).reset_index())
ws["before"] = ws.n / ws.n.mean()
ws["after"] = ws.wsum / ws.wsum.mean()
rng = np.random.default_rng(7)

fig, ax = plt.subplots(figsize=(11.0, 4.6), constrained_layout=True)
for k, (col, lab) in enumerate([(0, "before weighting"), (1, "after weighting")]):
    pass
for g, c, lab in [("control", V.CONTROL, "healthy control"), ("pd", V.PD, "Parkinson's")]:
    sub = ws[ws.group == g]
    for xpos, colname in [(0, "before"), (1, "after")]:
        jit = rng.uniform(-0.16, 0.16, len(sub))
        ax.plot(xpos + jit, sub[colname], "o", ms=8, color=c, alpha=0.75,
                label=lab if xpos == 0 and colname == "before" else None,
                **V.ring())
ax.axhline(1.0, color=V.RULE, lw=1.2, zorder=0)
ax.set_xticks([0, 1]); ax.set_xticklabels(["as recorded", "after inverse-count weighting"])
ax.set_xlim(-0.40, 1.40)
ax.set_ylabel("share of training influence\n(1.0 = an equal share)")
V.title(ax, "Longer recordings would otherwise carry more weight",
        "each dot is one subject; window count alone separates the groups at AUC 0.62")
_hi, _lo = ws["before"].max(), ws["before"].min()
ax.annotate(f"{_hi:.1f}x an equal share", (0.0 + 0.18, _hi),
            textcoords="offset points", xytext=(16, 0), fontsize=9.6,
            color=V.INK_2, va="center",
            arrowprops=dict(arrowstyle="-", color=V.RULE, lw=1))
ax.annotate(f"{_lo:.1f}x", (0.0 - 0.10, _lo), textcoords="offset points",
            xytext=(-40, 0), fontsize=9.6, color=V.INK_2, va="center",
            arrowprops=dict(arrowstyle="-", color=V.RULE, lw=1))
V.note(ax, 1.22, 1.0, "every subject\nexactly 1.0", color=V.INK_2, ha="left")
ax.legend(loc="upper right")
save(fig, "pipe_window_count.png")

# ══════════════════ 4 · direction of travel ══════════════════
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1.25, 1]})
a = axes[0]
a.hist(m["pos_frac_foot"].dropna(), bins=38, color=V.CONTROL, alpha=0.72,
       edgecolor=V.SURFACE, linewidth=0.8)
a.set_ylim(0, a.get_ylim()[1] * 1.22)
top = a.get_ylim()[1]
a.axvspan(0.4, 0.6, color=V.PD, alpha=0.10, zorder=0)
V.note(a, 0.5, top * 0.94, "straddles\nthe turn", ha="center", color=V.PD)
V.note(a, 0.14, top * 0.96, "walking away", ha="center")
V.note(a, 0.86, top * 0.96, "walking back", ha="center")
a.set_xlabel("share of the window's energy at positive Doppler")
a.set_ylabel("windows")
V.title(a, "Every window is labelled by direction of travel",
        "the source publication discards turning before measuring gait")

a = axes[1]
th = np.linspace(0.0, 0.45, 120)
kept = np.array([((m.pos_frac_foot - 0.5).abs() > x).mean() * 100 for x in th])
a.plot(th, kept, color=V.CONTROL, lw=2.0, solid_capstyle="round")
for x, col in [(0.1, V.ACCENT), (0.2, V.PD)]:
    k = ((m.pos_frac_foot - 0.5).abs() > x).mean() * 100
    a.plot([x], [k], "o", ms=10, color=col, **V.ring())
    a.annotate(f"threshold {x}\nkeeps {k:.0f}% of windows", (x, k),
               textcoords="offset points", xytext=(16, 4), fontsize=9.4,
               color=V.INK_2, va="center")
a.set_xlabel("how much of the turn to exclude")
a.set_ylabel("% of windows kept")
a.set_ylim(0, 105)
V.title(a, "What excluding the turn costs")
save(fig, "pipe_turn.png")

# ══════════════════ 5 · the early-stopping defect, with real data ══════════
H = json.loads((ROOT / "outputs/metrics/example_fold_history.json").read_text())
real = np.array([h["val_auc"] for h in H["history"]])
ep = np.arange(len(real))
best = int(H["best_epoch"])

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), constrained_layout=True)
a = axes[0]
a.plot(ep, np.full_like(ep, 0.5, dtype=float), "-o", color=V.BAD, lw=2.0, ms=6,
       **V.ring())
a.axvspan(5.5, ep[-1] + 0.5, color=V.INK_3, alpha=0.07, zorder=0)
a.plot([0], [0.5], "o", ms=13, color=V.BAD, **V.ring())
a.annotate("epoch 0 selected\nand never beaten", (0, 0.5), textcoords="offset points",
           xytext=(14, 40), fontsize=9.6, color=V.INK_2,
           arrowprops=dict(arrowstyle="-|>", color=V.INK_3, lw=1.2))
V.note(a, 13, 0.72, "training stops here:\npatience exhausted", ha="center")
a.set_ylabel("validation AUC")
a.set_xlabel("epoch")
V.title(a, "Before", "validation set is one subject, so the AUC is undefined "
                     "and was replaced by a constant")

a = axes[1]
a.plot(ep, real, "-o", color=V.GOOD, lw=2.0, ms=6, **V.ring())
a.plot([best], [real[best]], "o", ms=13, color=V.ACCENT, **V.ring())
a.annotate(f"best epoch {best}\nAUC {real[best]:.3f}", (best, real[best]),
           textcoords="offset points", xytext=(-16, -44), fontsize=9.6,
           color=V.INK_2, ha="center",
           arrowprops=dict(arrowstyle="-|>", color=V.INK_3, lw=1.2))
a.set_xlabel("epoch")
V.title(a, "After", "validation set is 8 stratified subjects, so the AUC is real")
a.axhline(0.5, color=V.RULE, lw=1.2, zorder=0)
V.note(a, 0.4, 0.512, "chance", ha="left")
a.set_ylim(0.44, 1.01)
axes[0].set_ylim(0.44, 1.01)
fig.legend(handles=[
    Line2D([], [], color=V.BAD, marker="o", lw=2, ms=6, label="constant substituted for an undefined value"),
    Line2D([], [], color=V.GOOD, marker="o", lw=2, ms=6, label="measured validation AUC (one real fold)"),
    Line2D([], [], color=V.ACCENT, marker="o", lw=0, ms=9, label="epoch whose weights are kept"),
], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.10))
save(fig, "pipe_bug.png")

# ══════════════════ 6 · the nested split ══════════════════
fig = plt.figure(figsize=(12.4, 3.9))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
segs = [("TRAIN", 49, V.CONTROL, "fits the weights"),
        ("INNER VALIDATION", 8, V.GOOD, "chooses the epoch"),
        ("TEST", 1, V.PD, "scored once, decides nothing")]
x = 2.0
for name, n, col, sub in segs:
    w = n / 58 * 96
    ax.add_patch(FancyBboxPatch((x, 40), w, 30, boxstyle="round,pad=0,rounding_size=1.4",
                                facecolor=col, alpha=0.13, edgecolor=col, lw=1.8,
                                mutation_aspect=0.34))
    ax.text(x + w / 2, 60, name, ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=V.INK)
    ax.text(x + w / 2, 49, f"{n} subject{'s' if n > 1 else ''}", ha="center",
            va="center", fontsize=10.5, color=col, fontweight="bold")
    ax.text(x + w / 2, 33, sub, ha="center", va="top", fontsize=9.6, color=V.INK_2)
    x += w + 1.1
ax.text(2, 88, "One fold, repeated 58 times", fontsize=13, fontweight="bold", color=V.INK)
ax.text(2, 79, "The held-out subject is not seen until training has finished, "
               "so no choice can be tuned on it.",
        fontsize=10, color=V.INK_2, style="italic")
ax.text(2, 12, "The fold's normalisation statistics also come from TRAIN alone, so the "
               "held-out subject never influences the scale its own input is measured on.",
        fontsize=10, color=V.INK_2)
save(fig, "pipe_split.png")

# ══════════════════ 7 · capacity against evidence ══════════════════
from src.models import SmallCNN, resnet18_finetune, count_parameters  # noqa: E402
rows = [("SmallCNN, from scratch", count_parameters(SmallCNN(2))["trainable"], V.CONTROL, "used"),
        ("ResNet-18, final layer", count_parameters(resnet18_finetune(2, freeze_until="layer4"))["trainable"], V.GOOD, "used"),
        ("ResNet-18, + final block", count_parameters(resnet18_finetune(2, freeze_until="layer3"))["trainable"], V.INK_3, "ablation"),
        ("ResNet-18, everything", count_parameters(resnet18_finetune(2, freeze_until=None))["trainable"], V.INK_3, "not used")]
fig, ax = plt.subplots(figsize=(11.0, 4.0), constrained_layout=True)
y = np.arange(len(rows))[::-1]
for yi, (lab, v, col, tag) in zip(y, rows):
    ax.barh(yi, v, height=0.55, color=col, alpha=0.85 if tag == "used" else 0.35,
            edgecolor=V.SURFACE, linewidth=2)
    ax.text(v * 1.35, yi, f"{v:,}", va="center", fontsize=10.5,
            fontweight="bold", color=V.INK)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
ax.set_xscale("log"); ax.set_xlim(3e2, 4e7)
ax.axvline(len(m), color=V.PD, lw=1.6)
ax.annotate(f"{len(m):,} training windows", (len(m), len(rows) - 0.35),
            textcoords="offset points", xytext=(9, 0), fontsize=9.6,
            color=V.PD, va="center", fontweight="bold")
ax.set_xlabel("trainable parameters (log scale)")
ax.grid(True, axis="x"); ax.grid(False, axis="y")
V.title(ax, "Capacity against evidence",
        "anything to the right of the line has more free parameters than we have examples")
ax.set_ylim(-0.75, len(rows) - 0.25)
fig.legend(handles=[
    Line2D([], [], color=V.CONTROL, lw=8, alpha=0.85, label="reported as a headline"),
    Line2D([], [], color=V.INK_3, lw=8, alpha=0.35, label="ablation only"),
], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.11))
save(fig, "pipe_models.png")

print("done")

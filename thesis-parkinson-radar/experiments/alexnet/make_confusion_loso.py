#!/usr/bin/env python3
"""Confusion-matrix figures for the discussion chapter.

alex_confusion_loso.png  single panel, never-seen subjects only (LOSO)
alex_confusion_pair.png  seen -> never-seen pair, panel titles leading with
                         who the subjects are rather than the protocol name,
                         so the contrast itself carries the argument

Both are drawn from the stored result JSONs; the experiment's own two-panel
figure (alex_confusion.png, protocol-titled) stays in ALEXNET.html.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
sys.path.insert(0, str(ROOT / "tools"))
import vizstyle as V  # noqa: E402

OUT = ROOT / "experiments/alexnet/outputs"
LOSO = json.load(open(OUT / "alexnet_loso_summary.json"))["summary"]
RND = json.load(open(OUT / "alexnet_random.json"))


def draw(ax, cm, title, sub):
    frac = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(frac, cmap=V.SEQ, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{frac[i, j]:.0%}", ha="center",
                    va="center", fontsize=12, fontweight="bold",
                    color="white" if frac[i, j] > 0.55 else V.INK)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred control", "pred PD"], fontsize=10.5)
    ax.set_yticklabels(["control", "PD"], fontsize=10.5)
    ax.grid(False)
    ax.set_title(f"{title}\n{sub}", loc="left", fontsize=11)


V.apply()
cm_loso = np.array(LOSO["confusion_matrix"])
cm_rand = np.sum([np.array(r["window"]["confusion"]) for r in RND], axis=0)

fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
draw(ax, cm_loso, "AlexNet on never-seen subjects",
     f"LOSO, one decision per person, threshold = prevalence = "
     f"{LOSO['threshold']:.2f}")
ax.set_ylabel("true class")
fig.savefig(OUT / "figures/alex_confusion_loso.png")
print("wrote", OUT / "figures/alex_confusion_loso.png")

fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), constrained_layout=True)
draw(axes[0], cm_rand, "Subjects seen in training",
     "random split, window level, pooled over 3 seeds")
draw(axes[1], cm_loso, "Never-seen subjects",
     f"LOSO, one decision per person, threshold = {LOSO['threshold']:.2f}")
axes[0].set_ylabel("true class")
fig.suptitle("The same architecture, on people it knows and on strangers",
             x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=V.INK)
fig.savefig(OUT / "figures/alex_confusion_pair.png")
print("wrote", OUT / "figures/alex_confusion_pair.png")

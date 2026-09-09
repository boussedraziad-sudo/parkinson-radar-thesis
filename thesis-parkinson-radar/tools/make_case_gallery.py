#!/usr/bin/env python3
"""res_case_gallery.png: one subject per confusion cell (TP, FN, FP, TN),
head to head for the two image models, foot channel, LOSO probabilities."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
sys.path.insert(0, str(ROOT / "tools"))
import vizstyle as V                                     # noqa: E402
V.apply()

CACHE = ROOT / "outputs/preprocessed_v2"
MAN = pd.read_csv(CACHE / "manifest.csv")
SC = pd.read_csv(ROOT / "outputs/metrics/subject_confusion.csv")

MODELS = [("resnet18_fc_v2", "ResNet-18 probe"), ("smallcnn_v2", "SmallCNN")]
CELLS = [  # (label, true, predicted, pick extreme prob: max or min)
    ("Correct: PD called PD",           1, 1, "max", V.GOOD),
    ("Missed: PD called control",       1, 0, "min", V.BAD),
    ("False alarm: control called PD",  0, 1, "max", V.BAD),
    ("Correct: control called control", 0, 0, "min", V.GOOD),
]


def subject_window(sid):
    """A representative window: the middle window of the subject's first
    recording, foot channel."""
    rows = MAN[MAN.subject_id == sid].sort_values(["test", "trial", "window_idx"])
    first = rows[(rows.test == rows.test.iloc[0]) & (rows.trial == rows.trial.iloc[0])]
    row = first.iloc[len(first) // 2]
    return np.load(CACHE / row.npy_path)[0]


fig, axes = plt.subplots(2, 4, figsize=(12.6, 6.6), constrained_layout=True)
for r, (mkey, mname) in enumerate(MODELS):
    d = SC[SC.model == mkey]
    for c, (label, t, p, pick, col) in enumerate(CELLS):
        a = axes[r][c]
        cell = d[(d.true == t) & (d.predicted == p)]
        if cell.empty:
            a.set_axis_off(); continue
        row = cell.loc[cell.prob.idxmax() if pick == "max" else cell.prob.idxmin()]
        a.imshow(subject_window(row.subject_id), aspect="auto", origin="lower",
                 cmap=V.SEQ)
        a.set_title(f"{label}\n{row.subject_id} · P(PD) = {row.prob:.2f}",
                    fontsize=9.6, color=col, loc="left", linespacing=1.35)
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    axes[r][0].set_ylabel(mname, fontsize=11, fontweight="bold")
fig.suptitle("One subject per confusion cell, per model: a representative window "
             "(foot channel) with the true class and the model's probability",
             x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=V.INK)
fig.savefig(ROOT / "outputs/figures/res_case_gallery.png")
print("   res_case_gallery.png")

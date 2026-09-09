#!/usr/bin/env python3
"""Results figures for the modelling document: the campaign in two charts."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
sys.path.insert(0, str(ROOT / "tools"))
import vizstyle as V                                      # noqa: E402
V.apply()
FIG = ROOT / "outputs/figures"
C = json.load(open(ROOT / "outputs/runs/campaign_summary.json"))

# ══════════════ 1 · the whole campaign on one axis ══════════════
ORDER = [
    ("smallcnn_v2",     "SmallCNN"),
    ("resnet18_fc_v2",  "ResNet-18 probe"),
    ("envlstm",         "Envelope-LSTM"),
    ("abl_foot_v2",     "foot channel only"),
    ("abl_noturn_v2",   "turn excluded (n=54)"),
]
GROUPS = [("Models", 0, 3), ("Exploratory", 3, 5)]

fig, ax = plt.subplots(figsize=(11.6, 5.2), constrained_layout=True)
ys = []
y = 0
for gi, (glabel, a, b) in enumerate(GROUPS):
    for k in range(a, b):
        ys.append(y); y += 1
    y += 0.9                                   # gap between groups
ax.set_ylim(max(ys) + 1.4, -2.4)

for x, lab, col, ty, ha in [(0.5, "chance", V.INK_3, -0.55, "right"),
                            (0.610, "recording duration alone", V.ACCENT, -1.75, "right"),
                            (0.620, "window count alone", V.PD, -0.55, "left")]:
    ax.axvline(x, color=col, lw=1.4, alpha=0.75, zorder=1)
    ax.text(x - 0.006 if ha == "right" else x + 0.006, ty, lab, ha=ha,
            va="center", fontsize=9.0, color=col)

for (key, label), yy in zip(ORDER, ys):
    d = C[key]
    ci = d.get("ci")
    if ci and ci[0] is not None:
        ax.plot([ci[0], ci[1]], [yy, yy], color=V.GRID, lw=3.2, zorder=2,
                solid_capstyle="round")
        for xv in ci:
            ax.plot([xv, xv], [yy - 0.16, yy + 0.16], color=V.INK_3, lw=1.4,
                    zorder=3)
    ax.plot([d["pooled"]], [yy], "o", ms=11, color=V.CONTROL, zorder=4, **V.ring())

ax.set_yticks(ys)
ax.set_yticklabels([lab for _, lab in ORDER], fontsize=10)
for glabel, a, b in GROUPS:
    ax.text(-0.36, (ys[a] + ys[b - 1]) / 2, glabel, rotation=90, ha="center",
            va="center", fontsize=9.5, color=V.INK_3, fontweight="bold",
            transform=ax.get_yaxis_transform(), clip_on=False)
for _, a, b in GROUPS[1:]:
    ax.axhline(ys[a] - 0.95, color=V.GRID, lw=1.0)
ax.set_xlim(0.33, 0.85)
ax.set_xlabel("subject-level AUC  (leave-one-subject-out)")
ax.grid(True, axis="x"); ax.grid(False, axis="y")
ax.set_title("Five configurations, one verdict", loc="left", pad=58)
ax.text(0, 1.03, "the three models of the setup chapter, plus two exploratory input variants;\n"
                 "no configuration clearly exceeds the duration floor once the\n"
                 "confidence intervals are taken into account",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9.8,
        color=V.INK_2, style="italic", linespacing=1.35)
ax.legend(handles=[
    Line2D([], [], marker="o", ls="", ms=10, color=V.CONTROL, label="pooled AUC, all 58 subjects"),
    Line2D([], [], color=V.GRID, lw=3.2, label="95% confidence interval"),
], loc="upper right", fontsize=9.5)
fig.savefig(FIG / "res_campaign.png")
plt.close(fig)
print("   res_campaign.png")

# ══════════════ 2 · what the corrected protocol changed ══════════════
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), constrained_layout=True)

a = axes[0]
pairs = [("SmallCNN", "smallcnn", "smallcnn_v2"),
         ("ResNet-18", "resnet18_fc", "resnet18_fc_v2"),
         ("foot only", "abl_foot", "abl_foot_v2")]
yy = np.arange(len(pairs))[::-1]
for (lab, k1, k2), yi in zip(pairs, yy):
    c1, c2 = C[k1]["corr_nwin"], C[k2]["corr_nwin"]
    a.annotate("", xy=(c2, yi), xytext=(c1, yi),
               arrowprops=dict(arrowstyle="-|>", color=V.INK_3, lw=1.8,
                               mutation_scale=16))
    a.plot([c1], [yi], "o", ms=10, color=V.PD, **V.ring())
    a.plot([c2], [yi], "o", ms=10, color=V.GOOD, **V.ring())
a.axvline(0, color=V.RULE, lw=1.2, zorder=0)
a.set_yticks(yy); a.set_yticklabels([p[0] for p in pairs])
a.set_xlim(-0.05, 0.32)
a.set_xlabel("correlation of subject score with window count")
V.title(a, "The duration residue, before and after",
        "red = original, green = corrected protocol")
a.grid(True, axis="x"); a.grid(False, axis="y")

a = axes[1]
labels = ["batch A", "batch B"]
v1 = [C["abl_foot"]["batch_A"], C["abl_foot"]["batch_B"]]
v2 = [C["abl_foot_v2"]["batch_A"], C["abl_foot_v2"]["batch_B"]]
x = np.arange(2)
a.plot(x, v1, "o-", ms=10, lw=2, color=V.PD, label="original protocol", **V.ring())
a.plot(x, v2, "o-", ms=10, lw=2, color=V.GOOD, label="corrected protocol", **V.ring())
a.axhline(0.5, color=V.RULE, lw=1.2, zorder=0)
V.note(a, 1.32, 0.505, "chance", ha="left")
for xi, u, w in zip((0, 1), v1, v2):
    a.annotate(f"{u:.2f}", (xi, u), textcoords="offset points",
               xytext=(14, -3), fontsize=9.6, color=V.PD)
    a.annotate(f"{w:.2f}", (xi, w), textcoords="offset points",
               xytext=(14, -3), fontsize=9.6, color=V.GOOD)
a.set_xticks(x); a.set_xticklabels(labels); a.set_xlim(-0.35, 1.8)
a.set_ylim(0.28, 0.84)
a.set_ylabel("within-batch AUC")
V.title(a, "The one apparent signal, retested",
        "under the corrected protocol the batches swap")
a.legend(fontsize=9.5, loc="lower left")
fig.savefig(FIG / "res_v1v2.png")
plt.close(fig)
print("   res_v1v2.png")

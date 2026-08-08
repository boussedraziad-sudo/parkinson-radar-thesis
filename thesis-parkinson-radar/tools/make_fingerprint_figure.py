#!/usr/bin/env python3
"""Rebuild 11_within_vs_between.png from the current feature table.

Two problems with the earlier version are fixed here.
  * Its title read "points below diagonal = identifiable", which is the wrong way
    round for the axes it draws: identifiable subjects sit ABOVE the line.
  * It used every feature the extractor returned, so a subject who always walks
    slowly could look identifiable through recording length alone. This version
    restricts the distance to the duration-invariant features, which makes the
    fingerprint claim a claim about gait shape rather than about elapsed time.
The seed is fixed so the figure is reproducible.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys
ROOT = pathlib.Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
sys.path.insert(0, str(ROOT))
from src import config as C

K_OTHERS = 10
SEED = 0

df = pd.read_csv(ROOT / "outputs/metrics/trial_features.csv")
cols = [c for c in C.DURATION_INVARIANT_FEATURES if c in df.columns]
X = df[cols].to_numpy(float)
ok = np.isfinite(X).all(axis=1)
X, meta = X[ok], df.loc[ok, ["subject_id", "group"]].reset_index(drop=True)
Xs = StandardScaler().fit_transform(X)
print(f"{len(Xs)} recordings, {len(cols)} duration-invariant features")

rng = np.random.default_rng(SEED)
rows = []
for s in sorted(meta.subject_id.unique()):
    own = np.flatnonzero(meta.subject_id.values == s)
    other = np.flatnonzero(meta.subject_id.values != s)
    if len(own) < 2:
        continue
    d = np.linalg.norm(Xs[own][:, None] - Xs[own][None, :], axis=-1)
    within = float(d[~np.eye(len(own), dtype=bool)].mean())
    centre = Xs[own].mean(axis=0)
    pick = rng.choice(other, size=min(K_OTHERS, len(other)), replace=False)
    between = float(np.linalg.norm(Xs[pick] - centre, axis=-1).mean())
    rows.append({"subject_id": s,
                 "group": meta.loc[meta.subject_id == s, "group"].iloc[0],
                 "within": within, "between": between})

d = pd.DataFrame(rows)
d["ratio"] = d.within / d.between
n_id = int((d.within < d.between).sum())

# Four subjects worth naming on the plot, chosen by rule rather than by eye.
med = d.ratio.median()
picks = {
    "strongest":  d.loc[d.ratio.idxmin()],                       # own walks nearly identical
    "typical":    d.iloc[int((d.ratio - med).abs().argsort().iloc[0])],
    "weakest":    d.loc[d.ratio.idxmax()],                       # closest to the line
    "variable":   d.loc[d.within.idxmax()],                      # own walks vary the most
}
CAP = {"strongest": "most distinctive", "typical": "typical",
       "weakest": "least distinctive", "variable": "most variable"}

stats = {"n_subjects": len(d), "n_features": len(cols),
         "within_median": round(float(d.within.median()), 3),
         "between_median": round(float(d.between.median()), 3),
         "ratio_median": round(float(d.ratio.median()), 3),
         "n_identifiable": n_id,
         "times_more_alike": round(float(d.between.median() / d.within.median()), 2),
         "examples": {k: {"id": r["subject_id"], "group": r["group"],
                          "within": round(float(r["within"]), 2),
                          "between": round(float(r["between"]), 2),
                          "times": round(float(r["between"] / r["within"]), 1),
                          "caption": CAP[k]}
                      for k, r in picks.items()}}
print(json.dumps(stats, indent=2))

CTRL, PD = "#2e86c1", "#c0392b"
fig, ax = plt.subplots(figsize=(9.4, 5.4), dpi=110)
lim = max(d.within.max(), d.between.max()) * 1.10

ax.fill_between([0, lim], [0, lim], [lim, lim], color="#2e86c1", alpha=0.055, lw=0)
ax.plot([0, lim], [0, lim], ls="--", color="#7c8a99", lw=1.5, zorder=2)
ax.text(lim * 0.63, lim * 0.60, "equal: no personal signature", rotation=39,
        ha="center", va="center", fontsize=10.5, color="#7c8a99",
        rotation_mode="anchor")
for g, c, lab in [("control", CTRL, "healthy control"), ("pd", PD, "Parkinson's")]:
    sub = d[d.group.astype(str).str.lower().str.startswith(g[:2])] if g == "pd" \
        else d[~d.group.astype(str).str.lower().str.startswith("pd")]
    ax.scatter(sub.within, sub.between, s=62, color=c, alpha=0.72,
               edgecolor="white", lw=0.8, zorder=4, label=f"{lab} (n={len(sub)})")

# name the four illustrative subjects, everything else stays anonymous
PLACE = {"strongest": (0.55, 8.55), "variable": (3.55, 9.75),
         "typical": (5.15, 7.05), "weakest": (4.85, 2.35)}
for k, (lx, ly) in PLACE.items():
    r = picks[k]
    col = PD if str(r["group"]).lower().startswith("pd") else CTRL
    ax.scatter([r["within"]], [r["between"]], s=180, facecolor="none",
               edgecolor=col, lw=2.0, zorder=5)
    ax.annotate(f'{r["subject_id"]}\n{CAP[k]}', xy=(r["within"], r["between"]),
                xytext=(lx, ly),
                ha="center", va="center", fontsize=10.5, color=col,
                linespacing=1.45, zorder=6,
                bbox=dict(boxstyle="round,pad=0.36", fc="white", ec=col, lw=1.1,
                          alpha=0.96),
                arrowprops=dict(arrowstyle="-", color=col, lw=1.1,
                                shrinkA=6, shrinkB=10, alpha=0.75))

ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("distance among a person's OWN recordings\n(smaller = their walks look alike)",
              fontsize=12, linespacing=1.5)
ax.set_ylabel("distance to OTHER people's recordings\n(larger = they look unlike everyone else)",
              fontsize=12, linespacing=1.5)
ax.set_title(f"Every one of the {len(d)} subjects sits above the line",
             fontsize=15, fontweight="bold", color="#12314e", pad=12)
ax.legend(loc="lower right", frameon=True, fontsize=11,
          bbox_to_anchor=(0.995, 0.015))
ax.grid(alpha=0.18, lw=0.7)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

ax.text(0.985, 0.215,
        f"typical own-recording distance  {stats['within_median']}\n"
        f"typical other-people distance  {stats['between_median']}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5,
        color="#5b6d80", linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#d5dde5", lw=1))

fig.tight_layout()
out = ROOT / "outputs/figures/11_within_vs_between.png"
fig.savefig(out, dpi=110, facecolor="white")
print("wrote", out)
(ROOT / "outputs/metrics/subject_fingerprint.json").write_text(json.dumps(stats, indent=2))

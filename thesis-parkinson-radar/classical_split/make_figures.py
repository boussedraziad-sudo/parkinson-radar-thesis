#!/usr/bin/env python3
"""Figures for the random-split detour (classical_split/)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import vizstyle as V                                   # noqa: E402
V.apply()

OUT = ROOT / "classical_split/outputs"
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
RES = json.loads((OUT / "results.json").read_text())
PRED = pd.read_csv(OUT / "predictions.csv")
CAMP = json.load(open(ROOT / "outputs/runs/campaign_summary.json"))
CACHE = ROOT / "outputs/preprocessed_v2"


def save(fig, name):
    fig.savefig(FIG / name); plt.close(fig); print("  ", name)


def rows(exp):
    return [r for r in RES if r["experiment"] == exp]


def collect(exp, path):
    """path e.g. ('test','window','auc') -> list over seeds."""
    out = []
    for r in rows(exp):
        v = r
        for k in path: v = v[k]
        out.append(v)
    return np.array(out, float)


# ═══════════ 1 · the inflation ladder ═══════════
LOSO = CAMP["resnet18_fc_v2"]
items = [
    ("Window-level random split\n(frame-level, leakiest)",
     collect("window_split", ("test", "window", "auc")), V.PD),
    ("Recording-level random split\n(the literature's hold-out)",
     collect("recording_split", ("test", "window", "auc")), V.PD),
    ("Same training, scored on\n8 never-seen subjects",
     collect("collapse_test", ("never_seen", "window", "auc")), V.ACCENT),
    ("Leave-one-subject-out\n(the thesis protocol)",
     np.array([LOSO["pooled"]]), V.CONTROL),
    ("LOSO, within one batch\n(gait only)",
     np.array([LOSO["batch_A"], LOSO["batch_B"]]), V.CONTROL),
]
fig, ax = plt.subplots(figsize=(11.6, 5.6), constrained_layout=True)
ys = np.arange(len(items))[::-1]
for (lab, vals, col), y in zip(items, ys):
    ax.plot([vals.min(), vals.max()], [y, y], color=V.GRID, lw=3, zorder=1)
    ax.plot(vals, np.full_like(vals, y), "o", ms=9, color=col, zorder=3,
            **V.ring())
    m = vals.mean()
    ax.annotate(f"{m:.2f}", (m, y), textcoords="offset points", xytext=(0, 13),
                ha="center", fontsize=10, fontweight="bold", color=col)
ax.axvline(0.5, color=V.INK_3, lw=1.4, alpha=0.8)
ax.set_ylim(-0.7, len(items) - 0.3)
ax.text(0.503, -0.5, "chance", ha="left", fontsize=9.2, color=V.INK_3)
ax.set_yticks(ys)
ax.set_yticklabels([i[0] for i in items], fontsize=10)
ax.set_xlim(0.35, 1.0)
ax.set_xlabel("AUC (window-level for random splits; subject-level for LOSO rows)")
ax.grid(True, axis="x"); ax.grid(False, axis="y")
ax.set_title("Same model, same data; only the split changes", loc="left", pad=26)
ax.text(0, 1.03, "top three rows: one dot per random seed; bottom two rows: the "
                 "LOSO campaign's pooled score, and its two within-batch scores",
        transform=ax.transAxes, fontsize=9.6, color=V.INK_2, style="italic")
save(fig, "cs_ladder.png")

# ═══════════ 2 · confusion matrices ═══════════
def pooled_confusion(exp, tag="test"):
    c = np.zeros((2, 2), int)
    for r in rows(exp):
        c += np.array(r[tag]["window"]["confusion"], int)
    return c

panels = [
    ("Window-level split", pooled_confusion("window_split")),
    ("Recording-level split", pooled_confusion("recording_split")),
    ("Never-seen subjects", pooled_confusion("collapse_test", "never_seen")),
]
fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.0), constrained_layout=True)
for a, (name, c) in zip(axes, panels):
    frac = c / c.sum(axis=1, keepdims=True)
    a.imshow(frac, cmap=V.SEQ, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{c[i,j]:,}\n{frac[i,j]:.0%}", ha="center",
                   va="center", fontsize=10.5, fontweight="bold",
                   color="white" if frac[i, j] > 0.55 else V.INK)
    a.set_xticks([0, 1]); a.set_yticks([0, 1])
    a.set_xticklabels(["pred control", "pred PD"], fontsize=9.5)
    a.set_yticklabels(["control", "PD"], fontsize=9.5)
    a.set_title(name, loc="left", fontsize=11)
    a.grid(False)
axes[0].set_ylabel("true class")
fig.suptitle("Window-level confusion, pooled over seeds (row-normalised %)",
             x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=V.INK)
save(fig, "cs_confusion.png")

# ═══════════ 3 · the collapse ═══════════
fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
seen_s = collect("collapse_test", ("test", "subject", "auc"))
seen_w = collect("collapse_test", ("test", "window", "auc"))
un_s = collect("collapse_test", ("never_seen", "subject", "auc"))
un_w = collect("collapse_test", ("never_seen", "window", "auc"))
# one mid-grey tone per repeat, dark enough for laser print, plus a direct
# label at the never-seen end so all three lines stay traceable even where
# two seen-side dots nearly coincide
REP_COLS = ["#403f3b", "#6e6d69", "#8a8985"]
for i, (sw, ss, uw, us) in enumerate(zip(seen_w, seen_s, un_w, un_s)):
    ax.plot([0, 1], [sw, uw], "-", color=REP_COLS[i], lw=2.2, zorder=2)
    ax.plot([0], [sw], "o", ms=10, color=V.PD, zorder=3, **V.ring())
    ax.plot([1], [uw], "o", ms=10, color=V.CONTROL, zorder=3, **V.ring())
    ax.annotate(f"repeat {i + 1}", (1, uw), textcoords="offset points",
                xytext=(16, 0), fontsize=10, fontweight="bold",
                color=REP_COLS[i], va="center")
ax.axhline(0.5, color=V.INK_3, lw=1.3, alpha=0.8)
V.note(ax, 1.38, 0.51, "chance", ha="left")
ax.set_xticks([0, 1])
ax.set_xticklabels(["test subjects the model\nSAW during training",
                    "the 8 subjects it\nNEVER saw"], fontsize=10)
ax.set_xlim(-0.35, 1.65)
ax.set_ylabel("window-level AUC")
V.title(ax, "The collapse test",
        "one line per repeat: the same trained model, scored on both groups")
save(fig, "cs_collapse.png")

# ═══════════ 4 · gallery of hits and misses ═══════════
p = PRED[(PRED.experiment == "recording_split") & (PRED["set"] == "test")].copy()
p = p.groupby("npy_path").agg(prob=("prob", "mean"), label=("label", "first"),
                              subject_id=("subject_id", "first")).reset_index()
p["conf_correct"] = np.where(p.label == 1, p.prob, 1 - p.prob)
hits = pd.concat([
    p[p.label == 0].nlargest(2, "conf_correct"),
    p[p.label == 1].nlargest(2, "conf_correct"),
])
misses = pd.concat([
    p[p.label == 0].nsmallest(2, "conf_correct"),
    p[p.label == 1].nsmallest(2, "conf_correct"),
])
fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.4), constrained_layout=True)
for axrow, chunk, kind in ((axes[0], hits, "correct"),
                           (axes[1], misses, "missed")):
    for a, (_, r) in zip(axrow, chunk.iterrows()):
        arr = np.load(CACHE / r.npy_path)[0]
        a.imshow(arr, aspect="auto", origin="lower", cmap=V.SEQ)
        truth = "PD" if r.label == 1 else "control"
        a.set_title(f"{truth} · P(PD)={r.prob:.2f}", fontsize=10,
                    color=V.GOOD if kind == "correct" else V.BAD, loc="left")
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
axes[0][0].set_ylabel("most confidently\nCORRECT", fontsize=10.5)
axes[1][0].set_ylabel("most confidently\nWRONG", fontsize=10.5)
fig.suptitle("Foot-channel windows the recording-split model got right and wrong "
             "(true class · predicted probability)",
             x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=V.INK)
save(fig, "cs_gallery.png")

# ═══════════ 5 · training curves ═══════════
fig, ax = plt.subplots(figsize=(9.6, 4.4), constrained_layout=True)
styles = {"window_split": (V.PD, "window-level split"),
          "recording_split": (V.ACCENT, "recording-level split"),
          "collapse_test": (V.CONTROL, "collapse test (seen pool)")}
seenlab = set()
for r in RES:
    col, lab = styles[r["experiment"]]
    h = pd.DataFrame(r["history"])
    ax.plot(h.epoch, h.val_auc, "-", color=col, lw=1.6, alpha=0.75,
            label=lab if lab not in seenlab else None)
    seenlab.add(lab)
ax.axhline(0.5, color=V.INK_3, lw=1.2, alpha=0.7)
V.note(ax, 0.2, 0.513, "chance", ha="left")
ax.set_xlabel("epoch"); ax.set_ylabel("validation AUC (window-level)")
V.title(ax, "Validation curves under random splits",
        "with familiar subjects in validation the score climbs immediately; "
        "a curve ends where early stopping triggered")
ax.legend(fontsize=9.5, loc="center right")
save(fig, "cs_curves.png")

# ═══════════ 6 · what a human could look for ═══════════
from src.data_loader import iter_trials, load_trial   # noqa: E402

MANI = pd.read_csv(CACHE / "manifest.csv")
dur = (MANI.groupby(["subject_id", "test", "trial"])
       .agg(label=("label", "first"), t0=("t_start_s", "min"),
            t1=("t_end_s", "max")).reset_index())
dur["d"] = dur.t1 - dur.t0
pick = {}
for lab in (0, 1):
    g = dur[(dur.label == lab) & (dur.test == "test1")]   # test1 = chair variant
    pick[lab] = g.iloc[(g.d - g.d.median()).abs().argsort().iloc[0]]

fig, axes = plt.subplots(2, 1, figsize=(12.2, 7.6), constrained_layout=True)
durations = {}
panels = []
for lab, name in ((0, "Control subject"), (1, "PD subject")):
    row = pick[lab]
    tp = next(t for t in iter_trials()
              if t.subject_id == row.subject_id and t.test == row.test
              and t.trial == row.trial)
    tr = load_trial(tp.path, representation="ce")
    t = np.asarray(tr["t_axis"], float).ravel()
    durations[lab] = t[-1] - t[0]
    panels.append((lab, name, tr, t))
xmax = max(durations.values())
for a, (lab, name, tr, t) in zip(axes, panels):
    dop = np.asarray(tr["doppler"], float).ravel()
    a.imshow(np.log1p(np.maximum(tr["ce_foot"], 0)), aspect="auto",
             origin="lower", extent=[0, t[-1] - t[0], dop[0], dop[-1]],
             cmap=V.SEQ)
    a.axhline(0, color="white", lw=0.8, alpha=0.5)
    a.set_xlim(0, xmax)
    a.set_ylabel("Doppler [Hz]")
    a.set_title(f"{name} · foot channel · one whole recording · "
                f"{durations[lab]:.1f} s, chair variant", loc="left",
                fontsize=11.5)
    a.grid(False)
axes[1].set_xlabel("time since the recording started [s]")
fig.suptitle("What to look for: one median-length recording per group, "
             "same variant, same time axis",
             x=0.01, ha="left", fontsize=13, fontweight="bold", color=V.INK)
save(fig, "cs_examples.png")

print("done")

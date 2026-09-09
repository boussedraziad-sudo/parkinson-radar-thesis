#!/usr/bin/env python3
"""Subject-level confusion matrices for the four headline models.

Per-subject LOSO probabilities are thresholded at the prevalence of the
positive class (25/58), as pre-registered. Outputs:
  outputs/figures/res_confusion.png        4 confusion panels
  outputs/figures/res_subject_scatter.png  per-subject probs, best deep model
  outputs/metrics/subject_confusion.csv    per model per subject
  reports/confusion_summary.json           derived metrics + hit/miss lists
"""
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
sys.path.insert(0, str(ROOT / "tools"))
import vizstyle as V                                      # noqa: E402
V.apply()
FIG = ROOT / "outputs/figures"
RUNS = ROOT / "outputs/runs"
METRICS = ROOT / "outputs/metrics"

# ══════════════ load the four per-subject score tables ══════════════
def deep(name):
    df = pd.read_csv(RUNS / f"{name}_folds.csv")
    return df.rename(columns={"subject_label": "true", "subject_prob": "prob"}
                     )[["subject_id", "true", "prob"]]

def classical():
    df = pd.read_csv(METRICS / "diag_per_subject.csv")
    return df.rename(columns={"label": "true", "subject_score": "prob"}
                     )[["subject_id", "true", "prob"]]

MODELS = [
    ("logreg_shape_clean", "Classical logistic regression", classical()),
    ("smallcnn_v2",        "SmallCNN",                       deep("smallcnn_v2")),
    ("resnet18_fc_v2",     "ResNet-18 probe",                deep("resnet18_fc_v2")),
    ("envlstm",            "Envelope-LSTM",                  deep("envlstm")),
]

n_pos = int(MODELS[0][2]["true"].sum())
n_all = len(MODELS[0][2])
THRESH = n_pos / n_all                                   # 25/58, pre-registered
print(f"threshold = prevalence = {n_pos}/{n_all} = {THRESH:.4f}")

# ══════════════ confusion + derived metrics per model ══════════════
summary, rows = {}, []
for key, label, df in MODELS:
    assert len(df) == n_all and int(df["true"].sum()) == n_pos, key
    df = df.copy()
    df["predicted"] = (df["prob"] >= THRESH).astype(int)
    df["correct"] = df["predicted"] == df["true"]
    tn = int(((df.true == 0) & (df.predicted == 0)).sum())
    fp = int(((df.true == 0) & (df.predicted == 1)).sum())
    fn = int(((df.true == 1) & (df.predicted == 0)).sum())
    tp = int(((df.true == 1) & (df.predicted == 1)).sum())
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    summary[key] = {
        "label": label,
        "threshold": THRESH,
        "confusion": [[tn, fp], [fn, tp]],
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_acc": (recall + specificity) / 2,
        "correct": df[df.correct][["subject_id", "true", "prob"]]
                     .to_dict("records"),
        "misclassified": df[~df.correct][["subject_id", "true", "prob"]]
                           .to_dict("records"),
    }
    for _, r in df.iterrows():
        rows.append({"model": key, "subject_id": r.subject_id,
                     "prob": r.prob, "true": int(r.true),
                     "predicted": int(r.predicted), "correct": bool(r.correct)})
    print(f"{label:34s} [[{tn},{fp}],[{fn},{tp}]]  "
          f"rec {recall:.3f}  spec {specificity:.3f}  prec {precision:.3f}  "
          f"f1 {f1:.3f}  bacc {(recall + specificity) / 2:.3f}")
    miss = ", ".join(f"{m['subject_id']}({m['prob']:.2f})"
                     for m in summary[key]["misclassified"])
    print(f"   missed: {miss}")

pd.DataFrame(rows).to_csv(METRICS / "subject_confusion.csv", index=False)
(ROOT / "reports/confusion_summary.json").write_text(
    json.dumps({"threshold": THRESH, "n_pd": n_pos, "n_total": n_all,
                "models": summary}, indent=2))
print("   subject_confusion.csv / confusion_summary.json")

# ══════════════ 1 · four confusion panels ══════════════
fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.9), constrained_layout=True)
for a, (key, label, _) in zip(axes, MODELS):
    c = np.array(summary[key]["confusion"], int)
    frac = c / c.sum(axis=1, keepdims=True)
    a.imshow(frac, cmap=V.SEQ, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{c[i,j]}\n{frac[i,j]:.0%}", ha="center",
                   va="center", fontsize=10.5, fontweight="bold",
                   color="white" if frac[i, j] > 0.55 else V.INK)
    a.set_xticks([0, 1]); a.set_yticks([0, 1])
    a.set_xticklabels(["pred control", "pred PD"], fontsize=9.5)
    a.set_yticklabels(["control", "PD"], fontsize=9.5)
    a.set_title(label, loc="left", fontsize=10.5)
    a.grid(False)
axes[0].set_ylabel("true class")
fig.suptitle("Subject-level confusion at the pre-registered threshold "
             f"(P(PD) >= {THRESH:.2f}, counts and row-normalised %)",
             x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=V.INK)
fig.savefig(FIG / "res_confusion.png")
plt.close(fig)
print("   res_confusion.png")

# ══════════════ 2 · per-subject scatter, best deep model ══════════════
BEST = "resnet18_fc_v2"
df = dict((k, d) for k, _, d in MODELS)[BEST].copy()
df["predicted"] = (df["prob"] >= THRESH).astype(int)
df["correct"] = df["predicted"] == df["true"]
df = df.sort_values("prob").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(11.6, 5.2), constrained_layout=True)
ax.axhline(THRESH, color=V.ACCENT, lw=1.5, alpha=0.9, zorder=1)
V.note(ax, len(df) - 0.5, THRESH + 0.012,
       f"threshold = prevalence = {n_pos}/{n_all} = {THRESH:.2f}",
       color=V.ACCENT, ha="right", va="bottom")
for _, r in df.iterrows():
    col = V.PD if r.true == 1 else V.CONTROL
    if r.correct:
        ax.plot(r.name, r.prob, "o", ms=8, color=col, zorder=3, **V.ring())
    else:
        ax.plot(r.name, r.prob, "X", ms=9.5, color=col, zorder=4,
                markeredgecolor=V.INK, markeredgewidth=0.9)
n_fp = int(((df.true == 0) & ~df.correct).sum())
n_fn = int(((df.true == 1) & ~df.correct).sum())
ax.set_xlabel("subjects, sorted by predicted P(PD)")
ax.set_ylabel("mean LOSO P(PD)")
ax.set_xlim(-1.5, len(df) + 0.5)
ax.grid(True, axis="y"); ax.grid(False, axis="x")
ax.set_xticks([])
V.title(ax, "Every subject's score from the best deep model",
        "ResNet-18 probe; an X marks a subject the "
        f"threshold misclassifies ({n_fp} controls above it, {n_fn} PD below it)")
ax.legend(handles=[
    Line2D([], [], marker="o", ls="", ms=8, color=V.CONTROL, label="control, correct"),
    Line2D([], [], marker="o", ls="", ms=8, color=V.PD, label="PD, correct"),
    Line2D([], [], marker="X", ls="", ms=9, color=V.INK_3,
           markeredgecolor=V.INK, label="misclassified"),
], loc="upper left", fontsize=9.5)
fig.savefig(FIG / "res_subject_scatter.png")
plt.close(fig)
print("   res_subject_scatter.png")

"""
Diagnostics that answer the committee's "how real / where does it fail?"
questions — all GPU-free, off the cached trial feature table.

  1. Per-subject LOSO breakdown. Which subjects are reliably right/wrong?
     Do errors cluster (hinting at an age/severity sub-structure)?
  2. Caliper-robustness sweep. Is the duration-matched result a fluke of the
     0.75 s caliper, or stable across caliper choices?
  3. Subject-level learning curve. Is AUC still rising at n≈50 subjects (more
     data would help) or saturated (the ceiling is the feature representation)?

Runnable headless:  ``.venv/bin/python -m src.diagnostics``
Figures → outputs/figures/diag_*.png ; summary → reports/diagnostics_summary.json
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import baseline as B
from . import config as C
from . import confound_analysis as CA

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})


# ---------------------------------------------------------------------------
# 1. Per-subject LOSO breakdown
# ---------------------------------------------------------------------------


def per_subject_breakdown(df: pd.DataFrame, feature_set: str = "shape_clean") -> dict:
    cols = B.feature_sets(df)[feature_set]
    subj = B.loso_subject_scores(df, cols, B._models()["logreg"])
    subj["pred"] = (subj["subject_score"] >= 0.5).astype(int)
    subj["correct"] = subj["pred"] == subj["label"]
    subj["group"] = subj["label"].map(C.LABEL_TO_GROUP)
    subj = subj.sort_values("subject_score").reset_index(drop=True)
    subj.to_csv(C.METRICS / "diag_per_subject.csv", index=False)

    # dot plot: score per subject, coloured by true group, marker = correct
    fig, ax = plt.subplots(figsize=(7, 11))
    y = np.arange(len(subj))
    for _, row in subj.iterrows():
        i = row.name
        ax.plot(row["subject_score"], i, "o" if row["correct"] else "X",
                color=C.GROUP_COLOURS[row["group"]], ms=8,
                markeredgecolor="black" if not row["correct"] else "none", mew=1.2)
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(subj["subject_id"], fontsize=6)
    ax.set_xlabel("mean LOSO P(PD)   |   X = misclassified, ○ = correct")
    ax.set_title(f"Per-subject prediction ({feature_set})\n"
                 f"blue=control  red=PD")
    fig.savefig(C.FIGURES / "diag_per_subject.png")
    plt.close(fig)

    wrong = subj[~subj["correct"]]
    return {
        "n_correct": int(subj["correct"].sum()),
        "n_total": int(len(subj)),
        "accuracy_at_0.5": float(subj["correct"].mean()),
        "misclassified": wrong[["subject_id", "group", "subject_score"]].to_dict("records"),
        "controls_wrong": int(((wrong.group == "control")).sum()),
        "pd_wrong": int(((wrong.group == "pd")).sum()),
    }


# ---------------------------------------------------------------------------
# 2. Caliper-robustness sweep
# ---------------------------------------------------------------------------


def caliper_sweep(df: pd.DataFrame,
                  calipers=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
                  feature_set: str = "shape_clean") -> dict:
    cols = B.feature_sets(df)[feature_set]
    rows = []
    for cal in calipers:
        m = CA.duration_matched_subset(df, caliper=cal)
        subj = B.loso_subject_scores(m, cols, B._models()["logreg"])
        try:
            auc = float(roc_auc_score(subj["label"], subj["subject_score"]))
        except ValueError:
            auc = float("nan")
        rows.append({"caliper_s": cal, "n_trials": int(len(m)),
                     "n_subjects": int(m.subject_id.nunique()), "auc": auc})
    res = pd.DataFrame(rows)
    res.to_csv(C.METRICS / "diag_caliper_sweep.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(res["caliper_s"], res["auc"], "o-", color="#2e86c1", lw=2, label="AUC")
    ax1.axhline(0.5, color="k", ls="--", lw=0.8)
    ax1.axhline(0.61, color="grey", ls=":", lw=0.8, label="duration-only floor")
    ax1.set_xlabel("matching caliper [s]"); ax1.set_ylabel("matched-subset AUC", color="#2e86c1")
    ax1.set_ylim(0.4, 0.8)
    ax2 = ax1.twinx()
    ax2.bar(res["caliper_s"], res["n_trials"], width=0.1, alpha=0.25, color="grey")
    ax2.set_ylabel("# trials retained", color="grey")
    ax1.set_title(f"Duration-matched AUC vs caliper ({feature_set})")
    ax1.legend(loc="lower right")
    fig.savefig(C.FIGURES / "diag_caliper_sweep.png")
    plt.close(fig)
    return {"sweep": rows,
            "auc_range": [float(res.auc.min()), float(res.auc.max())]}


# ---------------------------------------------------------------------------
# 3. Subject-level learning curve
# ---------------------------------------------------------------------------


def _stratified_subject_sample(subjects, labels, n, rng):
    out = []
    for lab in (0, 1):
        pool = subjects[labels == lab]
        k = max(1, int(round(n * (pool.size / subjects.size))))
        out.extend(rng.choice(pool, size=min(k, pool.size), replace=False))
    return list(out)


def learning_curve(df: pd.DataFrame, feature_set: str = "shape_clean",
                   sizes=(10, 20, 30, 40, 50), repeats: int = 30) -> dict:
    cols = B.feature_sets(df)[feature_set]
    subj_df = df.drop_duplicates("subject_id")[["subject_id", "label"]]
    subjects = subj_df.subject_id.to_numpy()
    labels = subj_df.label.to_numpy()
    rng = np.random.default_rng(C.RANDOM_STATE)

    rows = []
    for n in sizes:
        aucs = []
        for _ in range(repeats):
            train = _stratified_subject_sample(subjects, labels, n, rng)
            test = [s for s in subjects if s not in train]
            tr = df[df.subject_id.isin(train)]
            te = df[df.subject_id.isin(test)]
            if tr.label.nunique() < 2 or te.label.nunique() < 2:
                continue
            mdl = B._models()["logreg"].fit(tr[cols].to_numpy(float), tr["label"].to_numpy())
            te2 = te[["subject_id", "label"]].copy()
            te2["s"] = mdl.predict_proba(te[cols].to_numpy(float))[:, 1]
            sub = te2.groupby(["subject_id", "label"], as_index=False)["s"].mean()
            if sub.label.nunique() < 2:
                continue
            aucs.append(roc_auc_score(sub["label"], sub["s"]))
        rows.append({"n_train_subjects": n, "auc_mean": float(np.mean(aucs)),
                     "auc_std": float(np.std(aucs)), "n_valid_repeats": len(aucs)})
    res = pd.DataFrame(rows)
    res.to_csv(C.METRICS / "diag_learning_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(res["n_train_subjects"], res["auc_mean"], yerr=res["auc_std"],
                fmt="o-", capsize=4, color="#2e86c1")
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlabel("# training subjects"); ax.set_ylabel("held-out subject AUC")
    ax.set_ylim(0.4, 0.8)
    ax.set_title(f"Learning curve ({feature_set}, {repeats} repeats/point)\n"
                 "still rising → more data helps; flat → representation-limited")
    fig.savefig(C.FIGURES / "diag_learning_curve.png")
    plt.close(fig)

    slope = float(res["auc_mean"].iloc[-1] - res["auc_mean"].iloc[0]) if len(res) > 1 else 0.0
    return {"curve": rows, "auc_gain_10_to_50": slope}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> dict:
    df = C.load_trial_table()
    summary = {
        "per_subject": per_subject_breakdown(df),
        "caliper_sweep": caliper_sweep(df),
        "learning_curve": learning_curve(df),
    }
    (C.REPORTS / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2))

    ps = summary["per_subject"]
    print(f"=== Per-subject (shape_clean) ===")
    print(f"  accuracy@0.5 = {ps['accuracy_at_0.5']:.2f}  "
          f"({ps['n_correct']}/{ps['n_total']})  "
          f"wrong: {ps['controls_wrong']} control, {ps['pd_wrong']} PD")
    print(f"\n=== Caliper sweep ===")
    for r in summary["caliper_sweep"]["sweep"]:
        print(f"  caliper={r['caliper_s']:.2f}s  n={r['n_trials']:3d}  AUC={r['auc']:.3f}")
    print(f"\n=== Learning curve ===")
    for r in summary["learning_curve"]["curve"]:
        print(f"  n_train={r['n_train_subjects']:2d}  AUC={r['auc_mean']:.3f}±{r['auc_std']:.3f}")
    print(f"  gain 10→50 subjects: {summary['learning_curve']['auc_gain_10_to_50']:+.3f}")
    return summary


if __name__ == "__main__":
    main()

"""
Feature-based PD-vs-control baselines with subject-level LOSO-CV.

Runnable headless:  ``.venv/bin/python -m src.baseline``

This is the classical-ML floor the deep models (notebooks 08/09) must beat.
It runs entirely on the cached trial feature table — no .mat access, no GPU.

Design decisions that matter for a defensible result:
  * LEAVE-ONE-SUBJECT-OUT. Trials from one subject never span train/test.
  * Subject-level scoring. Trial probabilities are averaged to one score per
    held-out subject; metrics are computed on the 58 subject scores.
  * Multiple feature sets, including a DURATION-ONLY control. If a model can't
    beat "predict from trial length alone", it hasn't learned gait.
  * Permutation null. Subject labels are shuffled to build a null AUC
    distribution → an honest p-value for "better than chance".
  * Subject-bootstrap CIs (resample subjects, not trials).
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config as C
from . import stats as S

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})


# ---------------------------------------------------------------------------
# Models and feature sets
# ---------------------------------------------------------------------------


def _models() -> dict:
    return {
        "logreg": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0,
                                       class_weight="balanced",
                                       random_state=C.RANDOM_STATE)),
        ]),
        "svm_rbf": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, probability=True,
                        class_weight="balanced", random_state=C.RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=400, max_depth=None,
                                           class_weight="balanced", n_jobs=-1,
                                           random_state=C.RANDOM_STATE)),
        ]),
    }


def feature_sets(df: pd.DataFrame) -> dict:
    """Return named feature-column lists. `shape_clean` is data-driven."""
    inv = [c for c in C.DURATION_INVARIANT_FEATURES if c in df.columns]
    conf = [c for c in C.DURATION_CONFOUNDED_FEATURES if c in df.columns]
    # truly clean = invariant AND |spearman corr with duration| < 0.3
    dur_corr = S.correlation_with(df, inv, "duration_s", method="spearman").abs()
    clean = [f for f in inv if dur_corr.get(f, 0) < 0.3]
    return {
        "all_features": inv + conf,
        "duration_invariant": inv,
        "shape_clean": clean,
        "confounded_only": conf,
        "duration_only": ["duration_s"],
    }


# ---------------------------------------------------------------------------
# LOSO evaluation
# ---------------------------------------------------------------------------


def loso_subject_scores(df: pd.DataFrame, feature_cols: list[str], model) -> pd.DataFrame:
    """
    Run LOSO-CV; return one row per subject with the mean predicted PD prob.
    """
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy()
    groups = df["subject_id"].to_numpy()
    logo = LeaveOneGroupOut()

    trial_scores = np.full(len(df), np.nan)
    for tr, te in logo.split(X, y, groups):
        mdl = model
        mdl.fit(X[tr], y[tr])
        trial_scores[te] = mdl.predict_proba(X[te])[:, 1]

    res = df[["subject_id", "label"]].copy()
    res["trial_score"] = trial_scores
    subj = res.groupby(["subject_id", "label"], as_index=False)["trial_score"].mean()
    subj = subj.rename(columns={"trial_score": "subject_score"})
    return subj


def subject_metrics(subj: pd.DataFrame, threshold: float = 0.5) -> dict:
    y = subj["label"].to_numpy()
    s = subj["subject_score"].to_numpy()
    pred = (s >= threshold).astype(int)
    auc, lo, hi = S.bootstrap_metric_ci(
        y, s, subj["subject_id"].to_numpy(), roc_auc_score,
        n_resamples=C.BOOTSTRAP_RESAMPLES, random_state=C.RANDOM_STATE)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": auc, "auc_lo": lo, "auc_hi": hi,
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def permutation_pvalue(df, feature_cols, model, observed_auc, n=None) -> float:
    """Subject-label-shuffle null for AUC → one-sided p-value."""
    n = n or C.N_PERMUTATIONS
    rng = np.random.default_rng(C.RANDOM_STATE)
    subj_labels = df.drop_duplicates("subject_id").set_index("subject_id")["label"]
    null = np.empty(n)
    for i in range(n):
        permuted = pd.Series(
            rng.permutation(subj_labels.values), index=subj_labels.index)
        d2 = df.copy()
        d2["label"] = d2["subject_id"].map(permuted)
        subj = loso_subject_scores(d2, feature_cols, model)
        try:
            null[i] = roc_auc_score(subj["label"], subj["subject_score"])
        except ValueError:
            null[i] = 0.5
    return float((np.sum(null >= observed_auc) + 1) / (n + 1))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(run_permutation: bool = True) -> dict:
    df = C.load_trial_table()
    sets = feature_sets(df)
    models = _models()
    summary = {"feature_sets": {k: v for k, v in sets.items()}, "results": {}}

    # AUC grid: model × feature set
    grid = pd.DataFrame(index=list(models), columns=list(sets), dtype=float)
    roc_store = {}  # for the best model per set
    for sname, cols in sets.items():
        for mname, mdl in _models().items():  # fresh models each time
            subj = loso_subject_scores(df, cols, mdl)
            met = subject_metrics(subj)
            grid.loc[mname, sname] = met["auc"]
            summary["results"][f"{mname}|{sname}"] = met
            roc_store[(mname, sname)] = (subj["label"].to_numpy(),
                                         subj["subject_score"].to_numpy())

    grid.to_csv(C.METRICS / "baseline_auc_grid.csv")

    # permutation p-value for the headline model (logreg on shape_clean)
    headline = ("logreg", "shape_clean")
    if run_permutation:
        obs = summary["results"][f"{headline[0]}|{headline[1]}"]["auc"]
        p = permutation_pvalue(df, sets["shape_clean"], _models()[headline[0]],
                               obs, n=C.N_PERMUTATIONS)
        summary["permutation"] = {"model": "|".join(headline), "observed_auc": obs,
                                  "p_value": p, "n": C.N_PERMUTATIONS}

    _plot_auc_grid(grid)
    _plot_roc(roc_store, sets)
    summary["best"] = _best(summary["results"])
    (C.REPORTS / "baseline_summary.json").write_text(json.dumps(summary, indent=2))

    print("AUC grid (subject-level LOSO):")
    print(grid.round(3).to_string())
    if run_permutation:
        print(f"\nPermutation p (logreg|shape_clean): {summary['permutation']['p_value']:.4f}")
    print("\nBest:", json.dumps(summary["best"], indent=2))
    return summary


def _best(results: dict) -> dict:
    items = [(k, v["auc"]) for k, v in results.items()]
    k, a = max(items, key=lambda x: x[1])
    m = results[k]
    return {"config": k, "auc": a, "auc_ci": [m["auc_lo"], m["auc_hi"]],
            "sensitivity": m["sensitivity"], "specificity": m["specificity"]}


def _plot_auc_grid(grid: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(grid.to_numpy(dtype=float), cmap="viridis", vmin=0.4, vmax=1.0,
                   aspect="auto")
    ax.set_xticks(range(grid.shape[1])); ax.set_xticklabels(grid.columns, rotation=20, ha="right")
    ax.set_yticks(range(grid.shape[0])); ax.set_yticklabels(grid.index)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.75 else "black", fontweight="bold")
    fig.colorbar(im, label="subject-level ROC-AUC (LOSO)")
    ax.set_title("Baseline AUC: model × feature set\n(duration_only = confound floor)")
    fig.savefig(C.FIGURES / "baseline_auc_grid.png")
    plt.close(fig)


def _plot_roc(roc_store: dict, sets: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    # show logreg across feature sets to expose the confound
    for sname in ["all_features", "duration_invariant", "shape_clean",
                  "confounded_only", "duration_only"]:
        y, s = roc_store[("logreg", sname)]
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        ax.plot(fpr, tpr, lw=2, label=f"{sname}  AUC={auc:.2f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("1 − specificity"); ax.set_ylabel("sensitivity")
    ax.set_title("LogReg ROC by feature set (subject-level LOSO)")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(C.FIGURES / "baseline_roc.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

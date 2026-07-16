"""
Confound-control analyses — is the (weak) classical signal gait or trial length?

The baseline (`src/baseline.py`) showed subject-level AUC ~0.62 with a
duration-only floor of 0.61. This module runs the decisive follow-ups that
settle whether anything survives once trial length is removed:

  1. Per-test split. The duration confound lives in test1 (chair stand-up).
     If the signal is the confound, test2-only AUC collapses toward chance.
  2. Duration residualization. Linearly regress each feature on trial duration
     and classify on the residuals. If AUC collapses, the signal was duration.
  3. Duration-matched subset. Greedy 1:1 caliper matching on duration removes
     the group↔duration association non-parametrically; re-run the baseline.
  4. Feature importance. Which features the Random Forest actually leans on.

Runnable headless:  ``.venv/bin/python -m src.confound_analysis``
Figures → outputs/figures/confound_*.png ; summary → reports/confound_summary.json
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

from . import baseline as B
from . import config as C

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})

HEADLINE_SETS = ["all_features", "duration_invariant", "shape_clean", "duration_only"]


# ---------------------------------------------------------------------------
# 1. Per-test split
# ---------------------------------------------------------------------------


def per_test_baselines(df: pd.DataFrame) -> dict:
    sets = B.feature_sets(df)
    out: dict = {}
    for fs in HEADLINE_SETS:
        out[fs] = {}
        for test in ["both", "test1", "test2"]:
            sub = df if test == "both" else df[df["test"] == test]
            subj = B.loso_subject_scores(sub, sets[fs], B._models()["logreg"])
            out[fs][test] = float(roc_auc_score(subj["label"], subj["subject_score"]))
    return out


def _plot_per_test(res: dict) -> None:
    tests = ["both", "test1", "test2"]
    x = np.arange(len(HEADLINE_SETS))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, t in enumerate(tests):
        ax.bar(x + (i - 1) * w, [res[fs][t] for fs in HEADLINE_SETS], w, label=t)
    ax.axhline(0.5, color="k", ls="--", lw=0.8, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(HEADLINE_SETS, rotation=15, ha="right")
    ax.set_ylabel("subject-level LOSO AUC (logreg)")
    ax.set_ylim(0.35, 0.85)
    ax.set_title("AUC by test split — does test2 (no chair stand-up) hold up?")
    ax.legend()
    fig.savefig(C.FIGURES / "confound_per_test.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Duration residualization
# ---------------------------------------------------------------------------


def residualize_on_duration(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    d = df[["duration_s"]].to_numpy(float)
    for c in feature_cols:
        y = df[c].to_numpy(float)
        mask = np.isfinite(y) & np.isfinite(d[:, 0])
        if mask.sum() < 10:
            continue
        lr = LinearRegression().fit(d[mask], y[mask])
        res = y.copy()
        res[mask] = y[mask] - lr.predict(d[mask])
        out[c] = res
    return out


def residualized_baselines(df: pd.DataFrame) -> dict:
    sets = B.feature_sets(df)
    out = {}
    for fs in ["all_features", "duration_invariant", "shape_clean"]:
        cols = sets[fs]
        raw = B.subject_metrics(B.loso_subject_scores(df, cols, B._models()["logreg"]))["auc"]
        dfr = residualize_on_duration(df, cols)
        red = B.subject_metrics(B.loso_subject_scores(dfr, cols, B._models()["logreg"]))["auc"]
        out[fs] = {"raw_auc": float(raw), "residualized_auc": float(red)}
    return out


def _plot_residualized(res: dict) -> None:
    sets = list(res)
    x = np.arange(len(sets))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, [res[s]["raw_auc"] for s in sets], 0.4, label="raw features")
    ax.bar(x + 0.2, [res[s]["residualized_auc"] for s in sets], 0.4,
           label="duration-residualized")
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.axhline(0.61, color="grey", ls=":", lw=0.8, label="duration-only floor")
    ax.set_xticks(x); ax.set_xticklabels(sets, rotation=15, ha="right")
    ax.set_ylabel("subject-level LOSO AUC"); ax.set_ylim(0.35, 0.8)
    ax.set_title("Effect of linearly removing duration from each feature")
    ax.legend()
    fig.savefig(C.FIGURES / "confound_residualized.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Duration-matched subset (greedy 1:1 caliper matching)
# ---------------------------------------------------------------------------


def duration_matched_subset(df: pd.DataFrame, caliper: float = 0.75,
                            seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pd_ = df[df["group"] == "pd"].sample(frac=1.0, random_state=seed)
    ctrl = df[df["group"] == "control"]
    ctrl_idx = ctrl.index.to_numpy()
    ctrl_dur = ctrl["duration_s"].to_numpy(float)
    used = np.zeros(ctrl_idx.size, dtype=bool)
    keep = []
    for idx, d in zip(pd_.index, pd_["duration_s"].to_numpy(float)):
        diffs = np.where(used, np.inf, np.abs(ctrl_dur - d))
        j = int(np.argmin(diffs))
        if diffs[j] <= caliper:
            used[j] = True
            keep.append(idx)
            keep.append(int(ctrl_idx[j]))
    return df.loc[keep]


def matched_baseline(df: pd.DataFrame, caliper: float = 0.75) -> dict:
    m = duration_matched_subset(df, caliper=caliper)
    sets = B.feature_sets(df)
    c = m.loc[m.group == "control", "duration_s"]
    p = m.loc[m.group == "pd", "duration_s"]
    _, dur_p = sps.mannwhitneyu(p, c, alternative="two-sided") if len(c) and len(p) else (0, 1.0)
    res = {
        "n_trials": int(len(m)),
        "n_subjects": int(m.subject_id.nunique()),
        "n_pd_trials": int((m.group == "pd").sum()),
        "n_control_trials": int((m.group == "control").sum()),
        "duration_mwu_p_after_match": float(dur_p),
        "control_median_dur": float(c.median()), "pd_median_dur": float(p.median()),
        "auc": {},
    }
    for fs in ["all_features", "duration_invariant", "shape_clean"]:
        subj = B.loso_subject_scores(m, sets[fs], B._models()["logreg"])
        try:
            res["auc"][fs] = float(roc_auc_score(subj["label"], subj["subject_score"]))
        except ValueError:
            res["auc"][fs] = float("nan")
    return res, m


def _plot_matched(df: pd.DataFrame, m: pd.DataFrame, res: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, data, title in [(axes[0], df, "before matching (all trials)"),
                            (axes[1], m, "after duration matching")]:
        for g in ["control", "pd"]:
            v = data.loc[data.group == g, "duration_s"]
            ax.hist(v, bins=20, alpha=0.6, label=f"{g} (n={v.size})",
                    color=C.GROUP_COLOURS[g])
            ax.axvline(v.median(), color=C.GROUP_COLOURS[g], ls="--", lw=1.5)
        ax.set_xlabel("duration [s]"); ax.legend(fontsize=8); ax.set_title(title)
    axes[0].set_ylabel("trials")
    fig.suptitle(f"Duration matching (caliper 0.75 s) — post-match MWU p="
                 f"{res['duration_mwu_p_after_match']:.2f}, "
                 f"shape_clean AUC={res['auc']['shape_clean']:.2f}", y=1.02)
    fig.savefig(C.FIGURES / "confound_duration_matched.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Feature importance (Random Forest, permutation)
# ---------------------------------------------------------------------------


def feature_importance(df: pd.DataFrame, feature_set: str = "all_features") -> dict:
    cols = B.feature_sets(df)[feature_set]
    X = df[cols].to_numpy(float)
    y = df["label"].to_numpy()
    mask = np.isfinite(X).all(axis=1)
    X, y = X[mask], y[mask]
    mdl = B._models()["random_forest"].fit(X, y)
    pi = permutation_importance(mdl, X, y, n_repeats=20, random_state=C.RANDOM_STATE,
                               scoring="roc_auc")
    order = np.argsort(pi.importances_mean)[::-1]
    ranked = [(cols[i], float(pi.importances_mean[i]), float(pi.importances_std[i]))
              for i in order]
    fig, ax = plt.subplots(figsize=(8, 8))
    names = [r[0] for r in ranked][::-1]
    means = [r[1] for r in ranked][::-1]
    errs = [r[2] for r in ranked][::-1]
    conf = set(C.DURATION_CONFOUNDED_FEATURES)
    colours = ["#c0392b" if n in conf else "#2e86c1" for n in names]
    ax.barh(names, means, xerr=errs, color=colours)
    ax.set_xlabel("permutation importance (Δ AUC)  |  red = duration-confounded")
    ax.set_title(f"RF feature importance — {feature_set}")
    fig.savefig(C.FIGURES / "confound_feature_importance.png")
    plt.close(fig)
    return {"top5": ranked[:5]}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> dict:
    df = C.load_trial_table()
    per_test = per_test_baselines(df); _plot_per_test(per_test)
    resid = residualized_baselines(df); _plot_residualized(resid)
    matched, m = matched_baseline(df); _plot_matched(df, m, matched)
    importance = feature_importance(df)

    summary = {"per_test": per_test, "residualized": resid,
               "duration_matched": matched, "feature_importance": importance}
    (C.REPORTS / "confound_summary.json").write_text(json.dumps(summary, indent=2))

    print("=== Per-test AUC (logreg) ===")
    for fs in HEADLINE_SETS:
        print(f"  {fs:20s} both={per_test[fs]['both']:.3f}  "
              f"test1={per_test[fs]['test1']:.3f}  test2={per_test[fs]['test2']:.3f}")
    print("\n=== Duration-residualized AUC ===")
    for fs, v in resid.items():
        print(f"  {fs:20s} raw={v['raw_auc']:.3f} → residualized={v['residualized_auc']:.3f}")
    print(f"\n=== Duration-matched subset ===")
    print(f"  {matched['n_trials']} trials, {matched['n_subjects']} subjects, "
          f"post-match duration MWU p={matched['duration_mwu_p_after_match']:.2f}")
    for fs, a in matched["auc"].items():
        print(f"  {fs:20s} AUC={a:.3f}")
    print("\n=== Top RF features (permutation) ===")
    for n, mean, sd in importance["top5"]:
        print(f"  {n:32s} ΔAUC={mean:+.4f} ± {sd:.4f}")
    return summary


if __name__ == "__main__":
    main()

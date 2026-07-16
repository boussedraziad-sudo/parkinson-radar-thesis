"""
Statistical helpers for the EDA and results sections.

Deliberately dependency-light (numpy + scipy + statsmodels). Everything here
returns plain numbers / DataFrames so it can be dropped straight into a thesis
table or a figure caption.

Why these choices:
  - Mann-Whitney U: non-parametric, no normality assumption (gait features are
    skewed and the sample is small).
  - Benjamini-Hochberg FDR: ~25 features tested at once → control false
    discoveries without the over-conservatism of Bonferroni.
  - Cliff's delta: non-parametric effect size; a p-value alone says nothing
    about magnitude with n this small.
  - Subject-level bootstrap: trials within a subject are correlated, so CIs
    must resample subjects, not trials.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cliff's delta effect size in [-1, 1].

    delta = P(a > b) - P(a < b). |delta| thresholds (Romano et al.):
    <0.147 negligible, <0.33 small, <0.474 medium, else large.
    Computed via the U statistic for O(n log n) rather than O(n*m).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n, m = a.size, b.size
    if n == 0 or m == 0:
        return float("nan")
    # rank-based: U = sum of ranks of a in pooled - n(n+1)/2
    pooled = np.concatenate([a, b])
    ranks = stats.rankdata(pooled)
    u = ranks[:n].sum() - n * (n + 1) / 2.0
    return float(2.0 * u / (n * m) - 1.0)


def cliffs_magnitude(delta: float) -> str:
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def group_comparison_table(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str = "group",
    pos: str = "pd",
    neg: str = "control",
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Mann-Whitney U per feature with BH-FDR correction and Cliff's delta.

    Returns one row per feature sorted by corrected p-value, with medians,
    effect size + magnitude label, and a `significant_fdr` flag.
    """
    rows = []
    for c in feature_cols:
        a = df.loc[df[group_col] == pos, c].dropna().to_numpy()
        b = df.loc[df[group_col] == neg, c].dropna().to_numpy()
        if a.size < 3 or b.size < 3:
            continue
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        delta = cliffs_delta(a, b)
        rows.append({
            "feature": c,
            f"{neg}_median": float(np.median(b)),
            f"{pos}_median": float(np.median(a)),
            "cliffs_delta": delta,
            "effect": cliffs_magnitude(delta),
            "u_stat": float(u),
            "p_raw": float(p),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    rej, p_adj, *_ = multipletests(out["p_raw"], alpha=fdr_alpha, method="fdr_bh")
    out["p_fdr"] = p_adj
    out["significant_fdr"] = rej
    return out.sort_values("p_fdr").reset_index(drop=True)


def correlation_with(
    df: pd.DataFrame, feature_cols: list[str], target: str, method: str = "spearman"
) -> pd.Series:
    """Absolute-sorted correlation of each feature with a target column."""
    sub = df[feature_cols + [target]].dropna()
    corr = sub.corr(method=method)[target].drop(target)
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    metric_fn,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    random_state: int = 0,
) -> tuple[float, float, float]:
    """
    Subject-level bootstrap CI for a score-based metric (e.g. ROC-AUC).

    Resamples *subjects* (not rows) with replacement to respect the clustered
    structure. Returns (point_estimate, lo, hi).
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    groups = np.asarray(groups)
    uniq = np.unique(groups)

    point = float(metric_fn(y_true, y_score))
    stats_list = []
    for _ in range(n_resamples):
        sampled = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled])
        yt, ys = y_true[idx], y_score[idx]
        if np.unique(yt).size < 2:
            continue
        stats_list.append(metric_fn(yt, ys))
    lo = float(np.percentile(stats_list, 100 * alpha / 2))
    hi = float(np.percentile(stats_list, 100 * (1 - alpha / 2)))
    return point, lo, hi


def describe_by_group(df: pd.DataFrame, col: str, group_col: str = "group") -> pd.DataFrame:
    """Median / IQR / mean / std of `col` per group + the percentage gap of medians."""
    g = df.groupby(group_col)[col]
    out = g.agg(["count", "median", "mean", "std",
                 lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
    out.columns = ["n", "median", "mean", "std", "q25", "q75"]
    return out

"""
Exploratory data analysis on the trial-level feature table.

Runnable headless:  ``.venv/bin/python -m src.eda``
Produces figures in outputs/figures/eda_*.png, tables in outputs/metrics/,
and a machine-readable summary in reports/eda_summary.json.

All statistics use src.stats (non-parametric, FDR-corrected, effect sizes).
Plots carry the statistics in their titles/annotations so a figure is
self-contained for the thesis.
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import stats as S

plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10})


# ---------------------------------------------------------------------------
# Individual analyses (each returns a small dict that lands in the summary)
# ---------------------------------------------------------------------------


def fig_class_balance(df: pd.DataFrame) -> dict:
    subj = df.drop_duplicates("subject_id")
    by_group = subj["group"].value_counts()
    tri = df.groupby("subject_id").size()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(by_group.index, by_group.values,
                color=[C.GROUP_COLOURS[g] for g in by_group.index])
    for i, v in enumerate(by_group.values):
        axes[0].text(i, v + 0.3, str(v), ha="center", fontweight="bold")
    axes[0].set_title(f"Subjects by group (N={subj.shape[0]})")
    axes[0].set_ylabel("subjects")

    axes[1].hist(tri.values, bins=range(4, 9), align="left", color="#888", edgecolor="white")
    axes[1].set_title("Trials per subject")
    axes[1].set_xlabel("n trials")
    axes[1].set_ylabel("subjects")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "eda_class_balance.png")
    plt.close(fig)
    return {
        "n_subjects": int(subj.shape[0]),
        "subjects_per_group": by_group.to_dict(),
        "n_trials": int(df.shape[0]),
        "trials_per_subject": tri.value_counts().to_dict(),
    }


def fig_duration_confound(df: pd.DataFrame) -> dict:
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, test in zip(axes, ["test1", "test2"]):
        sub = df[df["test"] == test]
        for g in ["control", "pd"]:
            v = sub.loc[sub["group"] == g, "duration_s"].dropna()
            ax.hist(v, bins=20, alpha=0.65, label=f"{g} (n={v.size})",
                    color=C.GROUP_COLOURS[g])
            ax.axvline(v.median(), color=C.GROUP_COLOURS[g], ls="--", lw=1.5)
        c = sub.loc[sub.group == "control", "duration_s"].dropna()
        p = sub.loc[sub.group == "pd", "duration_s"].dropna()
        u, pval = sps.mannwhitneyu(p, c, alternative="two-sided")
        delta = S.cliffs_delta(p.to_numpy(), c.to_numpy())
        gap = 100 * (p.median() - c.median()) / c.median()
        ax.set_title(f"{test}: PD {gap:+.1f}% longer\nMWU p={pval:.1e}, δ={delta:+.2f}")
        ax.set_xlabel("trial duration [s]")
        ax.legend(fontsize=8)
        out[test] = {"pct_gap": float(gap), "mwu_p": float(pval), "cliffs_delta": float(delta)}
    axes[0].set_ylabel("trials")
    fig.suptitle("Duration confound — PD trials run longer, concentrated in test1", y=1.02)
    fig.savefig(C.FIGURES / "eda_duration_confound.png")
    plt.close(fig)
    return out


def fig_group_comparison(df: pd.DataFrame) -> dict:
    feats = [c for c in (C.DURATION_INVARIANT_FEATURES + C.DURATION_CONFOUNDED_FEATURES)
             if c in df.columns]
    table = S.group_comparison_table(df, feats)
    table.to_csv(C.METRICS / "eda_group_comparison.csv", index=False)

    # effect-size bar chart, coloured by confound status
    t = table.copy()
    conf = set(C.DURATION_CONFOUNDED_FEATURES)
    t["is_conf"] = t["feature"].isin(conf)
    t = t.sort_values("cliffs_delta")
    colours = ["#c0392b" if x else "#2e86c1" for x in t["is_conf"]]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh(t["feature"], t["cliffs_delta"], color=colours)
    ax.axvline(0, color="k", lw=0.8)
    for thr in (0.147, 0.33, 0.474):
        for s in (-1, 1):
            ax.axvline(s * thr, color="grey", ls=":", lw=0.6)
    ax.set_xlabel("Cliff's delta  (PD − control)   |  red = duration-confounded")
    ax.set_title("Per-feature effect size (trial-level, Mann–Whitney + BH-FDR)")
    fig.savefig(C.FIGURES / "eda_effect_sizes.png")
    plt.close(fig)

    sig = table[table["significant_fdr"]]
    return {
        "n_features_tested": int(table.shape[0]),
        "n_significant_fdr": int(sig.shape[0]),
        "top5_by_effect": table.reindex(table["cliffs_delta"].abs().sort_values(
            ascending=False).index).head(5)[
            ["feature", "cliffs_delta", "effect", "p_fdr"]].to_dict("records"),
    }


def fig_top_feature_distributions(df: pd.DataFrame, k: int = 6) -> dict:
    feats = [c for c in C.DURATION_INVARIANT_FEATURES if c in df.columns]
    table = S.group_comparison_table(df, feats)
    top = table.reindex(table["cliffs_delta"].abs().sort_values(ascending=False).index).head(k)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (_, row) in zip(axes.flat, top.iterrows()):
        c = row["feature"]
        data = [df.loc[df.group == "control", c].dropna(),
                df.loc[df.group == "pd", c].dropna()]
        parts = ax.violinplot(data, showmedians=True)
        for pc, g in zip(parts["bodies"], ["control", "pd"]):
            pc.set_facecolor(C.GROUP_COLOURS[g]); pc.set_alpha(0.6)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["control", "pd"])
        ax.set_title(f"{c}\nδ={row['cliffs_delta']:+.2f} ({row['effect']}), "
                     f"p_fdr={row['p_fdr']:.1e}", fontsize=9)
    fig.suptitle("Top duration-invariant discriminators (trial-level)", y=1.01)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "eda_top_feature_distributions.png")
    plt.close(fig)
    return {"shown": top["feature"].tolist()}


def fig_correlation_and_duration_audit(df: pd.DataFrame) -> dict:
    feats = [c for c in (C.DURATION_INVARIANT_FEATURES + C.DURATION_CONFOUNDED_FEATURES)
             if c in df.columns]
    # correlation heatmap
    corr = df[feats].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=90, fontsize=7)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=7)
    fig.colorbar(im, fraction=0.045)
    ax.set_title("Feature correlation (Spearman)")
    fig.savefig(C.FIGURES / "eda_correlation.png")
    plt.close(fig)

    # duration audit
    dur_corr = S.correlation_with(df, feats, "duration_s", method="spearman")
    fig, ax = plt.subplots(figsize=(8, 8))
    colours = ["#c0392b" if f in set(C.DURATION_CONFOUNDED_FEATURES) else "#2e86c1"
               for f in dur_corr.index]
    ax.barh(dur_corr.index[::-1], dur_corr.values[::-1], color=colours[::-1])
    ax.axvline(0, color="k", lw=0.8)
    for s in (-0.3, 0.3):
        ax.axvline(s, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("Spearman corr with trial duration  |  red = flagged confounded")
    ax.set_title("Duration-leakage audit per feature")
    fig.savefig(C.FIGURES / "eda_duration_audit.png")
    plt.close(fig)

    leaky = dur_corr[dur_corr.abs() > 0.3]
    return {
        "features_corr_duration_gt_0.3": {k: round(float(v), 3) for k, v in leaky.items()},
        "max_abs_corr_invariant": round(float(
            dur_corr[[f for f in C.DURATION_INVARIANT_FEATURES if f in dur_corr.index]]
            .abs().max()), 3),
    }


def fig_pca(df: pd.DataFrame) -> dict:
    feats = [c for c in C.DURATION_INVARIANT_FEATURES if c in df.columns]
    X = df[feats].to_numpy()
    mask = np.isfinite(X).all(axis=1)
    X, meta = X[mask], df.loc[mask, "group"].to_numpy()
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(10, len(feats)), random_state=C.RANDOM_STATE)
    Xp = pca.fit_transform(Xs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for g in ["control", "pd"]:
        m = meta == g
        axes[0].scatter(Xp[m, 0], Xp[m, 1], s=22, alpha=0.6,
                        c=C.GROUP_COLOURS[g], label=g)
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0].legend(); axes[0].set_title("PCA — duration-invariant features")

    cum = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(range(1, len(cum) + 1), cum, "o-")
    axes[1].axhline(0.9, color="grey", ls=":")
    axes[1].set_xlabel("components"); axes[1].set_ylabel("cumulative variance")
    axes[1].set_title("Scree (cumulative)")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "eda_pca.png")
    plt.close(fig)
    return {
        "pc1_var": float(pca.explained_variance_ratio_[0]),
        "pc2_var": float(pca.explained_variance_ratio_[1]),
        "n_pc_for_90pct": int(np.searchsorted(cum, 0.9) + 1),
    }


def main() -> dict:
    df = C.load_trial_table()
    summary = {
        "class_balance": fig_class_balance(df),
        "duration_confound": fig_duration_confound(df),
        "group_comparison": fig_group_comparison(df),
        "top_distributions": fig_top_feature_distributions(df),
        "correlation_audit": fig_correlation_and_duration_audit(df),
        "pca": fig_pca(df),
    }
    (C.REPORTS / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print("EDA complete. Figures →", C.FIGURES)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()

"""
Central configuration: paths, feature groupings, experiment defaults.

Everything that was previously hard-coded across notebooks/modules lives here
so a single edit propagates. Import as ``from src import config as C``.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"                       # symlink → ../shared_ziad
OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
METRICS = OUTPUTS / "metrics"
PREPROCESSED = OUTPUTS / "preprocessed"
MODELS = OUTPUTS / "models"
REPORTS = PROJECT_ROOT / "reports"                      # generated analysis write-ups

for _d in (FIGURES, METRICS, PREPROCESSED, MODELS, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

# Canonical derived tables (produced by EDA / forensics notebooks)
TRIAL_FEATURES_CSV = METRICS / "trial_features.csv"
INVENTORY_CSV = METRICS / "inventory.csv"
QC_REPORT_CSV = METRICS / "qc_report.csv"

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

GROUP_TO_LABEL = {"control": 0, "pd": 1}
LABEL_TO_GROUP = {v: k for k, v in GROUP_TO_LABEL.items()}
GROUP_COLOURS = {"control": "#3a86b8", "pd": "#e8543f"}

# ---------------------------------------------------------------------------
# Feature groupings
#
# The cached `trial_features.csv` was produced by an early version of
# `features.py` that still included raw energy sums. We keep an explicit split
# so analyses can demonstrate the duration confound rather than fall victim to
# it. See `../project_summary_and_setup.md` (duration-confound note).
# ---------------------------------------------------------------------------

# Raw time-summed energies → scale with trial length → leak the duration confound.
DURATION_CONFOUNDED_FEATURES = [
    "foot_total_energy", "torso_total_energy",
    "foot_torso_band_energy", "foot_foot_band_energy",
    "torso_torso_band_energy", "torso_foot_band_energy",
    "foot_env_std", "torso_env_std",
]

# Per-pixel statistics, spectral-shape descriptors and ratios → duration-invariant.
DURATION_INVARIANT_FEATURES = [
    "foot_mean", "foot_max", "foot_std",
    "foot_centroid_hz", "foot_bandwidth_hz", "foot_entropy",
    "foot_foot_to_torso_band_ratio",
    "torso_mean", "torso_max", "torso_std",
    "torso_centroid_hz", "torso_bandwidth_hz", "torso_entropy",
    "torso_foot_to_torso_band_ratio",
    "foot_torso_total_ratio",
    "foot_env_cv", "torso_env_cv",
]

META_COLUMNS = ["subject_id", "group", "test", "trial"]

# ---------------------------------------------------------------------------
# Experiment defaults
# ---------------------------------------------------------------------------

RANDOM_STATE = 0
N_PERMUTATIONS = 1000          # label-shuffle null for the baseline AUC
BOOTSTRAP_RESAMPLES = 2000     # subject-level bootstrap CI on metrics


def load_trial_table() -> "pandas.DataFrame":  # noqa: F821
    """Load trial features merged with trial durations; add integer label."""
    import pandas as pd

    feat = pd.read_csv(TRIAL_FEATURES_CSV)
    if INVENTORY_CSV.exists():
        inv = pd.read_csv(INVENTORY_CSV)[["subject_id", "test", "trial", "duration_s"]]
        feat = feat.merge(inv, on=["subject_id", "test", "trial"], how="left")
    feat["label"] = feat["group"].map(GROUP_TO_LABEL).astype(int)
    return feat

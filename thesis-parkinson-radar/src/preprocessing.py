"""
Preprocessing pipeline.

For each trial, this module:
  1. Loads `ce_foot` and `ce_torso` (or `|stft_foot|` / `|stft_torso|`).
  2. Splits along the time axis into fixed-length windows.
  3. Normalises each window (log + Z-score by default).
  4. Resizes to a common (H, W) — default 224x224 for ResNet compatibility.
  5. Saves one `.npy` per window under outputs/preprocessed/<group>/.
  6. Writes a manifest CSV listing every window with its label and provenance.

Training reads from the manifest + .npy cache; the original `.mat` files are
opened only once during this pass.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import radar_params as P
from .data_loader import DATA_ROOT, TrialPath, iter_trials, load_trial


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PreprocConfig:
    """Frozen 2026-08-06. Alternatives listed here become the §4.5 ablations."""

    window_s: float = 3.0          # ~2-3 gait cycles at a 1.0-1.2 s stride time
    stride_s: float = 1.5          # 50 % overlap (ablation: 2.0 / 0.5)
    representation: str = "ce"     # "ce" -> ce_foot/ce_torso; "stft_abs" -> |stft_*|
    out_size: tuple[int, int] = (224, 224)  # (H, W) for the resized spectrogram
    # "log" caches log1p ONLY and defers standardisation to the Dataset, where
    # the mean/std are computed from training subjects alone. That is the
    # leak-free default. "log_zscore" standardises each window against itself,
    # which destroys absolute amplitude (the bradykinesia carrier) and is kept
    # as the normalisation ablation.
    normalise: str = "log"         # "log" | "log_zscore" | "log_minmax" | "none"
    dtype: str = "float32"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def window_indices(t_axis: np.ndarray, window_s: float, stride_s: float) -> list[tuple[int, int]]:
    """Return (start_idx, end_idx_exclusive) pairs along the time axis."""
    if t_axis.size < 2:
        return []
    dt = float(np.median(np.diff(t_axis)))
    if dt <= 0:
        return []
    win = max(2, int(round(window_s / dt)))
    stride = max(1, int(round(stride_s / dt)))
    return [(s, s + win) for s in range(0, t_axis.size - win + 1, stride)]


def normalise_window(arr: np.ndarray, mode: str) -> np.ndarray:
    arr = arr.astype(np.float32)
    if mode == "log":
        # Compress the dynamic range and stop there. Standardisation happens in
        # WindowDataset using statistics from the training subjects only, so the
        # held-out subject never influences the scale its own input is measured on.
        return np.log1p(np.maximum(arr, 0)).astype(np.float32)
    if mode == "log_zscore":
        x = np.log1p(np.maximum(arr, 0))
        m, s = x.mean(), x.std() + 1e-6
        return ((x - m) / s).astype(np.float32)
    if mode == "log_minmax":
        x = np.log1p(np.maximum(arr, 0))
        lo, hi = x.min(), x.max()
        return ((x - lo) / max(hi - lo, 1e-6)).astype(np.float32)
    if mode == "none":
        return arr
    raise ValueError(f"unknown normalise mode {mode!r}")


def positive_energy_fraction(win: np.ndarray, doppler: np.ndarray) -> float:
    """Share of a window's energy at positive Doppler, i.e. moving towards the node.

    A window inside a single walking pass sits near 0 (walking away) or near 1
    (walking back). A window straddling the turn sits near 0.5. The source
    publication computes its gait parameters only after discarding the turning,
    standing and sitting stages; caching this per window lets us reproduce that
    choice as an ablation instead of always training on the whole recording.
    """
    d = np.asarray(doppler, dtype=float).ravel()
    if d.size != win.shape[0]:
        return float("nan")
    e = np.maximum(np.asarray(win, dtype=float), 0.0)
    # The contrast-enhanced arrays carry a per-recording background floor that
    # is direction-agnostic, so summing raw magnitude drags every window's
    # fraction toward 0.5 and dilutes the turn label. Remove each Doppler bin's
    # median over time (the stationary background) before comparing halves.
    e = np.maximum(e - np.median(e, axis=1, keepdims=True), 0.0)
    e = e.sum(axis=1)
    tot = float(e.sum())
    return float(e[d > 0].sum() / tot) if tot > 0 else float("nan")


def resize_2d(arr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear resize via torch (avoids extra deps; torch is required anyway)."""
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(arr.astype(np.float32))[None, None]  # (1, 1, H, W)
    # The time axis is downsampled ~21x; without an anti-aliasing prefilter,
    # bilinear sampling turns the spiky foot band into phase-dependent noise
    # that differs between two windows of the same walk.
    out = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False,
                        antialias=True)
    return out.squeeze().numpy()


# ---------------------------------------------------------------------------
# Trial-level processing
# ---------------------------------------------------------------------------


def process_trial(tp: TrialPath, cfg: PreprocConfig, out_root: Path) -> list[dict]:
    """Process one trial -> N windowed .npy files. Returns manifest rows."""
    if cfg.representation == "ce":
        tr = load_trial(tp.path, representation="ce")
        foot = tr["ce_foot"]
        torso = tr["ce_torso"]
    elif cfg.representation == "stft_abs":
        tr = load_trial(tp.path, representation="stft")
        foot = np.abs(tr["stft_foot"]).astype(np.float32)
        torso = np.abs(tr["stft_torso"]).astype(np.float32)
    else:
        raise ValueError(f"unknown representation {cfg.representation!r}")
    t = np.asarray(tr["t_axis"], dtype=float).ravel()
    dop = np.asarray(tr["doppler"], dtype=float).ravel()

    spans = window_indices(t, cfg.window_s, cfg.stride_s)
    rows: list[dict] = []
    for w_idx, (s, e) in enumerate(spans):
        foot_raw, torso_raw = foot[:, s:e], torso[:, s:e]
        foot_w = resize_2d(normalise_window(foot_raw, cfg.normalise), cfg.out_size)
        torso_w = resize_2d(normalise_window(torso_raw, cfg.normalise), cfg.out_size)
        arr = np.stack([foot_w, torso_w], axis=0)  # (2, H, W)

        out_dir = out_root / tp.group
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{tp.subject_id}_{tp.test}_{tp.trial}_w{w_idx:03d}.npy"
        np.save(out_path, arr.astype(cfg.dtype))

        rows.append({
            "subject_id": tp.subject_id,
            "group": tp.group,
            "label": 1 if tp.group == "pd" else 0,
            "test": tp.test,
            "trial": tp.trial,
            "window_idx": w_idx,
            "n_windows_trial": len(spans),
            "t_start_s": float(t[s]),
            "t_end_s": float(t[e - 1]),
            "n_doppler_bins_in": int(foot.shape[0]),
            "n_time_bins_in": int(e - s),
            # Per-window, per-channel statistics of the CACHED array. Storing them
            # lets any fold recover exact training-set mean/std from the manifest
            # alone, without re-reading a single .npy (law of total variance).
            "foot_mean": float(foot_w.mean()), "foot_std": float(foot_w.std()),
            "torso_mean": float(torso_w.mean()), "torso_std": float(torso_w.std()),
            # Direction of travel, for the turn-exclusion ablation.
            "pos_frac_foot": positive_energy_fraction(foot_raw, dop),
            "npy_path": str(out_path.relative_to(out_root)),
        })
    return rows


# ---------------------------------------------------------------------------
# Fold-level statistics, derived from the manifest alone
# ---------------------------------------------------------------------------

CHANNELS = ("foot", "torso")


def fold_norm_stats(manifest: pd.DataFrame, subjects: Iterable[str]) -> np.ndarray:
    """Per-channel (mean, std) over the windows of `subjects` only.

    Exact, and costs no disk reads: every window holds the same number of
    pixels, so the pooled mean is the mean of the per-window means and the
    pooled variance follows from the law of total variance,

        Var(X) = E[Var(X | window)] + Var(E[X | window]).

    Called once per fold with the TRAINING subjects, so the held-out subject
    never contributes to the scale its own input is measured against.
    """
    sub = manifest[manifest["subject_id"].isin(list(subjects))]
    if sub.empty:
        raise ValueError("fold_norm_stats: no windows for the requested subjects")
    out = np.zeros((len(CHANNELS), 2), dtype=np.float64)
    for i, ch in enumerate(CHANNELS):
        m = sub[f"{ch}_mean"].to_numpy(float)
        s = sub[f"{ch}_std"].to_numpy(float)
        mu = m.mean()
        var = (s ** 2).mean() + m.var()
        out[i] = (mu, np.sqrt(max(var, 1e-12)))
    return out


def add_window_weights(manifest: pd.DataFrame) -> pd.DataFrame:
    """Attach the inverse-window-count weight that neutralises the length confound.

    A longer recording yields more windows, so without this a slow subject
    contributes proportionally more gradient purely for being slow. Window
    count alone reaches AUC 0.621 on this corpus, so it is a real confound and
    not a theoretical one. Weights are normalised to average 1.0 so the
    effective learning rate is unchanged.
    """
    df = manifest.copy()
    per_subject = df.groupby("subject_id")["npy_path"].transform("size")
    df["n_windows_subject"] = per_subject
    w = 1.0 / per_subject
    df["sample_weight"] = (w / w.mean()).astype(float)
    return df


# ---------------------------------------------------------------------------
# Dataset-level processing
# ---------------------------------------------------------------------------


def preprocess_dataset(
    cfg: PreprocConfig | None = None,
    out_root: Path | None = None,
    excluded_paths: Iterable[Path] = (),
    limit: int | None = None,
) -> pd.DataFrame:
    """Iterate the dataset, write .npy windows, return + save manifest CSV."""
    cfg = cfg or PreprocConfig()
    out_root = Path(out_root) if out_root else (DATA_ROOT.parent / "outputs" / "preprocessed")
    out_root.mkdir(parents=True, exist_ok=True)
    excluded = {str(p) for p in excluded_paths}

    trials = [tp for tp in iter_trials() if str(tp.path) not in excluded]
    if limit:
        trials = trials[:limit]

    all_rows: list[dict] = []
    failed: list[tuple[TrialPath, str]] = []
    for tp in tqdm(trials, desc="preprocessing"):
        try:
            all_rows.extend(process_trial(tp, cfg, out_root))
        except Exception as exc:  # noqa: BLE001
            failed.append((tp, f"{type(exc).__name__}: {exc}"))

    manifest = pd.DataFrame(all_rows)
    manifest_path = out_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(f"\n  Windows written:   {len(manifest)}")
    print(f"  Trials processed:  {len(trials) - len(failed)} / {len(trials)}")
    print(f"  Manifest:          {manifest_path}")
    if failed:
        print(f"  Failed trials:     {len(failed)}")
        for tp, msg in failed[:10]:
            print(f"    {tp.subject_id}/{tp.test}/{tp.trial}: {msg}")
    return manifest

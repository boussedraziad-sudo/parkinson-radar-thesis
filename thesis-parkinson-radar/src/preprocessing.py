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
    window_s: float = 3.0          # window length in seconds
    stride_s: float = 1.5          # window stride (50% overlap by default)
    representation: str = "ce"     # "ce" -> ce_foot/ce_torso; "stft_abs" -> |stft_*|
    out_size: tuple[int, int] = (224, 224)  # (H, W) for the resized spectrogram
    normalise: str = "log_zscore"  # "log_zscore" | "log_minmax" | "none"
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


def resize_2d(arr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear resize via torch (avoids extra deps; torch is required anyway)."""
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(arr.astype(np.float32))[None, None]  # (1, 1, H, W)
    out = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
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
    t = np.asarray(tr["t_axis"], dtype=float)

    rows: list[dict] = []
    for w_idx, (s, e) in enumerate(window_indices(t, cfg.window_s, cfg.stride_s)):
        foot_w = normalise_window(foot[:, s:e], cfg.normalise)
        torso_w = normalise_window(torso[:, s:e], cfg.normalise)
        foot_w = resize_2d(foot_w, cfg.out_size)
        torso_w = resize_2d(torso_w, cfg.out_size)
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
            "t_start_s": float(t[s]),
            "t_end_s": float(t[e - 1]),
            "n_doppler_bins_in": int(foot.shape[0]),
            "n_time_bins_in": int(e - s),
            "npy_path": str(out_path.relative_to(out_root)),
        })
    return rows


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

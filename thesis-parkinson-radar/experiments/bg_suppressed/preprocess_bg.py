#!/usr/bin/env python3
"""Build the background-suppressed window cache (see CONFIG.md, frozen first).

Mirrors src.preprocessing.process_trial / preprocess_dataset exactly, with the
two pre-registered changes applied per recording BEFORE windowing:

  1. per-Doppler-bin background subtraction (median over time, clipped at 0),
     the same correction positive_energy_fraction v2 applies for turn labels;
  2. crop Doppler rows outside +-500 Hz (gait band) before resizing.

Everything downstream (windowing, log1p, antialiased 224x224 resize, manifest
schema) reuses the unchanged building blocks from src.preprocessing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_loader import TrialPath, iter_trials, load_trial
from src.preprocessing import (PreprocConfig, normalise_window,
                               positive_energy_fraction, resize_2d,
                               window_indices)

GAIT_BAND_HZ = 500.0
CACHE = Path(__file__).resolve().parent / "cache"


def suppress_background(arr: np.ndarray) -> np.ndarray:
    """Remove each Doppler bin's stationary background floor, per recording.

    Identical arithmetic to the v2 positive_energy_fraction correction:
    clip at zero, subtract the per-bin median over time, clip at zero again.
    """
    e = np.maximum(np.asarray(arr, dtype=np.float32), 0.0)
    return np.maximum(e - np.median(e, axis=1, keepdims=True), 0.0)


def process_trial_bg(tp: TrialPath, cfg: PreprocConfig, out_root: Path) -> list[dict]:
    """process_trial with background suppression + gait-band crop, else identical."""
    assert cfg.representation == "ce"
    tr = load_trial(tp.path, representation="ce")
    t = np.asarray(tr["t_axis"], dtype=float).ravel()
    dop = np.asarray(tr["doppler"], dtype=float).ravel()

    # Change 1: per-recording background subtraction, before windowing.
    foot = suppress_background(tr["ce_foot"])
    torso = suppress_background(tr["ce_torso"])

    # Change 2: crop to the gait band before windowing/resizing.
    keep = np.abs(dop) <= GAIT_BAND_HZ
    foot, torso, dop = foot[keep], torso[keep], dop[keep]

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
            "foot_mean": float(foot_w.mean()), "foot_std": float(foot_w.std()),
            "torso_mean": float(torso_w.mean()), "torso_std": float(torso_w.std()),
            "pos_frac_foot": positive_energy_fraction(foot_raw, dop),
            "npy_path": str(out_path.relative_to(out_root)),
        })
    return rows


def main() -> None:
    cfg = PreprocConfig()  # frozen defaults: 3.0/1.5 s, "ce", "log", 224x224
    CACHE.mkdir(parents=True, exist_ok=True)
    trials = list(iter_trials())

    all_rows: list[dict] = []
    failed: list[tuple[TrialPath, str]] = []
    for tp in tqdm(trials, desc="bg-suppressed preprocessing"):
        try:
            all_rows.extend(process_trial_bg(tp, cfg, CACHE))
        except Exception as exc:  # noqa: BLE001
            failed.append((tp, f"{type(exc).__name__}: {exc}"))

    manifest = pd.DataFrame(all_rows)
    manifest_path = CACHE / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(f"\n  Windows written:   {len(manifest)}")
    print(f"  Trials processed:  {len(trials) - len(failed)} / {len(trials)}")
    print(f"  Manifest:          {manifest_path}")
    if failed:
        print(f"  Failed trials:     {len(failed)}")
        for tp, msg in failed[:10]:
            print(f"    {tp.subject_id}/{tp.test}/{tp.trial}: {msg}")


if __name__ == "__main__":
    main()

"""
PyTorch Dataset over the preprocessed .npy window cache.

The manifest CSV (built by preprocessing.preprocess_dataset) is the single
source of truth: it tells the Dataset which file goes with which label and
which subject. Subject filtering is what makes LOSO-CV trivial — pass the
held-out subject(s) to the val Dataset and the rest to the train Dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Reads `.npy` windows referenced by a manifest DataFrame."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        root: str | Path,
        subjects: Sequence[str] | None = None,
        augment: bool = False,
        channel_mode: str = "both",
        norm_stats: np.ndarray | None = None,
        aug_time_flip: bool = True,
        aug_noise_std: float = 0.02,
        rng_seed: int | None = None,
        max_turn_frac: float | None = None,
    ):
        """
        norm_stats
            (n_channels, 2) array of (mean, std) from `fold_norm_stats`, always
            computed on TRAINING subjects. Applied at load time. None leaves the
            cached values untouched, which is correct only when the cache was
            written with a self-contained normalisation such as ``log_zscore``.
        max_turn_frac
            If set, drop windows whose share of positive-Doppler energy lies
            within `max_turn_frac` of 0.5, i.e. windows straddling the turn. The
            source publication discards turning, standing and sitting before
            computing gait parameters; this reproduces that choice.
        """
        self.root = Path(root)
        self.augment = augment
        self.channel_mode = channel_mode
        self.norm_stats = None if norm_stats is None else np.asarray(norm_stats, np.float32)
        self.aug_time_flip = aug_time_flip
        self.aug_noise_std = float(aug_noise_std)
        self._rng = np.random.default_rng(rng_seed)

        df = manifest
        if subjects is not None:
            df = df[df["subject_id"].isin(list(subjects))]
        if max_turn_frac is not None and "pos_frac_foot" in df.columns:
            keep = (df["pos_frac_foot"] - 0.5).abs() > max_turn_frac
            df = df[keep.fillna(True)]
        self.manifest = df.reset_index(drop=True)
        self._has_weights = "sample_weight" in self.manifest.columns

    def __len__(self) -> int:
        return len(self.manifest)

    def _select_channels(self, arr: np.ndarray) -> np.ndarray:
        # arr is (2, H, W) from preprocessing (foot, torso).
        if self.channel_mode == "both":
            return arr
        if self.channel_mode == "foot":
            return arr[:1]
        if self.channel_mode == "torso":
            return arr[1:2]
        if self.channel_mode == "both_3ch":
            return np.concatenate([arr, ((arr[0] + arr[1]) / 2)[None]], axis=0)
        if self.channel_mode == "diff":
            return np.stack([arr[0], arr[0] - arr[1]], axis=0)
        raise ValueError(self.channel_mode)

    def _standardise(self, arr: np.ndarray) -> np.ndarray:
        """Apply the training-fold mean/std, per channel."""
        if self.norm_stats is None:
            return arr
        n = min(arr.shape[0], self.norm_stats.shape[0])
        out = arr.copy()
        for c in range(n):
            mu, sd = self.norm_stats[c]
            out[c] = (out[c] - mu) / max(float(sd), 1e-6)
        # Derived channels (e.g. "diff", the averaged 3rd channel) inherit the
        # last real channel's scale rather than being left unnormalised.
        for c in range(n, arr.shape[0]):
            mu, sd = self.norm_stats[n - 1]
            out[c] = (out[c] - mu) / max(float(sd), 1e-6)
        return out

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        """Time reversal plus mild additive noise.

        Reversing time turns a walk away from the node into a walk towards it,
        which is a real thing a subject does twice per recording, so it is
        label-preserving here. It is a flag rather than a constant because it
        also destroys any asymmetry between the outward and return passes.
        """
        if self.aug_time_flip and self._rng.random() < 0.5:
            arr = arr[..., ::-1].copy()
        if self.aug_noise_std > 0 and self._rng.random() < 0.7:
            arr = arr + self._rng.normal(0.0, self.aug_noise_std, arr.shape).astype(np.float32)
        return arr

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, float]:
        """Returns (window, label, sample_weight).

        The weight is the inverse-window-count weight from
        `preprocessing.add_window_weights`, or 1.0 when it was not applied. It
        is carried per sample so the training loss can neutralise the
        window-count confound without resampling the data.
        """
        row = self.manifest.iloc[idx]
        arr = np.load(self.root / row["npy_path"]).astype(np.float32)
        arr = self._select_channels(arr)
        arr = self._standardise(arr)
        if self.augment:
            arr = self._augment(arr)
        w = float(row["sample_weight"]) if self._has_weights else 1.0
        return torch.from_numpy(np.ascontiguousarray(arr)), int(row["label"]), w

    @property
    def subjects(self) -> list[str]:
        return sorted(self.manifest["subject_id"].unique().tolist())

    @property
    def sample_weights(self) -> np.ndarray:
        """Per-window weights, if `add_window_weights` was applied upstream."""
        if "sample_weight" not in self.manifest.columns:
            return np.ones(len(self.manifest), dtype=np.float64)
        return self.manifest["sample_weight"].to_numpy(float)

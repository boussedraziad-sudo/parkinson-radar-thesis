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
    ):
        self.root = Path(root)
        self.augment = augment
        self.channel_mode = channel_mode

        df = manifest
        if subjects is not None:
            df = df[df["subject_id"].isin(list(subjects))]
        self.manifest = df.reset_index(drop=True)

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

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        # Mild augmentation: time flip, small additive Gaussian noise
        if np.random.rand() < 0.5:
            arr = arr[..., ::-1].copy()
        if np.random.rand() < 0.7:
            arr = arr + np.random.normal(0.0, 0.02, arr.shape).astype(np.float32)
        return arr

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.manifest.iloc[idx]
        arr = np.load(self.root / row["npy_path"]).astype(np.float32)
        arr = self._select_channels(arr)
        if self.augment:
            arr = self._augment(arr)
        return torch.from_numpy(arr), int(row["label"])

    @property
    def subjects(self) -> list[str]:
        return sorted(self.manifest["subject_id"].unique().tolist())

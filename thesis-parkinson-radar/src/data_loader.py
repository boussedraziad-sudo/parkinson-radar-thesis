"""
Load `stft_data.mat` files and iterate the dataset.

The dataset is described in `../shared_ziad/readme.txt`. Each trial's .mat
contains (Doppler-freq x time) matrices for foot and torso, in both complex
STFT and contrast-enhanced magnitude form, plus per-radar arrays and axes.

The dataset on disk is MATLAB v5 (not v7.3), so we use `scipy.io.loadmat`.
v5 preserves the MATLAB shape, so arrays come out already as (doppler, time);
no transpose is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Optional

import numpy as np
import scipy.io as sio

# ---------------------------------------------------------------------------
# Path conventions
# ---------------------------------------------------------------------------

# `data/` is a symlink to the dataset root (shared_ziad/).
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

GROUP_BY_PREFIX = {"fisc": "control", "fisp": "pd", "fis": "prodromal"}


# ---------------------------------------------------------------------------
# Lightweight trial descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialPath:
    """Identifies a single recording on disk."""

    subject_id: str  # e.g. "fisc_005"
    group: str  # "control" | "pd" | "prodromal"
    test: str  # "test1" | "test2"
    trial: str  # "trial1" | "trial2" | "trial3" | (rarely "trial4")
    path: Path

    @property
    def key(self) -> str:
        return f"{self.subject_id}/{self.test}/{self.trial}"


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------


def iter_trials(root: Path = DATA_ROOT) -> Iterator[TrialPath]:
    """Yield every `stft_data.mat` under `root` in deterministic order."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}. "
            "Did you create the data symlink? See README.md."
        )

    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        prefix = subject_dir.name.split("_", 1)[0]
        group = GROUP_BY_PREFIX.get(prefix)
        if group is None:
            continue  # skip unknown directories like a future readme.txt copy
        for test_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            for trial_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
                mat = trial_dir / "stft_data.mat"
                if mat.exists():
                    yield TrialPath(
                        subject_id=subject_dir.name,
                        group=group,
                        test=test_dir.name,
                        trial=trial_dir.name,
                        path=mat,
                    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

KEYS_MAGNITUDE = ("ce_foot", "ce_torso")
KEYS_COMPLEX = ("stft_foot", "stft_torso")
KEYS_PER_RADAR = ("all_doppler_time", "all_ce_time")


def _load_mat(path: Path, variables: list[str]) -> dict:
    """Load only the requested top-level variables from a MATLAB v5 .mat file."""
    return sio.loadmat(str(path), variable_names=variables, squeeze_me=False)


def load_trial(
    path: str | Path,
    representation: Literal["ce", "stft", "both"] = "ce",
    include_per_radar: bool = False,
) -> dict:
    """
    Open one trial's `stft_data.mat` and return a dict.

    Parameters
    ----------
    path
        Path to the `stft_data.mat` file.
    representation
        - "ce"   (default): return `ce_foot`, `ce_torso` (real magnitude)
        - "stft" : return `stft_foot`, `stft_torso` (complex)
        - "both" : return all four
    include_per_radar
        If True, also include `all_doppler_time` / `all_ce_time` (large arrays).

    Returns
    -------
    dict with at minimum the requested spectrograms plus `t_axis`, `doppler`,
    and provenance fields. Shape convention: (doppler, time).
    """
    path = Path(path)

    variables: list[str] = ["t_axis_target", "doppler_axis"]
    if representation in ("ce", "both"):
        variables += list(KEYS_MAGNITUDE)
    if representation in ("stft", "both"):
        variables += list(KEYS_COMPLEX)
    if include_per_radar:
        variables += list(KEYS_PER_RADAR)
        variables += ["idx_foot_max_snr", "idx_torso_max_snr"]

    raw = _load_mat(path, variables)

    out: dict = {"path": str(path)}
    out["t_axis"] = np.asarray(raw["t_axis_target"]).squeeze().astype(float)
    out["doppler"] = np.asarray(raw["doppler_axis"]).squeeze().astype(float)

    if representation in ("ce", "both"):
        out["ce_foot"] = np.ascontiguousarray(raw["ce_foot"]).astype(np.float32)
        out["ce_torso"] = np.ascontiguousarray(raw["ce_torso"]).astype(np.float32)
    if representation in ("stft", "both"):
        out["stft_foot"] = np.ascontiguousarray(raw["stft_foot"]).astype(np.complex64)
        out["stft_torso"] = np.ascontiguousarray(raw["stft_torso"]).astype(np.complex64)

    if include_per_radar:
        if "all_ce_time" in raw:
            out["all_ce_time"] = np.ascontiguousarray(raw["all_ce_time"]).astype(np.float32)
        for k in ("idx_foot_max_snr", "idx_torso_max_snr"):
            if k in raw:
                out[k] = np.asarray(raw[k]).squeeze()

    return out


# ---------------------------------------------------------------------------
# Quick inventory helpers
# ---------------------------------------------------------------------------


def list_keys(path: str | Path) -> list[str]:
    """Return the top-level variable names of a .mat file."""
    info = sio.whosmat(str(path))
    return [name for name, _shape, _dtype in info]


def quick_shape_info(path: str | Path, key: str = "ce_foot") -> Optional[tuple[int, ...]]:
    """Return the shape of one variable without loading data."""
    for name, shape, _dtype in sio.whosmat(str(path)):
        if name == key:
            return tuple(shape)
    return None


if __name__ == "__main__":
    # Simple smoke test: print first few trials and their groups.
    for i, t in enumerate(iter_trials()):
        print(f"{t.group:9s}  {t.subject_id}/{t.test}/{t.trial}  ->  {t.path.name}")
        if i >= 5:
            break

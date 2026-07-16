"""
Quality control for radar gait spectrograms.

Thresholds and expected ranges come from `src/radar_params.py`, which in turn
quotes the reference paper (López-Delgado et al., 2026). The readme of
`shared_ziad/` warns that the v1 release needs cleaning, so this module's
job is to identify which trials to exclude before training.

Two tiers of checks:
  1. CHEAP (metadata only via `scipy.io.whosmat`):
     - file opens, required variables present, shapes consistent
     - Doppler axis range and bin spacing match the dataset's actual layout
     - trial duration in plausible band
  2. DEEP (loads ce_foot / ce_torso via `scipy.io.loadmat`):
     - no NaN / Inf
     - spectrograms have non-trivial dynamic range (not all-flat)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio

from . import radar_params as P


REQUIRED_KEYS = (
    "ce_foot", "ce_torso", "stft_foot", "stft_torso",
    "t_axis_target", "doppler_axis",
)
SPECTROGRAM_KEYS = ("ce_foot", "ce_torso", "stft_foot", "stft_torso")


# ---------------------------------------------------------------------------
# Cheap checks
# ---------------------------------------------------------------------------


def _present_keys(info: list[tuple]) -> dict[str, tuple]:
    """Return {var_name: shape} for variables present in the .mat header."""
    return {name: tuple(shape) for name, shape, _ in info}


def check_required_keys(present: dict[str, tuple]) -> list[str]:
    missing = [k for k in REQUIRED_KEYS if k not in present]
    return [f"missing_keys:{','.join(missing)}"] if missing else []


def check_shape_consistency(present: dict[str, tuple]) -> list[str]:
    """All four spectrogram variables should share the same shape."""
    shapes = {k: present[k] for k in SPECTROGRAM_KEYS if k in present}
    if len(shapes) < 2:
        return []  # already covered by missing-keys
    if len(set(shapes.values())) > 1:
        return [f"shape_mismatch:{shapes}"]
    return []


def check_doppler_axis(d: np.ndarray) -> list[str]:
    out: list[str] = []
    if d.size == 0:
        return ["doppler_axis_empty"]

    # range: |max| or |min| should at least reach ~half the paper's 500 Hz band
    if d.max() < 0.5 * P.DOPPLER_EXPECTED_MAX_HZ or d.min() > 0.5 * P.DOPPLER_EXPECTED_MIN_HZ:
        out.append(f"doppler_range_narrow:[{d.min():.0f},{d.max():.0f}]Hz")

    # bin spacing: tolerate 0.5x-2x the dataset's actual 5 Hz spacing
    if d.size > 1:
        res = float(abs(np.median(np.diff(d))))
        lo, hi = 0.5 * P.DOPPLER_BIN_SPACING_HZ, 2.0 * P.DOPPLER_BIN_SPACING_HZ
        if not (lo <= res <= hi):
            out.append(f"doppler_bin_spacing_off:{res:.2f}Hz")

    return out


def check_trial_duration(t: np.ndarray) -> list[str]:
    if t.size < 2:
        return ["t_axis_too_short"]
    dur = float(t.max() - t.min())
    if dur < P.TRIAL_DURATION_S_MIN:
        return [f"trial_too_short:{dur:.1f}s"]
    if dur > P.TRIAL_DURATION_S_MAX:
        return [f"trial_too_long:{dur:.1f}s"]
    return []


# ---------------------------------------------------------------------------
# Deep checks
# ---------------------------------------------------------------------------


def check_no_invalid_values(ce_foot: np.ndarray, ce_torso: np.ndarray) -> list[str]:
    out = []
    for name, arr in (("ce_foot", ce_foot), ("ce_torso", ce_torso)):
        if not np.isfinite(arr).all():
            out.append(f"{name}_has_nan_or_inf")
    return out


def check_dynamic_range(ce_foot: np.ndarray, ce_torso: np.ndarray) -> list[str]:
    """Flag flat-line spectrograms (all-zero, all-equal, or near-DC noise)."""
    out = []
    for name, arr in (("ce_foot", ce_foot), ("ce_torso", ce_torso)):
        rng = float(arr.max() - arr.min())
        if rng < 1e-6:
            out.append(f"{name}_flat:range={rng:.2g}")
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def qc_file(path: str | Path, deep: bool = True) -> dict:
    """
    Run QC checks on one trial file.

    Returns a dict with:
      - path, ok (bool), n_fails (int), fails_str (semicolon-joined)
      - cheap metadata (doppler range/spacing, duration, shapes) for the report
    """
    path = Path(path)
    rec: dict = {"path": str(path), "ok": False, "fails": []}

    try:
        info = sio.whosmat(str(path))
        present = _present_keys(info)
        rec["fails"].extend(check_required_keys(present))
        rec["fails"].extend(check_shape_consistency(present))

        # Read only the axes for cheap checks (small arrays).
        axes_needed = [k for k in ("doppler_axis", "t_axis_target") if k in present]
        axes = sio.loadmat(str(path), variable_names=axes_needed, squeeze_me=True)

        if "doppler_axis" in axes:
            d = np.asarray(axes["doppler_axis"]).astype(float).ravel()
            rec["fails"].extend(check_doppler_axis(d))
            rec["doppler_min"] = float(d.min()) if d.size else float("nan")
            rec["doppler_max"] = float(d.max()) if d.size else float("nan")
            rec["doppler_n"] = int(d.size)

        if "t_axis_target" in axes:
            t = np.asarray(axes["t_axis_target"]).astype(float).ravel()
            rec["fails"].extend(check_trial_duration(t))
            rec["duration_s"] = float(t.max() - t.min()) if t.size > 1 else 0.0
            rec["t_n"] = int(t.size)

        if "ce_foot" in present:
            rec["ce_foot_shape"] = "x".join(map(str, present["ce_foot"]))
        if "ce_torso" in present:
            rec["ce_torso_shape"] = "x".join(map(str, present["ce_torso"]))

        if deep and "ce_foot" in present and "ce_torso" in present:
            full = sio.loadmat(str(path),
                               variable_names=["ce_foot", "ce_torso"],
                               squeeze_me=False)
            ce_foot = np.asarray(full["ce_foot"], dtype=np.float32)
            ce_torso = np.asarray(full["ce_torso"], dtype=np.float32)
            rec["fails"].extend(check_no_invalid_values(ce_foot, ce_torso))
            rec["fails"].extend(check_dynamic_range(ce_foot, ce_torso))

    except Exception as e:  # noqa: BLE001
        rec["fails"].append(f"open_failed:{type(e).__name__}:{e}")

    rec["ok"] = not rec["fails"]
    rec["n_fails"] = len(rec["fails"])
    rec["fails_str"] = "; ".join(rec["fails"])
    return rec


def qc_dataset(trial_paths: Iterable[Path], deep: bool = True) -> list[dict]:
    """Run qc_file over an iterable of trial paths and return the list of records."""
    return [qc_file(p, deep=deep) for p in trial_paths]

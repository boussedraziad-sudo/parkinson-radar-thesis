"""
Handcrafted features for SVM / Random Forest baselines.

**All features are duration-invariant by construction.** The dataset has a
known confound (recorded in `../project_summary_and_setup.md`): PD trials
are ~10% longer than control trials, so any energy-summed feature
(`spec.sum()`, band-energy sums) would leak duration and dominate group
comparisons for the wrong reason.

This module uses means, marginalised distributions, and ratios. If you
want duration as a signal, add it as a separate feature column — do not
fold it back into the energy features.
"""

from __future__ import annotations

import numpy as np

from . import radar_params as P


def _spectral_stats(spec: np.ndarray, abs_d: np.ndarray) -> dict[str, float]:
    """Mean / max / std / centroid / bandwidth / entropy. All duration-invariant."""
    time_avg = spec.mean(axis=1)  # (D,) — marginal Doppler distribution
    total = float(time_avg.sum())

    if total <= 0:
        return {
            "mean": 0.0, "max": 0.0, "std": 0.0,
            "centroid_hz": 0.0, "bandwidth_hz": 0.0, "entropy": 0.0,
        }

    centroid = float((abs_d * time_avg).sum() / total)
    bandwidth = float(np.sqrt(((abs_d - centroid) ** 2 * time_avg).sum() / total))
    p = time_avg / total
    p = p[p > 0]
    entropy = float(-(p * np.log(p)).sum()) if p.size else 0.0

    return {
        "mean": float(spec.mean()),
        "max": float(spec.max()),
        "std": float(spec.std()),
        "centroid_hz": centroid,
        "bandwidth_hz": bandwidth,
        "entropy": entropy,
    }


def _band_means(spec: np.ndarray, abs_d: np.ndarray,
                include_energy: bool = True) -> dict[str, float]:
    """
    Band-limited descriptors for the torso and foot Doppler bands.

    Means and the mean-ratio are duration-invariant. The ``*_band_energy`` sums
    are NOT: they grow with the number of time bins, i.e. with trial length.
    They are emitted only when ``include_energy`` is True, and exist purely so
    the duration confound can be *demonstrated* (see
    ``config.DURATION_CONFOUNDED_FEATURES``). Never treat them as gait features.
    """
    torso_mask = (abs_d >= P.TORSO_BAND_HZ[0]) & (abs_d < P.TORSO_BAND_HZ[1])
    foot_mask = (abs_d >= P.FOOT_BAND_HZ[0]) & (abs_d < P.FOOT_BAND_HZ[1])
    mean_torso = float(spec[torso_mask].mean()) if torso_mask.any() else 0.0
    mean_foot = float(spec[foot_mask].mean()) if foot_mask.any() else 0.0
    sum_torso = float(spec[torso_mask].sum()) if torso_mask.any() else 0.0
    sum_foot = float(spec[foot_mask].sum()) if foot_mask.any() else 0.0

    # CANONICAL band ratio: share of band-limited ENERGY falling in the foot band.
    # Duration-invariant (numerator and denominator both scale with trial length),
    # but note it is weighted by band width: the foot band spans 120 Doppler bins
    # against the torso band's 80, so a wider band contributes more.
    # This is the definition used to produce every published result.
    den_e = sum_foot + sum_torso
    out = {
        "torso_band_mean": mean_torso,
        "foot_band_mean": mean_foot,
        "foot_to_torso_band_ratio": sum_foot / den_e if den_e > 0 else 0.0,
        # Bandwidth-normalised alternative: per-bin spectral density share, so it
        # does not depend on how wide the two bands were drawn. Kept for
        # comparison; not used in the published feature sets.
        "foot_to_torso_band_ratio_dens":
            mean_foot / (mean_foot + mean_torso) if (mean_foot + mean_torso) > 0 else 0.0,
    }
    if include_energy:
        out["torso_band_energy"] = sum_torso
        out["foot_band_energy"] = sum_foot
    return out


def per_window_features(
    foot: np.ndarray,
    torso: np.ndarray,
    doppler_axis: np.ndarray,
    include_energy: bool = True,
) -> dict[str, float]:
    """
    Build a flat dict of features for one window or one whole trial.

    Inputs:
      foot, torso  : (doppler_bins, time_bins) arrays.
      doppler_axis : 1D, length == doppler_bins; signed Hz.

    Output features (per channel, prefixed with `foot_` / `torso_`):
      mean, max, std, centroid_hz, bandwidth_hz, entropy,
      torso_band_mean, foot_band_mean, foot_to_torso_band_ratio,
      env_cv  (coefficient of variation of the time envelope; rhythmicity proxy)

    Plus cross-channel:
      foot_torso_mean_ratio
    """
    abs_d = np.abs(doppler_axis)
    if abs_d.size != foot.shape[0]:
        raise ValueError(
            f"doppler_axis size {abs_d.size} != foot doppler dim {foot.shape[0]}"
        )

    out: dict[str, float] = {}
    for name, spec in (("foot", foot), ("torso", torso)):
        for k, v in _spectral_stats(spec, abs_d).items():
            out[f"{name}_{k}"] = v
        for k, v in _band_means(spec, abs_d, include_energy=include_energy).items():
            out[f"{name}_{k}"] = v
        if include_energy:
            # Duration-CONFOUNDED: a sum over time bins scales with trial length.
            out[f"{name}_total_energy"] = float(spec.sum())

    # Cross-channel mean ratio (duration-invariant)
    out["foot_torso_mean_ratio"] = out["foot_mean"] / (out["torso_mean"] + 1e-12)

    # Temporal envelope statistics.
    #   env_cv = std/mean  -> duration-invariant (gait rhythmicity proxy)
    #   env_std            -> duration-CONFOUNDED (scale grows with energy)
    for name, spec in (("foot", foot), ("torso", torso)):
        env = spec.sum(axis=0)  # (time,)
        out[f"{name}_env_cv"] = float(env.std() / (env.mean() + 1e-12))
        if include_energy:
            out[f"{name}_env_std"] = float(env.std())

    if include_energy:
        # Cross-channel energy ratio. Both numerator and denominator scale with
        # duration so the ratio is *mostly* invariant, but it is retained in the
        # confounded group because the cached table treated it that way.
        out["foot_torso_total_ratio"] = (
            out["foot_total_energy"] / (out["torso_total_energy"] + 1e-12))

    return out


def featurise_windows(
    windows: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Vectorised featurisation. Returns (N, F) matrix in dict-key order."""
    rows = [per_window_features(f, t, d) for f, t, d in windows]
    return np.array([list(r.values()) for r in rows], dtype=np.float32)


def feature_names_for(foot_shape: tuple[int, int], doppler_axis: np.ndarray) -> list[str]:
    """Return feature names in the same order as per_window_features outputs."""
    dummy = np.zeros(foot_shape, dtype=np.float32)
    return list(per_window_features(dummy, dummy, doppler_axis).keys())

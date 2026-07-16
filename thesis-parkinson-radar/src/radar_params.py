"""
Radar and STFT parameters from the reference paper:
    López-Delgado et al., "Radar Network for Gait Monitoring: Technology and
    Validation", IEEE TBME, 2026.

These describe the FMCW radar network that generated our dataset
(`shared_ziad/`). All numbers here are quoted from the paper unless marked
"derived" or "assumed". When a parameter is not stated in the paper, the
constant carries a `# not stated in paper` comment with the chosen default
and rationale.
"""

from __future__ import annotations

# -- Physical constants -----------------------------------------------------
C_LIGHT = 2.998e8  # m/s


# -- RF configuration (Section II, Table II; Section III intro) -------------
F_CARRIER_HZ = 23.0e9       # 23 GHz centre frequency
BANDWIDTH_HZ = 1.4e9        # 1.4 GHz chirp bandwidth
T_CHIRP_S = 625e-6          # 625 µs chirp duration
EIRP_DBM = 15.5             # ±1 dBm
HPF_CUTOFF_HZ = 10.0        # Range-time clutter HPF cutoff (Section II, step 1)

WAVELENGTH_M = C_LIGHT / F_CARRIER_HZ  # ≈ 0.01303 m


# -- STFT parameters (Section II, step 3) -----------------------------------
SLOW_TIME_RATE_HZ = 1.0 / T_CHIRP_S          # ≈ 1600 Hz, input rate to the STFT
STFT_WINDOW_S = 50e-3                        # 50 ms Hann window
STFT_WINDOW_SAMPLES = int(round(STFT_WINDOW_S * SLOW_TIME_RATE_HZ))  # ≈ 80 (derived)
STFT_HOP_SAMPLES = 1                         # 1-sample hop in the paper
STFT_OVERLAP_FRAC = 1.0 - STFT_HOP_SAMPLES / STFT_WINDOW_SAMPLES     # ≈ 0.9875 (derived)
STFT_FFT_SIZE = None  # not stated in paper — confirm from `doppler_axis.size` per file


# -- Doppler axis (derived from STFT window) --------------------------------
DOPPLER_RESOLUTION_HZ = 1.0 / STFT_WINDOW_S  # ≈ 20 Hz (derived, intrinsic resolution from the 50 ms window)
DOPPLER_NYQUIST_HZ = SLOW_TIME_RATE_HZ / 2   # ≈ 800 Hz (derived)
# The files in this dataset store 320 Doppler bins over ±800 Hz → 5 Hz bin spacing.
# That is 4× finer than the paper's intrinsic resolution; consistent with a 4× zero-padded FFT.
DOPPLER_BIN_SPACING_HZ = 5.0
# Paper figures display ±500 Hz; this is the band where gait content lives.
DOPPLER_EXPECTED_MIN_HZ = -500.0
DOPPLER_EXPECTED_MAX_HZ = +500.0


# -- Body-part velocity bands (Section III-B, Fig. 7) -----------------------
# Torso reflects at low Doppler (v ~1 m/s → |f_D| ~ 150 Hz)
# Feet reflect at high Doppler (v ~3-4 m/s → |f_D| up to ±500 Hz)
TORSO_BAND_HZ = (0.0, 200.0)        # |Doppler| band where torso dominates
FOOT_BAND_HZ = (200.0, 500.0)       # |Doppler| band where feet dominate


# -- Trial timing -----------------------------------------------------------
# Paper test protocol: TUG (trial 1, "10 reps at quick pace"); per-trial
# duration in seconds is not stated. Our dataset has 3 trials × 2 tests.
# Plausible band for any single trial:
TRIAL_DURATION_S_MIN = 5.0          # assumed lower bound (a meaningful walk)
TRIAL_DURATION_S_MAX = 180.0        # assumed upper bound (TUG×10 + slack)


# -- Network geometry (Section III, Fig. 4) ---------------------------------
N_FOOT_RADARS = 2                   # configuration C5 (paper's validation network)
N_TORSO_RADARS = 1                  # configuration C5 (C6 uses 2 torso radars)
WALK_CORRIDOR_M = 3.0
RADAR_TO_SUBJECT_M = 1.0
TORSO_HEIGHT_M = 1.0
FEET_HEIGHT_M = 0.15
BEAMWIDTH_DEG = 40


# -- Paper validation cohort (for context only — NOT our dataset) -----------
PAPER_COHORT = {
    "G1_young_healthy":  {"n": 3, "age_range": (21, 24)},
    "G2_older_healthy":  {"n": 2, "age_range": (55, 71)},
    "G3_parkinson":      {"n": 3, "age_range": (59, 78)},
}


# -- Helpers ----------------------------------------------------------------

def doppler_to_velocity_m_s(f_d_hz: float) -> float:
    """Convert Doppler frequency to radial velocity (paper Eq. relating both)."""
    return f_d_hz * WAVELENGTH_M / 2.0


def velocity_to_doppler_hz(v_m_s: float) -> float:
    """Inverse of doppler_to_velocity_m_s."""
    return 2.0 * v_m_s / WAVELENGTH_M

#!/usr/bin/env python3
"""Figure for the methods section "Coupling between the two channels".

Claim illustrated: in healthy walking each heel strike shows up at the same
instant in both channels, as a valley of the foot signature's upper contour
(the foot is planted, its speed near zero) and a local peak of the trunk speed
(the torso is fastest while the weight transfers over the standing leg).

Recording: the same healthy control used by guide_data_explained.png
(fisc_048, test2/trial1, 8.0 s). The stretch shown is the walk back toward
the radars, after the turn, where the gait is steady and the Doppler sign is
constant.

Both traces are drawn ON the signature, at their true Doppler values (review
request: "align the contour with the peaks of the corresponding micro-Doppler
signature"):
  * foot contour: per time column, the highest Doppler bin on the travel side
    whose log-magnitude exceeds a fixed fraction of the stretch's peak level,
    smoothed with a 100 ms Hann window. It traces the top of each swing arch
    and dips to a valley when the foot is planted.
  * trunk speed: energy-weighted mean Doppler of |ce_torso| over 15-250 Hz on
    the direction-of-travel side, smoothed with a 150 ms Hann window, drawn in
    Hz on the torso band itself.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

ROOT = Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import vizstyle as V                                       # noqa: E402
V.apply()

from src.data_loader import load_trial                     # noqa: E402

FIG = ROOT / "outputs/figures"
TFM_FIG = Path("/Users/ziad.boussedra/Desktop/Master Thesis/Ziad_Boussedra_TFM/figures")

# ── the recording and the stretch ────────────────────────────────────────────
REC = ROOT / "data/fisc_048/test2/trial1/stft_data.mat"   # control, 8.0 s
T0, T1 = 6.50, 9.75          # walk back toward the radars (turn ends ~6.3 s)
DETECT0, DETECT1 = 6.60, 9.72  # keep detections away from the smoothing edges
SIGN = +1                     # direction of travel: positive Doppler

tr = load_trial(REC)
t = np.asarray(tr["t_axis"], float).ravel()
d = np.asarray(tr["doppler"], float).ravel()
foot, torso = tr["ce_foot"], tr["ce_torso"]
dt = float(np.median(np.diff(t)))


def smooth(x: np.ndarray, seconds: float) -> np.ndarray:
    k = max(3, int(round(seconds / dt)))
    w = np.hanning(k)
    return np.convolve(x, w / w.sum(), mode="same")


view = (t >= T0) & (t <= T1)

def contours(spec, lo_hz, hi_hz, frac=0.45, sec=0.10):
    """Upper and lower edge of a signature on the travel side: per time column,
    the highest / lowest Doppler bin whose log-magnitude exceeds `frac` of the
    stretch's 99.5th percentile, interpolated over gaps and smoothed."""
    G = np.log1p(np.maximum(spec, 0))
    rows = (d * SIGN >= lo_hz) & (d * SIGN <= hi_hz)
    dd_ = np.abs(d[rows])
    thr_ = frac * np.percentile(G[rows][:, view], 99.5)
    up = np.full(t.shape, np.nan); lo = np.full(t.shape, np.nan)
    for j in np.where(view)[0]:
        idx = np.where(G[rows, j] >= thr_)[0]
        if idx.size:
            up[j], lo[j] = dd_[idx].max(), dd_[idx].min()
    ok_ = ~np.isnan(up) & view
    return (smooth(np.interp(t, t[ok_], up[ok_]), sec),
            smooth(np.interp(t, t[ok_], lo[ok_]), sec))

F = np.log1p(np.maximum(foot, 0))
foot_env, foot_low = contours(foot, 15, 780)      # foot: 15-780 Hz
torso_up, torso_low = contours(torso, 15, 330)    # torso band outline

# trunk speed: energy-weighted mean Doppler, travel side, 15-250 Hz
rows_t = (d * SIGN >= 15) & (d * SIGN <= 250)
E = np.abs(torso[rows_t, :])
cen = smooth((E * np.abs(d[rows_t])[:, None]).sum(axis=0) / (E.sum(axis=0) + 1e-9),
             0.15)

# ── heel-strike detection inside the stretch ────────────────────────────────
sel = (t >= DETECT0) & (t <= DETECT1)
ts, fe, ce = t[sel], foot_env[sel], cen[sel]
dist = int(0.35 / dt)                       # steps are ~0.6 s apart here
mins, _ = find_peaks(-fe, distance=dist, prominence=(fe.max() - fe.min()) * 0.12)
pks, _ = find_peaks(ce, distance=dist, prominence=(ce.max() - ce.min()) * 0.06)
t_min = ts[mins]
keep = (ts[pks] >= t_min.min() - 0.35) & (ts[pks] <= t_min.max() + 0.35)
pks = pks[keep]
t_pk = ts[pks]
print("foot-contour valleys at:", np.round(t_min, 2))
print("trunk-speed peaks at:   ", np.round(t_pk, 2))
offsets = [t_pk[np.argmin(np.abs(t_pk - m))] - m for m in t_min]
print("valley-to-nearest-peak offsets [s]:", np.round(offsets, 3))

# ── figure ──────────────────────────────────────────────────────────────────
fig, (axF, axT) = plt.subplots(
    2, 1, figsize=(15.0, 7.8), sharex=True,
    gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.09})

for ax in (axF, axT):
    ax.grid(False)

axF.imshow(F, aspect="auto", origin="lower",
           extent=[t[0], t[-1], d[0], d[-1]], cmap=V.SEQ)
axF.set_xlim(T0, T1)
axF.set_ylim(-60, 800)
axF.set_ylabel("Doppler [Hz]")

axT.imshow(np.log1p(np.maximum(torso, 0)), aspect="auto", origin="lower",
           extent=[t[0], t[-1], d[0], d[-1]], cmap=V.SEQ)
axT.set_ylim(-60, 330)
axT.set_ylabel("Doppler [Hz]")
axT.set_xlabel("time [s]")

# foot contours drawn on the signature at their true Doppler values:
# the upper edge (bold, carries the markers) and the lower edge (thin)
axF.plot(t[view], SIGN * foot_env[view], color="white", lw=2.2, alpha=0.95,
         solid_capstyle="round", zorder=5)
axF.plot(t[view], SIGN * foot_low[view], color="white", lw=1.4, alpha=0.9,
         ls=(0, (6, 3)), solid_capstyle="round", zorder=5)
min_y = SIGN * fe[mins]
axF.plot(t_min, min_y, "v", ms=10, color="white", markeredgecolor="#1a1a2e",
         markeredgewidth=1.2, zorder=6)

# torso band outline (thin) plus the trunk-speed trace (bold) on the band
axT.plot(t[view], SIGN * torso_up[view], color="white", lw=1.4, alpha=0.9,
         ls=(0, (6, 3)), solid_capstyle="round", zorder=5)
axT.plot(t[view], SIGN * torso_low[view], color="white", lw=1.4, alpha=0.9,
         ls=(0, (6, 3)), solid_capstyle="round", zorder=5)
show = (t >= t_min.min() - 0.30) & (t <= t_pk.max() + 0.33)
axT.plot(t[show], SIGN * cen[show], color="white", lw=2.2, alpha=0.95,
         solid_capstyle="round", zorder=5)
pk_y = SIGN * ce[pks]
axT.plot(t_pk, pk_y, "^", ms=10, color="white",
         markeredgecolor="#1a1a2e", markeredgewidth=1.2, zorder=6)

# heel-strike lines through both panels
for x in t_min:
    for ax in (axF, axT):
        ax.axvline(x, color="white", ls=(0, (5, 4)), lw=1.3, alpha=0.85)

SCRIM = dict(boxstyle="round,pad=0.38", facecolor="#141428", alpha=0.72,
             edgecolor="none")

j = min(3, len(t_min) - 1)
axF.annotate("foot planted (valley of the contour):\nthe swing ends, the foot is still",
             xy=(t_min[j], min_y[j]), xytext=(T1 - 0.08, 720),
             fontsize=10.5, color="white", ha="right", va="center",
             linespacing=1.4, bbox=SCRIM,
             arrowprops=dict(arrowstyle="-|>", color="white", lw=1.4,
                             shrinkB=8))
axF.text(6.58, 760, "solid contour: top of the foot signature (each arch is one foot swing); dashed: its lower edge",
         fontsize=10.5, color="white", ha="left", va="center", bbox=SCRIM)

j_annot = int(np.argmin(np.abs(t_pk - t_min[1])))
axT.annotate("torso near its fastest (peak)",
             xy=(t_pk[j_annot], pk_y[j_annot]), xytext=(t_pk[j_annot], 40),
             fontsize=10.5, color="white", ha="center", va="center",
             bbox=SCRIM, zorder=7,
             arrowprops=dict(arrowstyle="-|>", color="white", lw=1.4,
                             shrinkB=7))
axT.text(6.58, -30, "solid: trunk speed (energy-weighted mean Doppler of the band) at its true value,"
                    " peaking as the weight transfers onto the planted foot; dashed: the band's edges",
         fontsize=10, color="white", ha="left", va="center", style="italic", bbox=SCRIM)

V.title(axF,
        "Coupling between the two channels: each heel strike is a foot valley and a torso peak",
        "healthy control fisc_048, test 2 trial 1, walking back toward the radars after the turn; "
        "vertical dashed lines mark the valleys of the upper foot contour, triangles the trunk-speed peaks")

out = FIG / "guide_coupling.png"
fig.savefig(out)
plt.close(fig)
print("saved", out)

TFM_FIG.mkdir(parents=True, exist_ok=True)
import shutil
shutil.copyfile(out, TFM_FIG / "guide_coupling.png")
print("copied to", TFM_FIG / "guide_coupling.png")

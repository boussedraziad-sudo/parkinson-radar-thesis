"""Shared chart style for every figure in this project.

Palette validated with the dataviz validator (light surface, categorical):
    #2e86c1, #c0392b, #1baf7a, #4a3aa7
    lightness PASS · chroma PASS · CVD PASS (worst adjacent dE 12.8, target >=8)
    normal-vision PASS (worst 18.5, floor 15) · contrast PASS (all >=3:1)

Rules applied everywhere: solid hairline grid one shade off the surface, no top
or right spine, a legend whenever two or more series are present, direct labels
used selectively rather than on every point, and text in ink tokens rather than
in a series colour.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── roles ───────────────────────────────────────────────────────────────────
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
INK_3     = "#8a8985"
GRID      = "#e6e6e3"
RULE      = "#d2d2ce"

CONTROL   = "#2e86c1"   # slot 1  blue
PD        = "#c0392b"   # slot 2  red
GOOD      = "#1baf7a"   # slot 3  aqua   (the fixed / kept state)
ACCENT    = "#4a3aa7"   # slot 4  violet (highlight, never a series)
BAD       = PD          # the broken state reuses the red, never both at once

SEQ       = "magma"     # single-hue sequential, for spectrograms


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
        "figure.dpi": 130,
        "font.size": 10.5,
        "font.family": "sans-serif",
        "text.color": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlepad": 12,
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "xtick.color": INK_3,
        "ytick.color": INK_3,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.facecolor": SURFACE,
        "legend.edgecolor": RULE,
        "legend.fontsize": 9.5,
        "legend.borderpad": 0.6,
        "legend.labelcolor": INK_2,
    })


def title(ax, main: str, sub: str | None = None, sub_size: float = 9.8,
          pad: float | None = None) -> None:
    """Bold title with an optional muted second line, laid out above the axes."""
    if pad is None:
        pad = 20 if sub else 12
    ax.set_title(main, loc="left", pad=pad)
    if sub:
        ax.text(0, 1.035, sub, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=sub_size, color=INK_2, style="italic")


def note(ax, x, y, text, color=None, **kw):
    """Annotation in ink, not in a series colour."""
    kw.setdefault("fontsize", 9.6)
    kw.setdefault("va", "center")
    return ax.text(x, y, text, color=color or INK_2, **kw)


def ring(marker_kw: dict | None = None) -> dict:
    """2px surface ring, for markers that can overlap."""
    d = {"markeredgecolor": SURFACE, "markeredgewidth": 2.0}
    if marker_kw:
        d.update(marker_kw)
    return d

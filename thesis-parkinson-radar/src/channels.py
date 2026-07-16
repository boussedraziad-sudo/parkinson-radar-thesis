"""
Channel arrangement for foot/torso spectrograms.

Default model input is the 2-channel stack [ce_foot, ce_torso]. Ablations
swap to foot-only, torso-only, or a 3-channel variant suited to ImageNet
pretraining.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

ChannelMode = Literal["both", "foot", "torso", "both_3ch", "diff"]


def arrange_channels(foot: np.ndarray, torso: np.ndarray, mode: ChannelMode = "both") -> np.ndarray:
    """
    Return (C, H, W) tensor.

    Modes:
      both     -> 2 channels: [foot, torso]
      foot     -> 1 channel:  [foot]
      torso    -> 1 channel:  [torso]
      both_3ch -> 3 channels: [foot, torso, (foot+torso)/2]   (ImageNet-friendly)
      diff     -> 2 channels: [foot, foot-torso]              (highlight asymmetry)
    """
    if mode == "both":
        return np.stack([foot, torso], axis=0)
    if mode == "foot":
        return foot[np.newaxis]
    if mode == "torso":
        return torso[np.newaxis]
    if mode == "both_3ch":
        return np.stack([foot, torso, 0.5 * (foot + torso)], axis=0)
    if mode == "diff":
        return np.stack([foot, foot - torso], axis=0)
    raise ValueError(f"unknown channel mode {mode!r}")


def expected_n_channels(mode: ChannelMode) -> int:
    return {"both": 2, "foot": 1, "torso": 1, "both_3ch": 3, "diff": 2}[mode]

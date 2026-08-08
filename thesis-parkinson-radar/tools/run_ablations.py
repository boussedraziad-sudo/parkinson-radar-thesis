#!/usr/bin/env python3
"""Ablation battery: the four literature-motivated levers, SmallCNN, full LOSO.

Each ablation changes exactly one thing against the pre-registered headline
configuration (commit b0e7221). Motivations, from the verified literature
review's radar-domain lens:

  A1 per-window z-score  -- standardising each window against itself removes
     per-recording brightness, the channel through which the adaptive contrast
     enhancement (unknown formula, per-trial) can leak recording identity.
     Trade-off: also deletes absolute amplitude (the bradykinesia carrier).
  A2 raw |STFT|          -- bypass the contrast-enhanced arrays entirely. The
     CE step is the prime suspect for the acquisition-batch artifact, since the
     batch marker is a processing-run fingerprint. "Treat raw |stft| as the
     primary input and ce as the ablation" was the review's #1 recommendation.
  A3 turn exclusion      -- drop windows straddling the turn (|pos_frac-0.5|
     <= 0.1). The source publication measures gait only after discarding
     turning/standing/sitting; the group duration difference lives in exactly
     those stages, so this attacks the residual activity-composition confound.
  A4 foot channel only   -- the source publication shows foot nodes stay
     reliable for motorically impaired subjects while torso nodes degrade, and
     Hoshiga et al. report leg motion carrying the discriminative signal.
"""
import dataclasses
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
sys.path.insert(0, str(ROOT))
from src.models import SmallCNN                                        # noqa: E402
from src.preprocessing import PreprocConfig, preprocess_dataset        # noqa: E402
from src.train import TrainConfig, loso_cv, loso_summary, save_run     # noqa: E402

OUT = ROOT / "outputs/runs"
OUT.mkdir(parents=True, exist_ok=True)
BASE = TrainConfig()          # the pre-registered defaults


def ensure_cache(tag: str, **pp_kw) -> Path:
    """Build an alternative preprocessing cache if it does not exist yet."""
    root = ROOT / f"outputs/preprocessed_{tag}"
    manifest = root / "manifest.csv"
    if not manifest.exists():
        print(f"[cache] building {tag} ...", flush=True)
        t0 = time.time()
        preprocess_dataset(PreprocConfig(**pp_kw), out_root=root)
        print(f"[cache] {tag} done in {time.time()-t0:.0f}s", flush=True)
    return root


ABLATIONS = [
    # (name, cache_dir, train-config overrides)
    ("abl_zscore",  ensure_cache("zscore", normalise="log_zscore"), {}),
    ("abl_stft",    ensure_cache("stft", representation="stft_abs", normalise="log"), {}),
    ("abl_noturn",  ROOT / "outputs/preprocessed", {"max_turn_frac": 0.1}),
    ("abl_foot",    ROOT / "outputs/preprocessed", {"channel_mode": "foot"}),
]

for name, cache, over in ABLATIONS:
    if (OUT / f"{name}_summary.json").exists():
        print(f"[skip] {name} already done", flush=True)
        continue
    cfg = dataclasses.replace(BASE, **over)
    man = pd.read_csv(cache / "manifest.csv")
    in_ch = 1 if over.get("channel_mode") == "foot" else 2
    print(f"\n=== {name} | cache={cache.name} | overrides={over} ===", flush=True)
    t0 = time.time()
    folds = loso_cv(lambda: SmallCNN(in_channels=in_ch), man, cache, cfg,
                    checkpoint_dir=OUT / f"ckpt_{name}", verbose=True)
    summ = loso_summary(folds)
    summ["wall_clock_s"] = round(time.time() - t0, 1)
    summ["overrides"] = over
    summ["cache"] = cache.name
    save_run(folds, summ, cfg, OUT, name)
    print(f"--- {name} DONE  subject AUC {summ['subject_auc']:.4f} "
          f"({summ['wall_clock_s']/60:.1f} min) ---", flush=True)

print("\nABLATION BATTERY COMPLETE", flush=True)

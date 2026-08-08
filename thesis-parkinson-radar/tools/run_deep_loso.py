#!/usr/bin/env python3
"""Reporting LOSO runs for the deep models.

Pre-registered: the TrainConfig below is committed before this script is run and
is not touched afterwards. Each run writes per-fold rows, a summary, the full
per-epoch history and one checkpoint per fold.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
sys.path.insert(0, str(ROOT))
from src.models import SmallCNN, count_parameters, resnet18_finetune   # noqa: E402
from src.train import TrainConfig, loso_cv, loso_summary, save_run     # noqa: E402

CACHE = ROOT / "outputs/preprocessed"
OUT = ROOT / "outputs/runs"
OUT.mkdir(parents=True, exist_ok=True)
manifest = pd.read_csv(CACHE / "manifest.csv")

CFG = TrainConfig(
    epochs=20, lr=1e-3, weight_decay=1e-4, batch_size=32, patience=5,
    inner_val_subjects=8, channel_mode="both",
    augment_train=True, aug_time_flip=True, aug_noise_std=0.02,
    class_balance=True, window_weighting=True, max_turn_frac=None,
    seed=1337,
)

RUNS = [
    ("smallcnn", lambda: SmallCNN(in_channels=2)),
    ("resnet18_fc", lambda: resnet18_finetune(in_channels=2, freeze_until="layer4")),
]

for name, factory in RUNS:
    n = count_parameters(factory())
    print(f"\n=== {name} | trainable {n['trainable']:,} / {n['total']:,} "
          f"| device {CFG.device} ===", flush=True)
    t0 = time.time()
    folds = loso_cv(factory, manifest, CACHE, CFG,
                    checkpoint_dir=OUT / f"ckpt_{name}", verbose=True)
    summ = loso_summary(folds)
    summ["wall_clock_s"] = round(time.time() - t0, 1)
    summ["trainable_params"] = n["trainable"]
    save_run(folds, summ, CFG, OUT, name)
    print(f"--- {name} DONE  subject AUC {summ['subject_auc']:.4f}  "
          f"balanced acc {summ['balanced_accuracy']:.4f}  "
          f"({summ['wall_clock_s']/60:.1f} min) ---", flush=True)
    print(json.dumps(summ, indent=1), flush=True)

print("\nALL RUNS COMPLETE", flush=True)

#!/usr/bin/env python3
"""v2 reporting runs: both models under the corrected training protocol.

v2 fixes, all found by the adversarial code review and none informed by test
results: frozen ResNet blocks pinned to eval so BatchNorm cannot drift;
early stopping on SUBJECT-level inner-val AUC; antialiased resize; Doppler-axis
flip replacing the physically wrong time flip; meaningful noise + masking
augmentation; class weights on effective mass; fold seeds keyed to subject ids;
background-robust turn fractions. Cache rebuilt as outputs/preprocessed_v2 so
v1 stays reproducible.
"""
import json, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
sys.path.insert(0, str(ROOT))
from src.models import SmallCNN, count_parameters, resnet18_finetune
from src.preprocessing import PreprocConfig, preprocess_dataset
from src.train import TrainConfig, loso_cv, loso_summary, save_run

CACHE = ROOT / "outputs/preprocessed_v2"
OUT = ROOT / "outputs/runs"
if not (CACHE / "manifest.csv").exists():
    print("[cache] building v2 (antialias + robust turn fraction)...", flush=True)
    preprocess_dataset(PreprocConfig(), out_root=CACHE)

manifest = pd.read_csv(CACHE / "manifest.csv")
CFG = TrainConfig()          # v2 defaults, committed before this run
for name, factory in [
    ("smallcnn_v2", lambda: SmallCNN(in_channels=2)),
    ("resnet18_fc_v2", lambda: resnet18_finetune(in_channels=2, freeze_until="layer4")),
]:
    n = count_parameters(factory())
    print(f"\n=== {name} | trainable {n['trainable']:,} | device {CFG.device} ===", flush=True)
    t0 = time.time()
    folds = loso_cv(factory, manifest, CACHE, CFG,
                    checkpoint_dir=OUT / f"ckpt_{name}", verbose=True)
    summ = loso_summary(folds)
    summ["wall_clock_s"] = round(time.time() - t0, 1)
    summ["trainable_params"] = n["trainable"]
    save_run(folds, summ, CFG, OUT, name)
    print(f"--- {name} DONE  subject AUC {summ['subject_auc']:.4f}  "
          f"({summ['wall_clock_s']/60:.1f} min) ---", flush=True)
print("\nV2 RUNS COMPLETE", flush=True)

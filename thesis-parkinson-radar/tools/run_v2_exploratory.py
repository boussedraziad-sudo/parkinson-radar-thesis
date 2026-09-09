#!/usr/bin/env python3
"""Exploratory v2 runs: the two levers that survived the battery, under the
corrected protocol. Labelled exploratory; configuration count is reported."""
import dataclasses, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
sys.path.insert(0, str(ROOT))
from src.models import SmallCNN
from src.train import TrainConfig, loso_cv, loso_summary, save_run

CACHE = ROOT / "outputs/preprocessed_v2"
OUT = ROOT / "outputs/runs"
man = pd.read_csv(CACHE / "manifest.csv")

for name, over, in_ch in [
    ("abl_foot_v2", {"channel_mode": "foot"}, 1),
    # Turn exclusion is applied by PRE-FILTERING the manifest rather than inside
    # the Dataset: the inner split, the class guard and the window weights then
    # all see the filtered reality, and a fold cannot end up with a single-class
    # validation set. Threshold 0.05: at 0.1 the sharpened v2 labels remove all
    # windows of 24/58 subjects, which is not an interpretable experiment.
    ("abl_noturn_v2", {"turn_prefilter": 0.05}, 2),
]:
    if (OUT / f"{name}_summary.json").exists():
        print(f"[skip] {name}", flush=True); continue
    run_man = man
    thr = over.pop("turn_prefilter", None)
    if thr is not None:
        keep = (man.pos_frac_foot - 0.5).abs() > thr
        run_man = man[keep].reset_index(drop=True)
        lost = sorted(set(man.subject_id) - set(run_man.subject_id))
        print(f"[prefilter] |pos_frac-0.5|>{thr}: {len(run_man)}/{len(man)} windows, "
              f"{len(lost)} subjects excluded: {lost}", flush=True)
    cfg = dataclasses.replace(TrainConfig(), **over)
    print(f"\n=== {name} | overrides={over} thr={thr} ===", flush=True)
    t0 = time.time()
    folds = loso_cv(lambda: SmallCNN(in_channels=in_ch), run_man, CACHE, cfg,
                    checkpoint_dir=OUT / f"ckpt_{name}", verbose=True)
    summ = loso_summary(folds)
    summ["wall_clock_s"] = round(time.time() - t0, 1)
    summ["overrides"] = over
    save_run(folds, summ, cfg, OUT, name)
    print(f"--- {name} DONE  subject AUC {summ['subject_auc']:.4f} "
          f"({summ['wall_clock_s']/60:.1f} min) ---", flush=True)
print("\nEXPLORATORY V2 COMPLETE", flush=True)

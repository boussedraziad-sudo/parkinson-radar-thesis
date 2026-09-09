#!/usr/bin/env python3
"""Full LOSO on the background-suppressed cache, v2 protocol code as-is.

Pre-registered in CONFIG.md. src/train.py and src/models.py are used
unmodified; the only difference from the v2 reporting runs is the cache.

Usage:
  run_loso.py smoke   -> 2 folds per model, pipeline check, results discarded
  run_loso.py full    -> full 58-fold LOSO, both models, summaries written
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import roc_auc_score

from src.models import SmallCNN, count_parameters, resnet18_finetune
from src.train import TrainConfig, loso_cv, loso_summary, save_run

CACHE = HERE / "cache"
OUT = HERE / "outputs"

MODELS = [
    ("smallcnn_bg", lambda: SmallCNN(in_channels=2)),
    ("resnet18_fc_bg", lambda: resnet18_finetune(in_channels=2, freeze_until="layer4")),
]

V2_REFERENCE = {
    "smallcnn_v2": {"pooled": 0.6158, "batch_A": 0.4208, "batch_B": 0.3481, "corr_nwin": 0.047},
    "resnet18_fc_v2": {"pooled": 0.6545, "batch_A": 0.4583, "batch_B": 0.5407, "corr_nwin": 0.081},
    "batch label alone": {"pooled": 0.664},
}


def batch_metrics(folds: pd.DataFrame) -> dict:
    """Pooled/within-batch AUC, bootstrap CI, corr_nwin. Definitions per CONFIG.md."""
    batch = pd.read_csv(ROOT / "outputs/metrics/acquisition_batch.csv")
    sub_batch = batch.groupby("subject_id")["batch"].first()
    f = folds.dropna(subset=["subject_prob"]).copy()
    f["batch"] = f["subject_id"].map(sub_batch)
    y = f["subject_label"].to_numpy()
    p = f["subject_prob"].to_numpy()

    out = {"pooled": float(roc_auc_score(y, p)), "n": int(len(f))}

    rng = np.random.default_rng(1337)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(f), len(f))
        if len(set(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], p[idx]))
    out["ci"] = [round(float(np.percentile(boots, 2.5)), 3),
                 round(float(np.percentile(boots, 97.5)), 3)]

    for b in ("A", "B"):
        g = f[f["batch"] == b]
        out[f"batch_{b}"] = (float(roc_auc_score(g["subject_label"], g["subject_prob"]))
                             if g["subject_label"].nunique() == 2 else float("nan"))
        out[f"n_batch_{b}"] = int(len(g))

    out["corr_nwin"] = float(np.corrcoef(p, f["n_test_windows"].to_numpy(float))[0, 1])
    return out


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    manifest = pd.read_csv(CACHE / "manifest.csv")
    cfg = TrainConfig()  # v2 defaults, pre-registered
    OUT.mkdir(parents=True, exist_ok=True)

    subjects = None
    if mode == "smoke":
        subjects = sorted(manifest["subject_id"].unique())[:2]
        print(f"[smoke] 2 folds per model: {subjects}", flush=True)

    summary_all: dict = {"reference_v2": V2_REFERENCE}
    for name, factory in MODELS:
        n = count_parameters(factory())
        print(f"\n=== {name} ({mode}) | trainable {n['trainable']:,} | device {cfg.device} ===",
              flush=True)
        t0 = time.time()
        folds = loso_cv(factory, manifest, CACHE, cfg,
                        checkpoint_dir=None if mode == "smoke" else OUT / f"ckpt_{name}",
                        subjects=subjects, verbose=True)
        elapsed = time.time() - t0
        if mode == "smoke":
            print(f"[smoke] {name} ok in {elapsed:.0f}s; results discarded", flush=True)
            continue

        summ = loso_summary(folds)
        summ["wall_clock_s"] = round(elapsed, 1)
        summ["trainable_params"] = n["trainable"]
        summ.update(batch_metrics(folds))
        save_run(folds, summ, cfg, OUT, name)
        summary_all[name] = {k: summ[k] for k in
                             ("pooled", "ci", "n", "batch_A", "batch_B",
                              "n_batch_A", "n_batch_B", "corr_nwin",
                              "balanced_accuracy", "wall_clock_s")}
        print(f"--- {name} DONE  pooled AUC {summ['pooled']:.4f}  "
              f"A {summ['batch_A']:.4f}  B {summ['batch_B']:.4f}  "
              f"corr_nwin {summ['corr_nwin']:.3f}  ({elapsed/60:.1f} min) ---", flush=True)

    if mode != "smoke":
        (OUT / "bg_suppressed_summary.json").write_text(json.dumps(summary_all, indent=2))
        print(f"\nSummary: {OUT / 'bg_suppressed_summary.json'}", flush=True)
    print("\nBG-SUPPRESSED RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()

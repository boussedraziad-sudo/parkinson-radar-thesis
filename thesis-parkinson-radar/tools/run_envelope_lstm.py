#!/usr/bin/env python3
"""Envelope-LSTM baseline: extraction, then full nested LOSO. CPU by design.

The representation follows Hayashi/Saho, the literature's most trustworthy
model, adapted to this dataset: three velocity curves per 3.0 s window,

    ch0  foot upper envelope   the fastest moving part of the body (peak swing)
    ch1  foot speed centroid   the energy-weighted average speed, foot node
    ch2  torso speed centroid  the same, torso node

computed on the folded speed axis |Doppler| so the out-and-back walk does not
flip the curves' sign, resampled to 128 time steps. Windows are identical to
the image cache (3.0 s / 1.5 s), so results are directly comparable.

Curves are cached in raw Hz; standardisation happens per fold from training
subjects only, via per-window channel statistics stored in the manifest (the
same law-of-total-variance trick as the image cache). Runs on CPU so it can
train while the GPU ablation battery is busy.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

ROOT = Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
sys.path.insert(0, str(ROOT))
from src.data_loader import iter_trials, load_trial                     # noqa: E402
from src.models import EnvelopeLSTM, count_parameters                   # noqa: E402
from src.preprocessing import add_window_weights, window_indices        # noqa: E402
from src.train import (TrainConfig, loso_summary, make_inner_split,     # noqa: E402
                       save_run, set_seed, subject_labels, train_one_fold)

CACHE = ROOT / "outputs/preprocessed_env"
OUT = ROOT / "outputs/runs"
T_OUT = 128
N_CH = 3
UPPER_THR = 0.25          # envelope: highest |Doppler| bin above this fraction of the column max


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def window_envelopes(foot: np.ndarray, torso: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """(3, T_OUT) velocity curves in Hz for one window."""
    ef, et = np.maximum(foot, 0.0), np.maximum(torso, 0.0)

    def centroid(e):
        tot = e.sum(axis=0)
        return np.divide((speed[:, None] * e).sum(axis=0), tot,
                         out=np.zeros(e.shape[1]), where=tot > 0)

    colmax = ef.max(axis=0)
    mask = ef >= (UPPER_THR * np.maximum(colmax, 1e-12))[None, :]
    upper = np.where(mask, speed[:, None], 0.0).max(axis=0)

    curves = np.stack([upper, centroid(ef), centroid(et)])
    x_old = np.linspace(0.0, 1.0, curves.shape[1])
    x_new = np.linspace(0.0, 1.0, T_OUT)
    return np.stack([np.interp(x_new, x_old, c) for c in curves]).astype(np.float32)


def extract() -> pd.DataFrame:
    rows = []
    for tp in tqdm(list(iter_trials()), desc="envelopes"):
        tr = load_trial(tp.path, representation="ce")
        t = np.asarray(tr["t_axis"], float).ravel()
        speed = np.abs(np.asarray(tr["doppler"], float).ravel())
        spans = window_indices(t, 3.0, 1.5)
        for w_idx, (s, e) in enumerate(spans):
            arr = window_envelopes(tr["ce_foot"][:, s:e], tr["ce_torso"][:, s:e], speed)
            out_dir = CACHE / tp.group
            out_dir.mkdir(parents=True, exist_ok=True)
            p = out_dir / f"{tp.subject_id}_{tp.test}_{tp.trial}_w{w_idx:03d}.npy"
            np.save(p, arr)
            row = {"subject_id": tp.subject_id, "group": tp.group,
                   "label": 1 if tp.group == "pd" else 0,
                   "test": tp.test, "trial": tp.trial, "window_idx": w_idx,
                   "n_windows_trial": len(spans),
                   "npy_path": str(p.relative_to(CACHE))}
            for c in range(N_CH):
                row[f"ch{c}_mean"] = float(arr[c].mean())
                row[f"ch{c}_std"] = float(arr[c].std())
            rows.append(row)
    man = pd.DataFrame(rows)
    man.to_csv(CACHE / "manifest.csv", index=False)
    print(f"envelope windows: {len(man)}")
    return man


def env_fold_stats(manifest: pd.DataFrame, subjects) -> np.ndarray:
    """(N_CH, 2) per-channel (mean, std) over the training subjects' windows."""
    sub = manifest[manifest["subject_id"].isin(list(subjects))]
    out = np.zeros((N_CH, 2))
    for c in range(N_CH):
        m = sub[f"ch{c}_mean"].to_numpy(float)
        s = sub[f"ch{c}_std"].to_numpy(float)
        out[c] = (m.mean(), np.sqrt(max((s ** 2).mean() + m.var(), 1e-12)))
    return out


class EnvelopeDataset(Dataset):
    """(3, 128) curves, standardised with training-fold statistics."""

    def __init__(self, manifest, root, subjects=None, augment=False,
                 norm_stats=None, rng_seed=None, aug_noise_std=0.05):
        df = manifest
        if subjects is not None:
            df = df[df["subject_id"].isin(list(subjects))]
        self.manifest = df.reset_index(drop=True)
        self.root = Path(root)
        self.augment = augment
        self.norm_stats = norm_stats
        self.aug_noise_std = aug_noise_std
        self._rng = np.random.default_rng(rng_seed)
        self._has_w = "sample_weight" in self.manifest.columns

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        arr = np.load(self.root / row["npy_path"]).astype(np.float32)
        if self.norm_stats is not None:
            for c in range(arr.shape[0]):
                mu, sd = self.norm_stats[c]
                arr[c] = (arr[c] - mu) / max(float(sd), 1e-6)
        if self.augment:
            if self._rng.random() < 0.5:
                arr = arr[:, ::-1].copy()
            if self._rng.random() < 0.7:
                arr = arr + self._rng.normal(0, self.aug_noise_std, arr.shape).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(arr)), int(row["label"]), \
            float(row["sample_weight"]) if self._has_w else 1.0


# ---------------------------------------------------------------------------
# Nested LOSO (mirrors src.train.loso_cv with the envelope Dataset)
# ---------------------------------------------------------------------------


def main() -> None:
    if not (CACHE / "manifest.csv").exists():
        t0 = time.time()
        extract()
        print(f"extraction {time.time()-t0:.0f}s", flush=True)
    man = add_window_weights(pd.read_csv(CACHE / "manifest.csv"))

    cfg = TrainConfig(device="cpu", num_workers=0)      # GPU is busy with the battery
    labels = subject_labels(man)
    subjects = sorted(man["subject_id"].unique())
    print(f"EnvelopeLSTM {count_parameters(EnvelopeLSTM())} | device cpu", flush=True)

    records = []
    t0 = time.time()
    for i, held in enumerate(subjects):
        set_seed(cfg.seed + i)
        pool = [s for s in subjects if s != held]
        itr, iva = make_inner_split(pool, labels, cfg.inner_val_subjects, cfg.seed + i)
        stats = env_fold_stats(man, itr)
        mk = lambda ss, aug: EnvelopeDataset(man, CACHE, subjects=ss, augment=aug,
                                             norm_stats=stats, rng_seed=cfg.seed + i)
        fold = train_one_fold(EnvelopeLSTM(), mk(itr, True), mk(iva, False),
                              mk([held], False), cfg,
                              checkpoint_path=OUT / "ckpt_envlstm" / f"fold_{held}.pt")
        probs = fold.get("test_probs", [])
        records.append({
            "subject_id": held, "subject_label": int(labels.loc[held]),
            "subject_prob": float(np.mean(probs)) if probs else float("nan"),
            "best_val_window_auc": fold["best_val_window_auc"],
            "best_epoch": fold["best_epoch"], "epochs_run": fold["epochs_run"],
        })
        r = records[-1]
        print(f"  [{i+1:02d}/58] {held}  label={r['subject_label']}  "
              f"prob={r['subject_prob']:.3f}  val_auc={r['best_val_window_auc']:.3f}  "
              f"ep={r['best_epoch']}/{r['epochs_run']}", flush=True)

    folds = pd.DataFrame(records)
    summ = loso_summary(folds)
    summ["wall_clock_s"] = round(time.time() - t0, 1)
    summ["trainable_params"] = count_parameters(EnvelopeLSTM())["trainable"]
    save_run(folds, summ, cfg, OUT, "envlstm")
    print(f"--- envlstm DONE  subject AUC {summ['subject_auc']:.4f}  "
          f"({summ['wall_clock_s']/60:.1f} min) ---", flush=True)


if __name__ == "__main__":
    main()

"""
Nested LOSO-CV training loop.

The protocol, and the reason it is shaped this way:

    58 subjects
    ├──  1 → TEST       scored exactly once, informs no decision
    ├── ~8 → INNER VAL  stratified, so BOTH classes are present
    └── ~49→ TRAIN

Two properties follow, and both were broken in the previous version:

  * Early stopping needs a validation set containing both classes. A single
    held-out subject has one label, so a validation AUC computed on it is
    undefined; the old code silently substituted 0.5 every epoch, which meant
    the "best" epoch was always epoch 0 and every model trained for one epoch
    while the run still reported success. `_require_both_classes` now makes
    that failure loud.
  * Selecting the epoch on the held-out subject is leakage, and it is the exact
    practice Chapter 2 criticises. The held-out subject is now touched once,
    after training has finished.

Subject-level decisions are the mean of the window probabilities for that
subject, which is also what neutralises window count at scoring time.
"""

from __future__ import annotations

import dataclasses
import json
import random
import time
import zlib
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader

from .dataset import WindowDataset
from .preprocessing import add_window_weights, fold_norm_stats


# ---------------------------------------------------------------------------
# Configuration and reproducibility
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TrainConfig:
    """Pre-registered training configuration. Commit this before the reporting run."""

    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    patience: int = 5
    inner_val_subjects: int = 8
    channel_mode: str = "both"
    augment_train: bool = True
    aug_doppler_flip: bool = True     # Doppler-axis flip (was: time flip, wrong axis)
    aug_noise_std: float = 0.10       # 0.02 on unit-variance data was a no-op
    aug_mask_frac: float = 0.15       # SpecAugment-style time/Doppler masking
    class_balance: bool = True       # counter the 57/43 label imbalance
    window_weighting: bool = True    # counter the window-count confound
    max_turn_frac: float | None = None   # None keeps every window
    num_workers: int = 0
    seed: int = 1337
    device: str = dataclasses.field(default_factory=lambda: default_device())

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def default_device() -> str:
    """Apple GPU when present. Measured 2.75x faster than CPU on this workload."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _require_both_classes(labels: Sequence[int], where: str) -> None:
    """Guard against the degenerate-validation bug ever returning silently."""
    if len(set(int(x) for x in labels)) < 2:
        raise ValueError(
            f"{where} contains a single class. Validation AUC would be undefined "
            "and early stopping would select epoch 0. Increase "
            "TrainConfig.inner_val_subjects or check the split."
        )


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def subject_labels(manifest: pd.DataFrame) -> pd.Series:
    """One label per subject (labels are constant within a subject)."""
    return manifest.groupby("subject_id")["label"].first()


def make_inner_split(
    train_subjects: Sequence[str],
    labels: pd.Series,
    n_val: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Stratified subject-level split of the training pool into (train, inner-val)."""
    subs = np.asarray(sorted(train_subjects))
    y = labels.loc[subs].to_numpy()
    n_val = int(min(max(n_val, 2), len(subs) - 2))
    # Guarantee both classes land in the validation pool.
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n_val, random_state=seed)
    tr_idx, va_idx = next(sss.split(subs.reshape(-1, 1), y))
    inner_train, inner_val = subs[tr_idx].tolist(), subs[va_idx].tolist()
    _require_both_classes(labels.loc[inner_val].tolist(), "inner validation split")
    return inner_train, inner_val


def grouped_cv(
    manifest: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 1337,
) -> list[tuple[list[str], list[str]]]:
    """Subject-grouped stratified K-fold, returned as lists of subject ids.

    Used for the cheap smoke test: it exercises the whole pipeline in a few
    minutes rather than the ~18 minutes a full 58-fold LOSO costs, while still
    never letting one subject appear on both sides of a split.
    """
    labels = subject_labels(manifest)
    subs = labels.index.to_numpy()
    y = labels.to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out: list[tuple[list[str], list[str]]] = []
    for tr, te in splitter.split(subs.reshape(-1, 1), y, groups=subs):
        out.append((subs[tr].tolist(), subs[te].tolist()))
    return out


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------


def _eval_loader(model: nn.Module, loader: DataLoader, device: str) -> tuple[list[float], list[int]]:
    model.eval()
    probs: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for xb, yb, *_ in loader:
            p = torch.softmax(model(xb.to(device)), dim=1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(yb.numpy().tolist())
    return probs, labels


def subject_level_auc(ds: WindowDataset, probs: Sequence[float]) -> float:
    """AUC over per-subject MEAN window probabilities.

    Early stopping previously compared epochs on the pooled window-level AUC,
    which weights each subject by their window count. PD subjects contribute
    ~14 % more windows, so checkpoint selection was quietly rewarding the same
    duration channel the training weights neutralise, and it was misaligned
    with the reported metric. Scoring the validation set exactly the way test
    subjects are scored fixes both at once. Uses inner-validation data only.
    """
    df = ds.manifest[["subject_id", "label"]].copy()
    df["prob"] = list(probs)
    per = df.groupby("subject_id").agg(label=("label", "first"), prob=("prob", "mean"))
    return float(roc_auc_score(per["label"], per["prob"]))


def _loader(ds: WindowDataset, cfg: TrainConfig, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=shuffle,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        drop_last=False,
    )


def _class_weights(ds: WindowDataset, device: str) -> torch.Tensor | None:
    # Balance the classes by their EFFECTIVE (sample-weighted) mass, not the raw
    # window counts: the inverse-count weights already change how much each
    # window contributes, and computing class balance on raw counts would tilt
    # the loss against whichever class has longer recordings.
    y = ds.manifest["label"].to_numpy()
    sw = ds.sample_weights
    m_pos, m_neg = float(sw[y == 1].sum()), float(sw[y == 0].sum())
    if m_pos <= 0 or m_neg <= 0:
        return None
    total = m_pos + m_neg
    w = torch.tensor([total / (2 * m_neg), total / (2 * m_pos)], dtype=torch.float32)
    return w.to(device)


def train_one_fold(
    model: nn.Module,
    train_ds: WindowDataset,
    val_ds: WindowDataset,
    test_ds: WindowDataset | None,
    cfg: TrainConfig,
    *,
    checkpoint_path: Path | None = None,
    verbose: bool = False,
) -> dict:
    """Train on `train_ds`, early-stop on `val_ds`, score `test_ds` exactly once."""
    device = cfg.device
    model = model.to(device)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    cw = _class_weights(train_ds, device) if cfg.class_balance else None
    # reduction="none" so the per-sample window weight can be applied on top of
    # the per-class weight; the two corrections are independent and multiply.
    crit = nn.CrossEntropyLoss(weight=cw, reduction="none")

    tr = _loader(train_ds, cfg, shuffle=True)
    va = _loader(val_ds, cfg, shuffle=False)

    # Fail loudly rather than silently selecting epoch 0 (the old bug).
    _require_both_classes(val_ds.manifest["label"].tolist(), "validation Dataset")

    best_auc, best_epoch = -1.0, -1
    best_state: dict | None = None
    patience_left = cfg.patience
    history: list[dict] = []

    for epoch in range(cfg.epochs):
        model.train()
        running, seen = 0.0, 0
        for xb, yb, wb in tr:
            xb, yb = xb.to(device), yb.to(device)
            wb = wb.to(device, dtype=torch.float32)
            opt.zero_grad()
            per_sample = crit(model(xb), yb)
            loss = (per_sample * wb).sum() / wb.sum().clamp_min(1e-8)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
            seen += xb.size(0)
        train_loss = running / max(seen, 1)

        probs, _ = _eval_loader(model, va, device)
        auc = subject_level_auc(val_ds, probs)   # both classes guaranteed present
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": auc})
        if verbose:
            print(f"      epoch {epoch:02d}  loss={train_loss:.4f}  val_auc={auc:.3f}")

        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "best_epoch": best_epoch,
                    "best_val_auc": best_auc, "config": cfg.to_dict()}, checkpoint_path)

    out = {
        "best_val_window_auc": float(best_auc),
        "best_epoch": int(best_epoch),
        "epochs_run": len(history),
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "history": history,
    }
    if test_ds is not None and len(test_ds) > 0:
        te = _loader(test_ds, cfg, shuffle=False)
        out["test_probs"], out["test_labels"] = _eval_loader(model, te, device)
        out["n_test_windows"] = len(test_ds)
    return out


# ---------------------------------------------------------------------------
# Full nested LOSO
# ---------------------------------------------------------------------------


def loso_cv(
    model_factory: Callable[[], nn.Module],
    manifest: pd.DataFrame,
    cache_root: Path,
    cfg: TrainConfig | None = None,
    *,
    checkpoint_dir: Path | None = None,
    subjects: Sequence[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Nested LOSO. One fold per subject; the held-out subject is scored once."""
    cfg = cfg or TrainConfig()
    manifest = add_window_weights(manifest) if cfg.window_weighting else manifest
    labels = subject_labels(manifest)
    all_subjects = sorted(manifest["subject_id"].unique())
    todo = list(subjects) if subjects is not None else all_subjects

    records: list[dict] = []
    for i, held_out in enumerate(todo):
        # Key the fold seed to the SUBJECT, not the loop index, so re-running a
        # subset of subjects reproduces the identical folds.
        fold_seed = cfg.seed + zlib.crc32(held_out.encode()) % 100_000
        set_seed(fold_seed)
        pool = [s for s in all_subjects if s != held_out]
        inner_train, inner_val = make_inner_split(
            pool, labels, cfg.inner_val_subjects, seed=fold_seed)

        # Normalisation statistics come from the training subjects ONLY.
        stats = fold_norm_stats(manifest, inner_train)
        common = dict(channel_mode=cfg.channel_mode, norm_stats=stats,
                      max_turn_frac=cfg.max_turn_frac)

        train_ds = WindowDataset(manifest, cache_root, subjects=inner_train,
                                 augment=cfg.augment_train,
                                 aug_doppler_flip=cfg.aug_doppler_flip,
                                 aug_noise_std=cfg.aug_noise_std,
                                 aug_mask_frac=cfg.aug_mask_frac,
                                 rng_seed=fold_seed, **common)
        val_ds = WindowDataset(manifest, cache_root, subjects=inner_val,
                               augment=False, **common)
        test_ds = WindowDataset(manifest, cache_root, subjects=[held_out],
                                augment=False, **common)

        ckpt = None if checkpoint_dir is None else Path(checkpoint_dir) / f"fold_{held_out}.pt"
        t0 = time.time()
        fold = train_one_fold(model_factory(), train_ds, val_ds, test_ds, cfg,
                              checkpoint_path=ckpt)

        probs = fold.get("test_probs", [])
        subj_prob = float(np.mean(probs)) if probs else float("nan")
        subj_label = int(labels.loc[held_out])
        records.append({
            "subject_id": held_out,
            "subject_label": subj_label,
            "subject_prob": subj_prob,
            "n_train_subjects": len(inner_train),
            "n_val_subjects": len(inner_val),
            "best_val_window_auc": fold["best_val_window_auc"],
            "best_epoch": fold["best_epoch"],
            "epochs_run": fold["epochs_run"],
            "n_train_windows": fold["n_train_windows"],
            "n_test_windows": fold.get("n_test_windows", 0),
            "elapsed_s": time.time() - t0,
        })
        if verbose:
            r = records[-1]
            print(f"  [{i+1:02d}/{len(todo)}] {held_out}  label={subj_label}  "
                  f"prob={subj_prob:.3f}  val_auc={r['best_val_window_auc']:.3f}  "
                  f"ep={r['best_epoch']}/{r['epochs_run']}  ({r['elapsed_s']:.0f}s)")

    return pd.DataFrame(records)


def loso_summary(folds: pd.DataFrame, threshold: float | None = None) -> dict:
    """Aggregate per-fold subject-level predictions into overall metrics.

    The decision threshold defaults to the prevalence of the positive class
    rather than 0.5, because the classifier is trained on an imbalanced corpus
    and an uncalibrated 0.5 cut systematically under-predicts the minority
    class. AUC, the headline metric, is unaffected by this choice.
    """
    f = folds.dropna(subset=["subject_prob"])
    if f["subject_label"].nunique() < 2:
        return {"subject_auc": float("nan"), "n": int(len(f))}
    y = f["subject_label"].to_numpy()
    p = f["subject_prob"].to_numpy()
    thr = float(y.mean()) if threshold is None else float(threshold)
    pred = (p >= thr).astype(int)
    return {
        "subject_auc": float(roc_auc_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "subject_f1": float(f1_score(y, pred, zero_division=0)),
        "threshold": thr,
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "n": int(len(f)),
        "mean_epochs_run": float(f["epochs_run"].mean()),
        "mean_best_epoch": float(f["best_epoch"].mean()),
    }


def save_run(folds: pd.DataFrame, summary: dict, cfg: TrainConfig,
             out_dir: Path, name: str) -> None:
    """Persist a run so a result can always be traced back to its configuration."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out_dir / f"{name}_folds.csv", index=False)
    (out_dir / f"{name}_summary.json").write_text(
        json.dumps({"summary": summary, "config": cfg.to_dict()}, indent=2))

"""
LOSO-CV training loop.

`train_one_fold` trains a model on a train Dataset and reports validation
metrics on a val Dataset. `loso_cv` orchestrates one fold per subject.

Subject-level decisions are obtained by averaging window probabilities for
the held-out subject (the standard aggregation for window-based gait CNNs).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from .dataset import WindowDataset


def _eval_loader(model: nn.Module, loader: DataLoader, device: str) -> tuple[list[float], list[int]]:
    model.eval()
    probs: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            p = torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(yb.numpy().tolist())
    return probs, labels


def train_one_fold(
    model: nn.Module,
    train_ds: WindowDataset,
    val_ds: WindowDataset,
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    patience: int = 5,
    device: str = "cpu",
    verbose: bool = False,
) -> dict:
    """Train one fold; early-stop on val AUC; restore best weights."""
    model = model.to(device)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    crit = nn.CrossEntropyLoss()
    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    va = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    best_auc = 0.0
    best_state: dict | None = None
    patience_left = patience
    history: list[dict] = []

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        train_loss = running / max(len(train_ds), 1)

        probs, labels = _eval_loader(model, va, device)
        auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.5
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": auc})
        if verbose:
            print(f"    epoch {epoch:02d}  train_loss={train_loss:.4f}  val_auc={auc:.3f}")

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_probs, val_labels = _eval_loader(model, va, device)

    return {
        "best_val_window_auc": best_auc,
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "val_probs": val_probs,
        "val_labels": val_labels,
        "history": history,
    }


def loso_cv(
    model_factory: Callable[[], nn.Module],
    manifest: pd.DataFrame,
    cache_root: Path,
    *,
    channel_mode: str = "both",
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cpu",
    augment_train: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full LOSO-CV. One fold per unique subject.

    model_factory must return a fresh, unfit model each call.
    Returns one row per subject with train/val window counts and subject-level prediction.
    """
    subjects = sorted(manifest["subject_id"].unique())
    records: list[dict] = []

    for held_out in subjects:
        train_subj = [s for s in subjects if s != held_out]
        train_ds = WindowDataset(manifest, cache_root, subjects=train_subj,
                                 augment=augment_train, channel_mode=channel_mode)
        val_ds = WindowDataset(manifest, cache_root, subjects=[held_out],
                               augment=False, channel_mode=channel_mode)

        t0 = time.time()
        fold = train_one_fold(
            model_factory(), train_ds, val_ds,
            epochs=epochs, lr=lr, batch_size=batch_size, device=device,
        )
        subj_prob = float(np.mean(fold["val_probs"])) if fold["val_probs"] else 0.0
        subj_label = int(fold["val_labels"][0]) if fold["val_labels"] else -1
        records.append({
            "subject_id": held_out,
            "subject_label": subj_label,
            "subject_prob": subj_prob,
            "subject_pred": int(subj_prob >= 0.5),
            "best_window_auc": fold["best_val_window_auc"],
            "n_train_windows": fold["n_train_windows"],
            "n_val_windows": fold["n_val_windows"],
            "elapsed_s": time.time() - t0,
        })
        if verbose:
            print(f"  LOSO[{held_out}]  label={subj_label}  prob={subj_prob:.3f}  "
                  f"window_auc={fold['best_val_window_auc']:.3f}  ({records[-1]['elapsed_s']:.0f}s)")

    return pd.DataFrame(records)


def loso_summary(folds: pd.DataFrame) -> dict:
    """Aggregate per-fold subject-level predictions into overall metrics."""
    if folds["subject_label"].nunique() < 2:
        return {"subject_auc": 0.0, "subject_f1": 0.0, "confusion_matrix": [], "n": len(folds)}
    auc = roc_auc_score(folds["subject_label"], folds["subject_prob"])
    f1 = f1_score(folds["subject_label"], folds["subject_pred"])
    cm = confusion_matrix(folds["subject_label"], folds["subject_pred"]).tolist()
    return {"subject_auc": float(auc), "subject_f1": float(f1),
            "confusion_matrix": cm, "n": int(len(folds))}

#!/usr/bin/env python3
"""AlexNet on this dataset, both ways: the literature's architecture under
our protocol and under theirs.

AlexNet is the architecture behind the field's cautionary 97.8 % result
(Hayashi et al. 2021, spectrogram CNN). Running it here answers, with the
field's own model, whether the modest numbers of the thesis come from the
dataset or from the methodology:

  Part 1  AlexNet under leave-one-subject-out (our protocol)
  Part 2  AlexNet under a recording-level random 70/15/15 split, 3 seeds
          (the literature's hold-out)

Adaptations mirror the thesis's ResNet treatment: ImageNet-pretrained, the
first convolution's RGB filters averaged and tiled to 2 radar channels, the
final layer replaced for 2 classes. The convolutional features are frozen and
the full classifier block trains (54.5 M parameters), with the learning rate
of the source literature (1e-4). Same v2 window cache, same augmentation.
Random-split runs mimic the literature: window-level early stopping, no
window weighting.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torchvision import models as tvm

ROOT = Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
sys.path.insert(0, str(ROOT))
import src.train as T                                    # noqa: E402
from src.dataset import WindowDataset                    # noqa: E402
from src.train import TrainConfig, loso_cv, loso_summary, save_run  # noqa: E402

CACHE = ROOT / "outputs/preprocessed_v2"
OUT = ROOT / "experiments/alexnet/outputs"
OUT.mkdir(parents=True, exist_ok=True)
MAN = pd.read_csv(CACHE / "manifest.csv")
MAN["rec_id"] = MAN.subject_id + "|" + MAN.test + "|" + MAN.trial

SMOKE = "--smoke" in sys.argv


def alexnet_adapted(in_channels: int = 2, num_classes: int = 2) -> nn.Module:
    net = tvm.alexnet(weights=tvm.AlexNet_Weights.IMAGENET1K_V1)
    old = net.features[0]
    new = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding)
    with torch.no_grad():
        avg = old.weight.mean(dim=1, keepdim=True)
        new.weight.copy_(avg.repeat(1, in_channels, 1, 1))
        new.bias.copy_(old.bias)
    net.features[0] = new
    net.classifier[6] = nn.Linear(4096, num_classes)
    for p in net.features.parameters():
        p.requires_grad = False
    return net


def cfg_base(**kw):
    d = dict(lr=1e-4)
    if SMOKE:
        d.update(epochs=2, patience=2)
    d.update(kw)
    return TrainConfig(**d)


# ---------------- Part 1: LOSO ----------------
def part1():
    cfg = cfg_base()
    if SMOKE:
        subs = (sorted(MAN[MAN.label == 0].subject_id.unique())[:7]
                + sorted(MAN[MAN.label == 1].subject_id.unique())[:5])
        man = MAN[MAN.subject_id.isin(subs)]
    else:
        man = MAN
    t0 = time.time()
    folds = loso_cv(alexnet_adapted, man, CACHE, cfg, verbose=True)
    summ = loso_summary(folds)
    summ["wall_clock_s"] = round(time.time() - t0, 1)
    save_run(folds, summ, cfg, OUT, "alexnet_loso" + ("_smoke" if SMOKE else ""))
    print(f"--- LOSO DONE subject AUC {summ['subject_auc']:.4f} "
          f"({summ['wall_clock_s']/60:.1f} min) ---", flush=True)


# ---------------- Part 2: random recording split ----------------
def rows_norm_stats(df):
    out = np.zeros((2, 2))
    for i, ch in enumerate(("foot", "torso")):
        m = df[f"{ch}_mean"].to_numpy(float)
        sd = df[f"{ch}_std"].to_numpy(float)
        out[i] = (m.mean(), np.sqrt(max((sd ** 2).mean() + m.var(), 1e-12)))
    return out


def wmetrics(y, p, thr=0.5):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= thr).astype(int)
    return {"auc": float(roc_auc_score(y, p)),
            "accuracy": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "confusion": confusion_matrix(y, pred).tolist(), "n": int(len(y))}


def grouped(df, probs, key):
    g = df[[key, "label"]].copy(); g["prob"] = list(probs)
    per = g.groupby(key).agg(label=("label", "first"), prob=("prob", "mean"))
    return wmetrics(per["label"].to_numpy(), per["prob"].to_numpy())


def part2():
    # literature-style early stopping: window-level validation AUC.
    # Patched only for this part and restored afterwards, so the LOSO part
    # keeps the protocol's subject-level stopping.
    orig = T.subject_level_auc
    T.subject_level_auc = lambda ds, probs: float(
        roc_auc_score(ds.manifest["label"], list(probs)))
    results, preds = [], []
    seeds = (501,) if SMOKE else (501, 502, 503)
    for seed in seeds:
        T.set_seed(seed)
        units = MAN.groupby("rec_id")["label"].first().reset_index()
        tr_u, rest = train_test_split(units, test_size=0.30, random_state=seed,
                                      stratify=units["label"])
        va_u, te_u = train_test_split(rest, test_size=0.50, random_state=seed,
                                      stratify=rest["label"])
        pick = lambda u: MAN[MAN.rec_id.isin(u.rec_id)]
        tr, va, te = pick(tr_u), pick(va_u), pick(te_u)
        stats = rows_norm_stats(tr)
        cfg = cfg_base(seed=seed)
        mk = lambda df, aug: WindowDataset(df, CACHE, augment=aug,
                                           norm_stats=stats, rng_seed=seed)
        model = alexnet_adapted()
        res = T.train_one_fold(model, mk(tr, True), mk(va, False),
                               mk(te, False), cfg)
        row = {"seed": seed, "best_epoch": res["best_epoch"],
               "window": wmetrics(res["test_labels"], res["test_probs"]),
               "recording": grouped(te, res["test_probs"], "rec_id"),
               "subject": grouped(te, res["test_probs"], "subject_id"),
               "test_subjects_seen_in_train": float(
                   te["subject_id"].isin(tr["subject_id"]).mean())}
        results.append(row)
        d = te[["npy_path", "subject_id", "rec_id", "label"]].copy()
        d["prob"], d["seed"] = list(res["test_probs"]), seed
        preds.append(d)
        print(f"  random seed={seed}: window AUC {row['window']['auc']:.3f}, "
              f"acc {row['window']['accuracy']:.3f}", flush=True)
    sfx = "_smoke" if SMOKE else ""
    (OUT / f"alexnet_random{sfx}.json").write_text(json.dumps(results, indent=1))
    pd.concat(preds).to_csv(OUT / f"alexnet_random_preds{sfx}.csv", index=False)
    T.subject_level_auc = orig


if __name__ == "__main__":
    print(f"AlexNet experiment (smoke={SMOKE}) device={TrainConfig().device}",
          flush=True)
    part2()
    part1()
    print("ALEXNET EXPERIMENT COMPLETE", flush=True)

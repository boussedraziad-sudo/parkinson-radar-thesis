#!/usr/bin/env python3
"""The random-split detour: validate the way the literature does, on our data.

Three experiments, all with the same ResNet-18 linear probe (transfer learning)
on the same v2 window cache, differing ONLY in how train/val/test are drawn:

  E1 window_split     70/15/15 over individual WINDOWS, stratified by label.
                      The leakiest design (frame-level splits in the review).
  E2 recording_split  70/15/15 over whole RECORDINGS (Hayashi-style hold-out):
                      a recording's windows stay together, but one subject's
                      six recordings can straddle train and test.
  E3 collapse_test    8 subjects (stratified: 3 PD / 5 control) are held out
                      ENTIRELY.
                      The remaining 50 get a recording-level 70/15/15 split.
                      The same trained model is then scored twice: on the test
                      recordings of SEEN subjects, and on every window of the
                      8 NEVER-SEEN subjects. If the score collapses on the
                      never-seen group, the model was recognising people.

Deliberately mimicking the literature (and documented as such):
  - early stopping on WINDOW-level validation AUC (not subject-level),
  - no inverse-window-count weighting,
  - window-level accuracy/F1/confusion at threshold 0.5 as headline metrics.
Everything else (model, lr, epochs, augmentation, cache) matches the thesis's
v2 protocol so the split is the only variable that changed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

ROOT = Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
sys.path.insert(0, str(ROOT))
import src.train as T                              # noqa: E402
from src.dataset import WindowDataset              # noqa: E402
from src.models import resnet18_finetune           # noqa: E402

CACHE = ROOT / "outputs/preprocessed_v2"
OUT = ROOT / "classical_split/outputs"
OUT.mkdir(parents=True, exist_ok=True)

MAN = pd.read_csv(CACHE / "manifest.csv")
MAN["rec_id"] = MAN.subject_id + "|" + MAN.test + "|" + MAN.trial

# The literature early-stops on window-level validation metrics; reproduce that
# for these runs only (module-local patch, the LOSO code path is untouched).
T.subject_level_auc = lambda ds, probs: float(
    roc_auc_score(ds.manifest["label"], list(probs)))


def rows_norm_stats(df: pd.DataFrame) -> np.ndarray:
    """fold_norm_stats over an arbitrary row subset (law of total variance)."""
    out = np.zeros((2, 2), dtype=np.float64)
    for i, ch in enumerate(("foot", "torso")):
        m = df[f"{ch}_mean"].to_numpy(float)
        s = df[f"{ch}_std"].to_numpy(float)
        out[i] = (m.mean(), np.sqrt(max((s ** 2).mean() + m.var(), 1e-12)))
    return out


def split_frames(df, seed, unit=None):
    """70/15/15 split. unit=None splits rows (windows); otherwise groups."""
    if unit is None:
        tr, rest = train_test_split(df, test_size=0.30, random_state=seed,
                                    stratify=df["label"])
        va, te = train_test_split(rest, test_size=0.50, random_state=seed,
                                  stratify=rest["label"])
        return tr, va, te
    units = df.groupby(unit)["label"].first().reset_index()
    tr_u, rest_u = train_test_split(units, test_size=0.30, random_state=seed,
                                    stratify=units["label"])
    va_u, te_u = train_test_split(rest_u, test_size=0.50, random_state=seed,
                                  stratify=rest_u["label"])
    pick = lambda u: df[df[unit].isin(u[unit])]
    return pick(tr_u), pick(va_u), pick(te_u)


def window_metrics(labels, probs, thr=0.5):
    y, p = np.asarray(labels), np.asarray(probs)
    pred = (p >= thr).astype(int)
    return {
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "confusion": confusion_matrix(y, pred).tolist(),   # [[tn,fp],[fn,tp]]
        "n": int(len(y)),
    }


def grouped_metrics(df, probs, key):
    """Metrics after averaging window probs per recording or per subject."""
    g = df[[key, "label"]].copy()
    g["prob"] = list(probs)
    per = g.groupby(key).agg(label=("label", "first"), prob=("prob", "mean"))
    if per["label"].nunique() < 2:
        return {"auc": None, "n": int(len(per))}
    return window_metrics(per["label"].to_numpy(), per["prob"].to_numpy())


def eval_rows(model, df, stats, cfg):
    ds = WindowDataset(df, CACHE, augment=False, norm_stats=stats)
    probs, labels = T._eval_loader(model, T._loader(ds, cfg, shuffle=False),
                                   cfg.device)
    return ds, probs, labels


def run_once(exp, seed, train_df, val_df, test_df, extra=None):
    T.set_seed(seed)
    stats = rows_norm_stats(train_df)
    cfg = T.TrainConfig(seed=seed)          # epochs 20, lr 1e-3, patience 5
    mk = lambda df, aug: WindowDataset(df, CACHE, augment=aug,
                                       norm_stats=stats, rng_seed=seed)
    model = resnet18_finetune(2, freeze_until="layer4")
    res = T.train_one_fold(model, mk(train_df, True), mk(val_df, False),
                           None, cfg, verbose=False)

    row = {"experiment": exp, "seed": seed,
           "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
           "best_val_window_auc": res["best_val_window_auc"],
           "n_train": len(train_df), "n_val": len(val_df),
           "history": res["history"]}
    preds_rows = []

    def score(tag, df):
        _, probs, labels = eval_rows(model, df, stats, cfg)
        row[tag] = {
            "window": window_metrics(labels, probs),
            "recording": grouped_metrics(df, probs, "rec_id"),
            "subject": grouped_metrics(df, probs, "subject_id"),
            "test_subjects_seen_in_train": float(
                df["subject_id"].isin(train_df["subject_id"]).mean()),
        }
        d = df[["npy_path", "subject_id", "rec_id", "label"]].copy()
        d["prob"] = list(probs)
        d["experiment"], d["seed"], d["set"] = exp, seed, tag
        preds_rows.append(d)

    score("test", test_df)
    for tag, df in (extra or {}).items():
        score(tag, df)
    print(f"  {exp} seed={seed}: window AUC "
          f"{row['test']['window']['auc']:.3f}, acc "
          f"{row['test']['window']['accuracy']:.3f}, best epoch "
          f"{row['best_epoch']}", flush=True)
    return row, pd.concat(preds_rows)


def main():
    results, preds = [], []

    # E1: window-level random split (frame-level, the leakiest)
    for seed in (101, 102, 103):
        tr, va, te = split_frames(MAN, seed, unit=None)
        r, p = run_once("window_split", seed, tr, va, te)
        results.append(r); preds.append(p)

    # E2: recording-level random split (Hayashi-style hold-out)
    for seed in (201, 202, 203, 204, 205):
        tr, va, te = split_frames(MAN, seed, unit="rec_id")
        r, p = run_once("recording_split", seed, tr, va, te)
        results.append(r); preds.append(p)

    # E3: collapse test — same leaky training, plus 8 never-seen subjects
    subj = MAN.groupby("subject_id")["label"].first().reset_index()
    for seed in (301, 302, 303):
        rest, held = train_test_split(subj, test_size=8, random_state=seed,
                                      stratify=subj["label"])
        seen = MAN[MAN.subject_id.isin(rest.subject_id)]
        unseen = MAN[MAN.subject_id.isin(held.subject_id)]
        tr, va, te = split_frames(seen, seed, unit="rec_id")
        r, p = run_once("collapse_test", seed, tr, va, te,
                        extra={"never_seen": unseen})
        r["held_out_subjects"] = sorted(held.subject_id.tolist())
        results.append(r); preds.append(p)

    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    pd.concat(preds).to_csv(OUT / "predictions.csv", index=False)
    print(f"wrote {OUT/'results.json'} and predictions.csv "
          f"({sum(len(p) for p in preds)} prediction rows)")


if __name__ == "__main__":
    main()

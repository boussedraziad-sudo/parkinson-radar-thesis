#!/usr/bin/env python3
"""Grad-CAM over the v2 LOSO checkpoints.

For every fold (one checkpoint per held-out subject) this script loads the
trained model, replays the held-out subject's windows through it with the same
fold-local standardisation the model was trained under, and computes Grad-CAM
maps for the PD logit at the last convolutional stage:

    smallcnn_v2     -> third conv block ReLU output   (B, 64, 56, 56)
    resnet18_fc_v2  -> layer4 output                  (B, 512, 7, 7)

The weighted-sum-then-ReLU map is upsampled to 224x224 and summarised per
window as the mean attention inside three Doppler row bands (torso <= 200 Hz,
foot 200-500 Hz, background > 500 Hz, both Doppler signs pooled) plus the
share of total attention mass per band and the attention centre of mass.

Fold-local statistics: the training run derived normalisation stats from the
inner-train subjects only (seeded by the held-out subject id), and that split
is fully deterministic, so we reproduce the EXACT stats each checkpoint saw
rather than the all-minus-held-out approximation.

Outputs
    outputs/metrics/gradcam_summary.csv      per-window rows, both models
    outputs/figures/gradcam_gallery.png      8 example windows with overlays
    outputs/figures/gradcam_profile.png      aggregate Doppler attention profile

Usage
    .venv/bin/python tools/run_gradcam.py --validate   # 1 fold sanity check
    .venv/bin/python tools/run_gradcam.py              # all folds, both models
"""

from __future__ import annotations

import argparse
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

ROOT = Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import vizstyle as V  # noqa: E402
from src.dataset import WindowDataset  # noqa: E402
from src.models import SmallCNN, resnet18_finetune  # noqa: E402
from src.preprocessing import fold_norm_stats  # noqa: E402
from src.train import make_inner_split, subject_labels  # noqa: E402

CACHE = ROOT / "outputs/preprocessed_v2"
RUNS = ROOT / "outputs/runs"
METRICS = ROOT / "outputs/metrics"
FIGS = ROOT / "outputs/figures"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
H = 224  # cached window height (Doppler rows) and Grad-CAM output size
DOPPLER_SPAN_HZ = (-800.0, 800.0)  # 320 input bins at 5 Hz, row 0 = -800 Hz
TORSO_HZ, FOOT_HZ = 200.0, 500.0
STRADDLE_THR = 0.15  # |pos_frac_foot - 0.5| <= thr -> window straddles the turn

MODELS = {
    "smallcnn_v2": {
        "ckpt_dir": RUNS / "ckpt_smallcnn_v2",
        "build": lambda: SmallCNN(in_channels=2),
        # features[10] is the ReLU closing the third conv block, before GAP.
        "target": lambda m: m.features[10],
    },
    "resnet18_fc_v2": {
        "ckpt_dir": RUNS / "ckpt_resnet18_fc_v2",
        "build": lambda: resnet18_finetune(in_channels=2, pretrained=False,
                                           freeze_until="layer4"),
        "target": lambda m: m.layer4,
    },
}


# ---------------------------------------------------------------------------
# Doppler-row band masks on the 224-row resized axis
# ---------------------------------------------------------------------------

def row_freqs() -> np.ndarray:
    """Centre frequency (Hz) of each of the 224 Doppler rows, ascending."""
    lo, hi = DOPPLER_SPAN_HZ
    return lo + (np.arange(H) + 0.5) * (hi - lo) / H


ROW_F = row_freqs()
MASK_TORSO = np.abs(ROW_F) <= TORSO_HZ
MASK_FOOT = (np.abs(ROW_F) > TORSO_HZ) & (np.abs(ROW_F) <= FOOT_HZ)
MASK_BG = np.abs(ROW_F) > FOOT_HZ


# ---------------------------------------------------------------------------
# Grad-CAM machinery
# ---------------------------------------------------------------------------

class GradCAM:
    """Hooks one layer; produces per-window maps for the PD logit (class 1)."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.act: torch.Tensor | None = None
        self.grad: torch.Tensor | None = None
        self._h = target_layer.register_forward_hook(self._fwd_hook)

    def _fwd_hook(self, module, inputs, output):
        self.act = output
        output.register_hook(self._grad_hook)

    def _grad_hook(self, g):
        self.grad = g

    def run(self, xb: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Returns (cams, probs): cams (B, 224, 224) max-normalised, probs (B,)."""
        self.model.zero_grad(set_to_none=True)
        # Frozen backbones (resnet) have requires_grad=False everywhere below
        # the head; making the INPUT require grad forces every intermediate
        # activation to build graph so the hook receives a gradient.
        xb = xb.requires_grad_(True)
        logits = self.model(xb)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        # Per-sample gradients are independent in a feedforward net in eval
        # mode, so one backward on the summed PD logit serves the whole batch.
        logits[:, 1].sum().backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)          # (B, K, 1, 1)
        cam = torch.relu((w * self.act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(H, H), mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze(1)
        peak = cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        return (cam / peak).detach().cpu().numpy(), probs

    def close(self):
        self._h.remove()


def load_fold_model(name: str, subject: str) -> tuple[torch.nn.Module, dict]:
    spec = MODELS[name]
    ckpt = torch.load(spec["ckpt_dir"] / f"fold_{subject}.pt",
                      map_location="cpu", weights_only=False)
    model = spec["build"]()
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    return model, ckpt


def fold_stats_exact(manifest: pd.DataFrame, subject: str, cfg: dict) -> np.ndarray:
    """Reproduce the exact inner-train normalisation stats of this fold."""
    labels = subject_labels(manifest)
    all_subjects = sorted(manifest["subject_id"].unique())
    fold_seed = int(cfg["seed"]) + zlib.crc32(subject.encode()) % 100_000
    pool = [s for s in all_subjects if s != subject]
    inner_train, _ = make_inner_split(pool, labels,
                                      int(cfg["inner_val_subjects"]),
                                      seed=fold_seed)
    return fold_norm_stats(manifest, inner_train)


def cam_band_summary(cam: np.ndarray) -> dict:
    """Band summaries of one (224, 224) max-normalised map."""
    prof = cam.mean(axis=1)  # mean over time -> per-Doppler-row attention
    tot = float(prof.sum())
    com = float((np.abs(ROW_F) * prof).sum() / tot) if tot > 0 else float("nan")
    out = {
        "att_mean_torso": float(cam[MASK_TORSO].mean()),
        "att_mean_foot": float(cam[MASK_FOOT].mean()),
        "att_mean_bg": float(cam[MASK_BG].mean()),
        "att_com_abs_hz": com,
    }
    if tot > 0:
        out["att_mass_torso"] = float(prof[MASK_TORSO].sum() / tot)
        out["att_mass_foot"] = float(prof[MASK_FOOT].sum() / tot)
        out["att_mass_bg"] = float(prof[MASK_BG].sum() / tot)
    else:
        out["att_mass_torso"] = out["att_mass_foot"] = out["att_mass_bg"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Per-fold pass
# ---------------------------------------------------------------------------

def run_fold(name: str, subject: str, manifest: pd.DataFrame,
             keep_cams: bool = False):
    """Grad-CAM for one held-out subject. Returns (records, profiles, cams?)."""
    model, ckpt = load_fold_model(name, subject)
    cfg = ckpt["config"]
    stats = fold_stats_exact(manifest, subject, cfg)
    ds = WindowDataset(manifest, CACHE, subjects=[subject], augment=False,
                       channel_mode=cfg["channel_mode"], norm_stats=stats,
                       max_turn_frac=cfg["max_turn_frac"])
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    engine = GradCAM(model, MODELS[name]["target"](model))

    records, profiles, cams_all = [], [], []
    ds_idx = 0
    for xb, yb, _ in loader:
        cams, probs = engine.run(xb.to(DEVICE))
        for j in range(cams.shape[0]):
            row = ds.manifest.iloc[ds_idx]
            pf = float(row["pos_frac_foot"])
            rec = {
                "model": name,
                "subject_id": subject,
                "group": row["group"],
                "label": int(row["label"]),
                "test": row["test"],
                "trial": row["trial"],
                "window_idx": int(row["window_idx"]),
                "ds_idx": ds_idx,
                "t_start_s": float(row["t_start_s"]),
                "npy_path": row["npy_path"],
                "pos_frac_foot": pf,
                "turn_straddle": 1.0 - 2.0 * abs(pf - 0.5) if np.isfinite(pf) else float("nan"),
                "prob_pd": float(probs[j]),
                "correct": int((probs[j] >= 0.5) == bool(row["label"])),
                "best_val_auc": float(ckpt["best_val_auc"]),
            }
            rec.update(cam_band_summary(cams[j]))
            records.append(rec)
            profiles.append(cams[j].mean(axis=1))
            if keep_cams:
                cams_all.append(cams[j])
            ds_idx += 1
    engine.close()
    del model
    return records, profiles, cams_all


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def pick_gallery(df: pd.DataFrame) -> pd.DataFrame:
    """8 windows: 2 each of PD/control x correct/wrong, at extreme confidence."""
    picks = []
    d = df.dropna(subset=["prob_pd"])
    pd_rows = d[d["label"] == 1].sort_values("prob_pd", ascending=False)
    ct_rows = d[d["label"] == 0].sort_values("prob_pd", ascending=True)
    picks.append(pd_rows.head(2).assign(panel="PD, called PD"))
    picks.append(ct_rows.head(2).assign(panel="Control, called control"))
    picks.append(pd_rows.tail(2).assign(panel="PD, called control"))
    picks.append(ct_rows.tail(2).assign(panel="Control, called PD"))
    return pd.concat(picks, ignore_index=True)


def make_gallery(gal: pd.DataFrame, manifest: pd.DataFrame, name: str,
                 out_path: Path) -> None:
    """2x4 gallery: foot-channel spectrogram in grey, Grad-CAM overlay."""
    import matplotlib.pyplot as plt

    # Recompute cams only for the selected folds.
    cams_by_key: dict[tuple, np.ndarray] = {}
    for subject in gal["subject_id"].unique():
        need = gal[gal["subject_id"] == subject]
        _, _, cams = run_fold(name, subject, manifest, keep_cams=True)
        for _, r in need.iterrows():
            cams_by_key[(subject, int(r["ds_idx"]))] = cams[int(r["ds_idx"])]

    V.apply()
    fig, axes = plt.subplots(2, 4, figsize=(16.5, 8.0))
    order = ["PD, called PD", "Control, called control",
             "PD, called control", "Control, called PD"]
    # Row 1: correct calls. Row 2: wrong calls. Two examples per category,
    # arranged so each column pair shares a category.
    slots = []
    for cat in order[:2]:
        slots.extend(gal[gal["panel"] == cat].itertuples())
    for cat in order[2:]:
        slots.extend(gal[gal["panel"] == cat].itertuples())

    for ax, r in zip(axes.ravel(), slots):
        arr = np.load(CACHE / r.npy_path).astype(np.float32)  # (2, 224, 224)
        spec = arr[0]  # foot channel, cached log1p scale
        cam = cams_by_key[(r.subject_id, int(r.ds_idx))]
        extent = [0.0, 3.0, DOPPLER_SPAN_HZ[0], DOPPLER_SPAN_HZ[1]]
        ax.imshow(spec, aspect="auto", origin="lower", cmap="gray_r",
                  extent=extent)
        ax.imshow(cam, aspect="auto", origin="lower", cmap=V.SEQ,
                  extent=extent, alpha=0.45, vmin=0, vmax=1)
        for f in (-FOOT_HZ, -TORSO_HZ, TORSO_HZ, FOOT_HZ):
            ax.axhline(f, color=V.SURFACE, lw=0.6, ls=":", alpha=0.8)
        ax.grid(False)
        col = V.PD if r.label == 1 else V.CONTROL
        ax.set_title(f"{r.panel}\n{r.subject_id}  p(PD)={r.prob_pd:.2f}",
                     fontsize=12, loc="left", color=col, pad=6)
        ax.tick_params(labelsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Doppler (Hz)", fontsize=11)
    for ax in axes[1, :]:
        ax.set_xlabel("time (s)", fontsize=11)
    fig.suptitle(f"Grad-CAM, {name}: where the PD logit looks "
                 "(overlay alpha 0.45; dotted lines mark the 200/500 Hz bands)",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)


def make_profile_figure(profiles: dict[str, np.ndarray], df: pd.DataFrame,
                        out_path: Path) -> None:
    """(1) mean attention vs Doppler per model; (2) straddling vs single."""
    import matplotlib.pyplot as plt

    V.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8))

    def shade(ax):
        for lo, hi in [(-FOOT_HZ, -TORSO_HZ), (TORSO_HZ, FOOT_HZ)]:
            ax.axvspan(lo, hi, color=V.ACCENT, alpha=0.07, lw=0)
        ax.axvspan(-TORSO_HZ, TORSO_HZ, color=V.GOOD, alpha=0.07, lw=0)

    colors = {"smallcnn_v2": V.CONTROL, "resnet18_fc_v2": V.PD}
    for name, prof in profiles.items():
        ax1.plot(ROW_F, prof.mean(axis=0), color=colors.get(name, V.INK),
                 lw=1.8, label=name)
    shade(ax1)
    ax1.set_xlabel("Doppler (Hz)")
    ax1.set_ylabel("mean Grad-CAM attention")
    ax1.set_xlim(DOPPLER_SPAN_HZ)
    V.title(ax1, "Attention over the Doppler axis",
            "mean over all windows and folds; green = torso band, violet = foot band")
    ax1.legend()

    sc = df[df["model"] == "smallcnn_v2"].reset_index(drop=True)
    prof_sc = profiles["smallcnn_v2"]
    strad = sc["turn_straddle"] >= (1.0 - 2.0 * STRADDLE_THR)
    strad = strad & sc["turn_straddle"].notna()
    single = ~strad & sc["turn_straddle"].notna()
    ax2.plot(ROW_F, prof_sc[strad.to_numpy()].mean(axis=0), color=V.ACCENT,
             lw=1.8, label=f"turn-straddling (n={int(strad.sum())})")
    ax2.plot(ROW_F, prof_sc[single.to_numpy()].mean(axis=0), color=V.INK_2,
             lw=1.8, label=f"single-direction (n={int(single.sum())})")
    shade(ax2)
    ax2.set_xlabel("Doppler (Hz)")
    ax2.set_ylabel("mean Grad-CAM attention")
    ax2.set_xlim(DOPPLER_SPAN_HZ)
    V.title(ax2, "Turn-straddling vs single-direction windows",
            f"smallcnn_v2; straddling = |pos_frac_foot - 0.5| <= {STRADDLE_THR}")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def corr_line(x: pd.Series, y: pd.Series, label: str) -> str:
    m = x.notna() & y.notna()
    if m.sum() < 3:
        return f"    {label}: insufficient data"
    r, p = pearsonr(x[m], y[m])
    rs, ps = spearmanr(x[m], y[m])
    return (f"    {label}: pearson r={r:+.3f} (p={p:.1e}), "
            f"spearman rho={rs:+.3f} (p={ps:.1e})")


def report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("GRAD-CAM FINDINGS")
    print("=" * 74)
    n_rows = {"torso": int(MASK_TORSO.sum()), "foot": int(MASK_FOOT.sum()),
              "bg": int(MASK_BG.sum())}
    print(f"Doppler row bands (of {H}): torso |f|<=200 Hz -> {n_rows['torso']} rows, "
          f"foot 200-500 Hz -> {n_rows['foot']} rows, "
          f"background >500 Hz -> {n_rows['bg']} rows")
    for name, g in df.groupby("model"):
        print(f"\n--- {name}  ({g['subject_id'].nunique()} folds, {len(g)} windows) ---")
        mass = g[["att_mass_torso", "att_mass_foot", "att_mass_bg"]].mean()
        dens = g[["att_mean_torso", "att_mean_foot", "att_mean_bg"]].mean()
        print(f"  attention MASS share:    torso {mass['att_mass_torso']:.3f}  "
              f"foot {mass['att_mass_foot']:.3f}  background {mass['att_mass_bg']:.3f}")
        print(f"  attention density (mean): torso {dens['att_mean_torso']:.3f}  "
              f"foot {dens['att_mean_foot']:.3f}  background {dens['att_mean_bg']:.3f}")
        print(f"  attention centre of mass: {g['att_com_abs_hz'].mean():.0f} Hz (|f|)")
        for lab, tag in [(1, "PD windows     "), (0, "control windows")]:
            gg = g[g["label"] == lab]
            print(f"  {tag}: mass torso {gg['att_mass_torso'].mean():.3f}  "
                  f"foot {gg['att_mass_foot'].mean():.3f}  "
                  f"bg {gg['att_mass_bg'].mean():.3f}  "
                  f"com {gg['att_com_abs_hz'].mean():.0f} Hz")
        print("  correlation with turn straddle (1 = straddling the turn):")
        print(corr_line(g["turn_straddle"], g["att_mass_torso"], "torso mass"))
        print(corr_line(g["turn_straddle"], g["att_mass_foot"], "foot mass "))
        print(corr_line(g["turn_straddle"], g["att_mass_bg"], "bg mass   "))
        print(corr_line(g["pos_frac_foot"], g["att_com_abs_hz"], "pos_frac_foot vs att-CoM"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="run a single fold per model and stop")
    ap.add_argument("--models", nargs="*", default=list(MODELS),
                    help="subset of models to run")
    args = ap.parse_args()

    manifest = pd.read_csv(CACHE / "manifest.csv")
    subjects = sorted(manifest["subject_id"].unique())
    if args.validate:
        subjects = subjects[:1]

    all_records: list[dict] = []
    profiles: dict[str, np.ndarray] = {}
    for name in args.models:
        ckpt_dir = MODELS[name]["ckpt_dir"]
        missing = [s for s in subjects if not (ckpt_dir / f"fold_{s}.pt").exists()]
        if missing:
            print(f"[{name}] SKIPPED: {len(missing)} checkpoints missing "
                  f"(e.g. {missing[:3]})")
            continue
        t0 = time.time()
        recs, profs = [], []
        for i, s in enumerate(subjects):
            r, p, _ = run_fold(name, s, manifest)
            recs.extend(r)
            profs.extend(p)
            if (i + 1) % 10 == 0 or i == len(subjects) - 1:
                print(f"[{name}] fold {i + 1}/{len(subjects)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        all_records.extend(recs)
        profiles[name] = np.asarray(profs, dtype=np.float32)

    df = pd.DataFrame(all_records)
    METRICS.mkdir(parents=True, exist_ok=True)
    csv_path = METRICS / "gradcam_summary.csv"
    df.drop(columns=["ds_idx", "npy_path"]).to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}  ({len(df)} rows)")

    if args.validate:
        report(df)
        print("\nVALIDATION PASS DONE (1 fold per model). Re-run without "
              "--validate for the full pass.")
        return

    FIGS.mkdir(parents=True, exist_ok=True)
    if "smallcnn_v2" in profiles:
        gal = pick_gallery(df[df["model"] == "smallcnn_v2"])
        make_gallery(gal, manifest, "smallcnn_v2", FIGS / "gradcam_gallery.png")
        print(f"wrote {FIGS / 'gradcam_gallery.png'}")
    if profiles:
        make_profile_figure(profiles, df, FIGS / "gradcam_profile.png")
        print(f"wrote {FIGS / 'gradcam_profile.png'}")

    report(df)


if __name__ == "__main__":
    main()

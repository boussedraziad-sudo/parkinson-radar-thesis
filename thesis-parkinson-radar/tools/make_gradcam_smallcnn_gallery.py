#!/usr/bin/env python3
"""Render the Grad-CAM example gallery for smallcnn_v2.

Same approach as make_gradcam_resnet_gallery.py: pick the 2x4 layout
(confident correct calls on top, confident mistakes below) from the stored
per-window summary and re-render it, recomputing the maps only for the
folds the gallery needs. Used to regenerate the figure with layout/font
tweaks without rerunning the full 58-fold pass.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/ziad.boussedra/Desktop/Master Thesis/thesis-parkinson-radar")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import run_gradcam as G  # noqa: E402

NAME = "smallcnn_v2"

df = pd.read_csv(G.METRICS / "gradcam_summary.csv")
d = df[df["model"] == NAME]
print(f"{NAME}: {d['subject_id'].nunique()} folds, {len(d)} windows in summary")
assert d["subject_id"].nunique() == 58, "expected all 58 folds in the summary"

gal_keys = G.pick_gallery(d)[["subject_id", "test", "trial", "window_idx",
                              "panel", "prob_pd"]]
manifest = pd.read_csv(G.CACHE / "manifest.csv")

# The stored summary dropped ds_idx/npy_path, which the renderer needs, so
# replay only the folds the gallery picked and match windows back up.
recs = []
for s in gal_keys["subject_id"].unique():
    r, _, _ = G.run_fold(NAME, s, manifest)
    recs.extend(r)
full = pd.DataFrame(recs)
gal = full.merge(gal_keys.drop(columns=["prob_pd"]),
                 on=["subject_id", "test", "trial", "window_idx"])
assert len(gal) == len(gal_keys), (len(gal), len(gal_keys))
chk = gal.merge(gal_keys, on=["subject_id", "test", "trial", "window_idx"])
assert (chk["prob_pd_x"] - chk["prob_pd_y"]).abs().max() < 1e-4, "prob mismatch"

out = G.FIGS / "gradcam_gallery.png"
G.make_gallery(gal, manifest, NAME, out)
print(f"wrote {out}")

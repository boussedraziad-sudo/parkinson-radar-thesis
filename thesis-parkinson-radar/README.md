# Parkinson's Disease Detection from Radar Gait Signals

Master's thesis project — UPM, 2026.
See `../project_summary_and_setup.md` for the full project description, methodology, and weekly work plan.

## Repository layout

```
thesis-parkinson-radar/
├── data/                  # Symlink to ../shared_ziad (the dataset; never committed)
├── src/                   # Library code — importable; analysis modules run headless
│   ├── config.py          # Central paths, feature groupings, experiment defaults
│   ├── data_loader.py     # Open .mat files (scipy v5), iterate subjects/trials
│   ├── radar_params.py    # Reference-paper STFT/Doppler constants
│   ├── qc.py              # Quality-control checks
│   ├── stats.py           # MWU + BH-FDR, Cliff's delta, subject-bootstrap CIs
│   ├── eda.py             # EDA figures + statistics      (python -m src.eda)
│   ├── preprocessing.py   # Windowing, normalization, resizing → .npy cache
│   ├── channels.py        # Foot/torso channel arrangement
│   ├── features.py        # Handcrafted features (duration-invariant)
│   ├── baseline.py        # Classical LOSO-CV ML          (python -m src.baseline)
│   ├── confound_analysis.py # Per-test / residualized / matched confound tests
│   ├── diagnostics.py     # Per-subject errors, caliper sweep, learning curve
│   ├── dataset.py         # PyTorch Dataset over preprocessed .npy windows
│   ├── models.py          # SmallCNN + ResNet-18 fine-tune
│   ├── train.py           # Deep LOSO-CV training loop
│   └── interpret.py       # Grad-CAM
├── notebooks/             # Runnable analysis + experiments
│   ├── 01_data_inspection.ipynb     Open one trial, plot ce_foot/ce_torso
│   ├── 02_dataset_overview.ipynb    Inventory, class balance, integrity, durations
│   ├── 03_eda_spectrograms.ipynb    Visual EDA: control vs PD, mean per group, diff map
│   ├── 04_quality_control.ipynb     QC pass; qc_report.csv + excluded_files.csv
│   ├── 05_feature_eda.ipynb         Per-trial feature extraction → trial_features.csv
│   ├── 06_preprocessing.ipynb       Window + normalize + resize → manifest.csv (deep track)
│   ├── 07_baselines.ipynb           (legacy) window-feature SVM/RF skeleton
│   ├── 08_cnn.ipynb                 SmallCNN with LOSO-CV       (deep track)
│   ├── 09_resnet.ipynb              ResNet-18 fine-tune LOSO-CV (deep track)
│   ├── 10_gradcam.ipynb             Grad-CAM interpretation     (deep track)
│   ├── 11_data_forensics.ipynb      Empirical answers to open questions → FINDINGS.md
│   ├── 12_eda_statistics.ipynb      Rigorous EDA (MWU/FDR/Cliff's δ, PCA) — wraps src.eda
│   └── 13_baseline_models.ipynb     Classical LOSO baselines + permutation — wraps src.baseline
├── outputs/
│   ├── preprocessed/      # Windowed .npy cache + manifest.csv  (deep track)
│   ├── models/            # Per-fold checkpoints
│   ├── figures/           # eda_*.png, baseline_*.png, Grad-CAM maps
│   └── metrics/           # trial_features.csv, baseline_auc_grid.csv, inventory.csv …
├── reports/               # eda_summary.json, baseline_summary.json
├── subjects.csv           # subject_id, group, n_trials, age/sex/UPDRS (TBC)
├── excluded_files.csv     # Output of QC pass (Notebook 04)
├── RESULTS.md             # Real results + thesis-direction decision  ← start here
├── FINDINGS.md            # Notebook-11 forensics + draft email to data owner
├── requirements.txt           # Full stack incl. torch (deep track, conda)
├── requirements-analysis.txt  # Lightweight stack (classical track, .venv)
└── .gitignore
```

## Two tracks

**Classical track** — EDA + handcrafted-feature baselines. Runs on a plain
laptop, no GPU, no 106 GB data (works off the cached `trial_features.csv`):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-analysis.txt
.venv/bin/python -m src.eda                # → outputs/figures/eda_*.png + reports/eda_summary.json
.venv/bin/python -m src.baseline           # → baseline_*.png + reports/baseline_summary.json
.venv/bin/python -m src.confound_analysis  # → confound_*.png + reports/confound_summary.json
.venv/bin/python -m src.diagnostics        # → diag_*.png + reports/diagnostics_summary.json
# Notebooks 12 & 13 wrap the EDA / baseline functions for interactive use.
```

**Deep track** — windowing → CNN/ResNet → Grad-CAM. Needs torch + the dataset,
ideally the GAPS GPU box:

```bash
conda create -n thesis python=3.10 -y && conda activate thesis
pip install -r requirements.txt
jupyter lab                          # notebooks 06 → 08 → 09 → 10
```

Each notebook starts with a `setup` cell that adds the repo root to `sys.path` so `from src.* import ...` works from `notebooks/`.

## Results so far

See **[RESULTS.md](RESULTS.md)** for the full write-up. Short version: classical
trial-level features give a subject-level LOSO AUC of ~0.62. Confound-control
tests (duration-matched subset, duration-residualized features, test2-only) show
this is a **weak but genuine gait signal — not just trial length**: it survives
duration matching (0.62–0.64 at post-match p=0.95) and rises to 0.68 once
duration is residualized out (permutation **p = 0.022**, significant). It is the
honest floor the windowed CNN must beat, and fixed-length windows remove the
duration confound by construction.

## Open-questions workflow

`FINDINGS.md` holds the empirical answers from Notebook 11 (executed): the
reverse-engineered `ce_*` mapping (adaptive per-trial, not element-wise),
Doppler-axis uniformity, the two file anomalies (both benign), Doppler symmetry,
and the subject-fingerprint result (within/between ratio 0.37 → LOSO essential).
Its bottom section is a ready-to-send draft email to the dataset owner
(Ignacio López-Delgado) asking for the age/sex/UPDRS/medication table, the `ce_*`
formula, and confirmation on the anomalies.

## Reference parameters

Radar/STFT constants quoted from the validation paper (López-Delgado et al., 2026) live in
[`src/radar_params.py`](src/radar_params.py): 23 GHz carrier, 1.4 GHz bandwidth, 50 ms Hann
STFT window with 1-sample hop, ~20 Hz Doppler resolution, ±500 Hz displayed band, torso ≤ 200 Hz,
foot 200–500 Hz. See Section 3 of `../project_summary_and_setup.md` for the full table.

## Dataset summary

| Group | Folder prefix | Count |
|---|---|---|
| Healthy controls | `fisc_` | 33 |
| Parkinson's | `fisp_` | 25 |
| Prodromal | `fis_` | 0 (Phase 2) |

Expected 6 `.mat` files per subject (2 tests × 3 trials). Confirmed anomalies in the v1 release:

- **`fisp_022`** has an extra `test1/trial4` (7 files total)
- **`fisp_048`** is missing `test2/trial2` (5 files total)

Both are flagged in `subjects.csv` and surfaced by Notebook 02.

## Data — never commit

The dataset is 106 GB. `.gitignore` excludes `data/`, `*.mat`, `*.npy`, and `outputs/preprocessed/`.
The `data/` symlink points at `../shared_ziad`; if you move the repo, recreate it with:

```bash
ln -sfn /path/to/shared_ziad data
```

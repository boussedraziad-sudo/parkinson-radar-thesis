# Parkinson's Disease Detection from Radar Gait Signals — Project Summary

**Student:** Ziad Boussedra (`ziad.boussedra@alumnos.upm.es`)
**Institution:** Universidad Politécnica de Madrid (UPM)
**Supervisor:** Juan Ignacio Godino Llorente (Nacho) — `juiggo@gmail.com`
**Research group contacts:** Ignacio Esteban López-Delgado (`ie.lopez@upm.es`), Jesús Grajal (`jesus.grajal@upm.es`)
**Lab support:** Camilo (account setup), Amber (calendar reservations)
**Duration:** ~2 months

---

## 1. Project Summary

This thesis extends a validated FMCW radar-based gait monitoring system developed at UPM into an automated Parkinson's disease (PD) screening framework. The reference work (*Radar Network for Gait Monitoring: Technology and Validation* — López-Delgado et al., IEEE TBME, 2026) established that a multi-node radar network can accurately extract spatiotemporal gait parameters from both healthy and PD populations. However, it does not perform automated disease classification — that is the contribution of this thesis.

### What we are building
An end-to-end machine learning pipeline that takes radar-derived micro-Doppler spectrograms as input and outputs a classification decision: healthy vs Parkinson's disease.

### Why it matters
Current clinical PD assessment relies on subjective evaluation by specialists in controlled environments. A non-contact, radar-based screening system could enable continuous, privacy-preserving monitoring in domestic settings — without wearables, cameras, or patient effort.

---

## 2. Dataset

### Source
Downloaded locally from UPMdrive (`shared_ziad`). Total size: **~109 GB** — full dataset available locally on Mac. The folder also contains `readme.txt` (authoritative description of variables and protocol — confirmed by reading it).

### Subject naming convention (confirmed)
- **`fisc_XXX`** — healthy controls
- **`fisp_XXX`** — Parkinson's disease patients
- **`fis_XXX`** — prodromal candidates (not uploaded yet; do not use)

### Clinical groups (counts from disk)
| Group | Folder prefix | Count |
|---|---|---|
| Healthy controls | `fisc_` (005–074) | **33** |
| Parkinson's patients | `fisp_` (012–052) | **25** |
| Prodromal | `fis_` | 0 — *Phase 2, not yet uploaded* |
| **Total subjects available** | | **58** |

> Dataset is far larger and better-balanced than the original estimate. Class balance is ~57% controls / ~43% PD — no strong imbalance to mitigate. The remaining clinical detail to resolve is the **young (G1) vs older (G2) split inside controls**, which is not encoded in folder names and requires an age table from the team.

**File-count anomalies found in the v1 release** (confirmed by filesystem inventory, 2026-05-28):
- `fisp_022` — **7** files instead of 6 (extra `test1/trial4`)
- `fisp_048` — **5** files instead of 6 (missing `test2/trial2`)

Expected files: 33×6 + 23×6 + 7 + 5 = **348** (confirmed by `find shared_ziad -name 'stft_data.mat' | wc -l`). Both anomalies are flagged in `thesis-parkinson-radar/subjects.csv`.

### Folder structure (confirmed)
```
shared_ziad/
├── readme.txt              # Authoritative dataset description
├── fisc_005/               # Healthy control
│   ├── test1/              # TUG with chair
│   │   ├── trial1/stft_data.mat
│   │   ├── trial2/stft_data.mat
│   │   └── trial3/stft_data.mat
│   └── test2/              # TUG without chair (stand-start)
│       ├── trial1/stft_data.mat
│       ├── trial2/stft_data.mat
│       └── trial3/stft_data.mat
├── fisp_012/               # PD patient (same 2×3 structure)
│   ...
└── fisp_052/
```

Per subject: **2 tests × 3 trials = 6 spectrogram files** (nominal). Actual total: **348 `.mat` files** across 58 subjects (see anomalies above). Files are 240–310 MB each; full dataset is 106 GB.

### Test protocol (confirmed from readme)
- **test1** — Timed-Up-and-Go (TUG): stand from chair → walk → turn → walk back → sit
- **test2** — TUG **without** chair (begin and end standing)
- **trial1 / trial2 / trial3** — each test repeated 3 times

> The reference paper describes a longer 5-trial protocol (TUG + 3 paced walks + 120 s walk). The released dataset uses the simpler 2-test × 3-trial design above — design ablations accordingly.

### What `stft_data.mat` contains (confirmed from readme)

The MATLAB file exposes both the **complex STFT** and a **contrast-enhanced magnitude** version of the same data, plus per-radar arrays:

| Variable | Type | Dimensions | Notes |
|---|---|---|---|
| `stft_foot` | complex | Doppler-freq × time | **Pre-combined across foot radars** (SNR-selected) |
| `stft_torso` | complex | Doppler-freq × time | **Pre-combined across torso radars** (SNR-selected) |
| `ce_foot` | real | Doppler-freq × time | `|stft_foot|` after contrast enhancement |
| `ce_torso` | real | Doppler-freq × time | `|stft_torso|` after contrast enhancement |
| `all_doppler_time` | complex | Doppler-freq × time × radar idx | Per-radar STFT (before combination) |
| `all_ce_time` | real | Doppler-freq × time × radar idx | Per-radar magnitude after CE |
| `idx_foot_max_snr` | indices | — | Foot radar indices used in pre-combination |
| `idx_torso_max_snr` | indices | — | Torso radar indices used in pre-combination |
| `all_loc`, `all_dir` | metadata | — | Radar location (foot/torso) and orientation (±) |
| `all_target_time` | real | time × radar idx | Raw time-domain signal before STFT |
| `t_axis_target` | vector | — | Time-axis values |
| `doppler_axis` | vector | — | Doppler-frequency-axis values |
| `param` | struct | — | Radar configuration parameters |

**Implication for the pipeline:** SNR-based combination of multiple foot/torso radars is **already done** in `stft_foot`/`stft_torso`. The remaining "fusion" decision is just **how to arrange foot and torso together** as model input (e.g. 2-channel stack), not how to combine radars within each group. Custom recombination is still possible via `all_doppler_time` + the SNR-index variables if needed for ablations.

### Empirical findings (confirmed by running notebooks 01–05, 2026-05-29)

| Finding | Value | Where it lives |
|---|---|---|
| `.mat` format | **MATLAB v5** (not v7.3) — use `scipy.io.loadmat`, not `h5py` | `src/data_loader.py`, `src/qc.py` |
| Array layout | `(doppler, time)` directly — **no transpose** needed | `src/data_loader.py` |
| Doppler axis size | **320 bins** over ±800 Hz | confirmed in Notebook 02 |
| Doppler bin spacing | **5 Hz** (4× zero-padded vs the paper's 20 Hz intrinsic resolution) | `src/radar_params.py:DOPPLER_BIN_SPACING_HZ` |
| Trial-duration confound | **PD trials ~10% longer** than control — any energy-summed feature would leak duration | flagged in Notebook 02 § 5b, Notebook 05 § 0; `src/features.py` rewritten to be duration-invariant |
| `ce_*` provenance | **Not described in the reference paper** (confirmed by exhaustive grep — 0 hits for contrast/equalise/normalise/dB as a processing step). The CE pipeline is a dataset-specific addition. | Notebook 11 § 1 reverse-engineers the formula; awaits confirmation from Ignacio |
| Torso-radar combination | Per paper Fig. 6: "*prior to the combination, the Doppler-time matrix of node 2 is flipped with respect to the Doppler axis*" — a deliberate operation that collapses motion direction in the combined torso signal | Notebook 11 § 6 verifies the resulting Doppler symmetry empirically |
| CE formula (Notebook 11 § 1) | **Not element-wise.** Best dB fit (`a·log10(\|stft\|)+b`) gets R² between 0.46 and 0.95 across trials, and parameters drift an order of magnitude — looks like per-trial adaptive normalisation on top of a log/dB stage. | Email to Ignacio asks for the actual operation |
| Subject fingerprint (Notebook 11 § 8) | Median within/between subject distance ratio = **0.374**; 58/58 subjects below diagonal. | **LOSO is essential**, k-fold would be massively over-optimistic |
| Duration confound by test (Notebook 11 § 7) | test1: PD +10.1% ; test2: PD +3.4% | Confound is concentrated in the chair-stand (test1). test2 is comparatively clean |
| Doppler symmetry (Notebook 11 § 6) | `ce_foot` median +/− Doppler energy ratio **1.005 ± 0.043** | Direction is folded — safe to halve the Doppler axis as an input-size ablation |
| `param` reveals (Notebook 11 § 3) | Actual capture parameters: `BW=1.38 GHz` (paper rounds to 1.4), `ftrabajo=23.5 GHz` (paper rounds to 23). Capture date 2024-02-20. Undocumented `all_fig_*` variables present in every file. | Worth updating `src/radar_params.py` to match the actual capture values; ask Ignacio about the `all_fig_*` variables |
| Anomalies resolved (Notebook 11 §§ 4-5) | `fisp_022/test1/trial4` is a clean retry (similarities 0.82-0.86, in normal range) — keep. `fisp_048/test2/trial2` missing — trial1↔trial3 similarity 0.87 looks like a normal retry pair, so trial2 was likely a clean deletion. Proceed with 5 files. | Both still flagged in the email for the team to confirm intent |
| Doppler-axis uniformity (Notebook 11 § 2) | 320 bins, Δf = 5.00 Hz, range ±800 Hz — **identical in every one of the 348 files** (zero outliers) | Preprocessing can safely assume a fixed Doppler dimension |

> ⚠️ **Data cleaning required.** The readme states: *"Strict data cleaning is needed. This will be done for future data transfers, but this version provides a good starting point."* Plan an explicit QC pass on all `.mat` files before training.

---

## 3. Methodology

### Pipeline overview
```
stft_data.mat  (one per trial — foot/torso STFTs already SNR-combined)
      │
      ▼
[1] Load & inspect (ce_foot, ce_torso as default representation)
      │
      ▼
[2] Channel arrangement
      │  default:  2-channel stack [ce_foot, ce_torso]
      │  ablations: foot-only / torso-only
      │             custom recombination via all_doppler_time + idx_*_max_snr
      ▼
[3] Spectrogram preprocessing
      │  Log / Z-score normalize → segment into windows (2–5 s) → resize
      ▼
[4] Dataset construction
      │  Labels from folder prefix: fisc → 0 (control), fisp → 1 (PD)
      │  Leave-One-Subject-Out split (subject never in both train and test)
      ▼
[5] Classification
      │  Baseline: SVM, Random Forest on handcrafted features
      │  Deep learning: CNN, ResNet-18 (transfer learning from ImageNet)
      ▼
[6] Evaluation & interpretation
         AUC-ROC, F1, sensitivity, specificity + Grad-CAM visualization
```

### Key technical decisions
- **Input representation:** `ce_foot` + `ce_torso` (magnitude after contrast enhancement) as default — already pre-processed for CNN input. Compare against `|stft_*|` and log-magnitude as ablations.
- **Channel arrangement:** foot/torso radars are already SNR-combined within each group, so the only fusion decision left is foot-only vs torso-only vs 2-channel stack (default).
- **Window length:** 2–5 second segments, each containing multiple gait cycles.
- **Evaluation protocol:** Leave-One-Subject-Out cross-validation (LOSO-CV) — 59 folds total. Subject-level aggregation: average window probabilities to a single per-subject decision.
- **Primary metric:** AUC-ROC. Report F1, sensitivity, specificity as secondary.
- **Interpretation:** Grad-CAM on CNN to identify which time-frequency regions drive PD classification.

### Reference radar / STFT parameters

Extracted from López-Delgado et al. (the validation paper for this radar network). Constants live in `thesis-parkinson-radar/src/radar_params.py`. Note that the **paper's validation cohort is NOT our dataset** — paper validated on 8 subjects (3 G1 + 2 G2 + 3 G3); our dataset is a separate, larger collection (33 + 25 = 58 subjects, simpler 2-test × 3-trial protocol).

| Parameter | Value | Source |
|---|---|---|
| Carrier frequency | 23 GHz | Section II, Table II |
| Bandwidth | 1.4 GHz | Section III intro |
| Chirp duration `T_c` | 625 µs | Section III intro |
| Slow-time sample rate | 1 / `T_c` ≈ **1.6 kHz** | derived |
| Wavelength `λ` | ≈ 13.0 mm | derived |
| STFT window | **50 ms Hann** (≈ 80 samples) | Section II, step 3 |
| STFT hop | **1 sample** (≈ 98.75% overlap) | Section II, step 3 |
| Doppler resolution | **≈ 20 Hz** | derived (`1 / 50 ms`) |
| Doppler band (displayed) | **±500 Hz** | Figs. 3, 6, 8 |
| Torso Doppler band | 0 – 200 Hz (v ≤ 1 m/s) | Section III-B |
| Foot Doppler band | 200 – 500 Hz (v up to 3–4 m/s) | Section III-B |
| Clutter HPF cutoff | 10 Hz | Section II, step 1 |
| Radar geometry | torso at 1.0 m, feet at 0.15 m, subject 1 m away on a 3 m walking corridor | Section III, Fig. 4 |
| Beamwidth | 40° | Section II, Table II |

**Parameters the paper does NOT specify** (we set them ourselves; see `radar_params.py`):
- FFT size used in the STFT — verify per-file from `doppler_axis.size`
- Contrast-enhancement formula behind the `ce_*` arrays — the paper never describes a CE step; this is an addition to our dataset that the data owner should clarify
- SNR thresholds for trial validity — derive empirically from the dry-run + QC pass
- TUG duration in seconds — paper only says "10 reps at quick pace" without a duration figure
- Medication state for PD subjects — not stated

### Models
| Stage | Model | Purpose |
|---|---|---|
| Baseline | SVM + Random Forest | Fast, interpretable reference |
| Core | Custom CNN (3–4 layers) | Trained from scratch on spectrograms |
| Main | ResNet-18 fine-tuned | Transfer learning from ImageNet |
| Phase 2 | Transfer learning to prodromals | Future extension |

---

## 4. Work plan

### Week 1 — Data exploration and project foundation
- [x] **Project scaffold** at `thesis-parkinson-radar/` (2026-05-28; extended 2026-06-11)
  - **`src/`** library — 16 modules, all parse cleanly:
    - `config.py`, `data_loader.py`, `radar_params.py`, `qc.py` — foundation
    - `stats.py`, `eda.py`, `baseline.py`, `confound_analysis.py`, `diagnostics.py` — classical track (runnable, executed)
    - `preprocessing.py`, `channels.py`, `features.py`, `dataset.py`, `models.py`, `train.py`, `interpret.py` — deep track (implemented)
  - **`notebooks/`** — 10 Jupyter notebooks orchestrate the workflow (see list below)
  - `data/` symlink → `../shared_ziad`; `.gitignore`, `requirements.txt`, `README.md`
- [x] **Subject inventory** → `thesis-parkinson-radar/subjects.csv`
  - 58 rows: 33 control + 25 PD; columns ready for future age/sex/UPDRS/medication
  - Two anomalies auto-flagged: `fisp_022` (extra `test1/trial4`), `fisp_048` (missing `test2/trial2`)
- [x] **Reference-paper parameters captured** → `src/radar_params.py` (STFT window 50 ms, hop 1 sample, Doppler resolution ~20 Hz, band ±500 Hz, torso/foot bands)
- [ ] **Install Python env, launch Jupyter, run notebooks 01–03** (user action — Mac local)
  ```bash
  cd thesis-parkinson-radar
  conda create -n thesis python=3.10 -y && conda activate thesis
  pip install -r requirements.txt
  jupyter lab
  ```
- [ ] **Notebook 01 (data inspection)** — confirm axes look right (time on X, Doppler on Y); transpose in `data_loader._read_dataset` if not
- [ ] **Notebook 02 (dataset overview)** — review trial-duration histogram and Doppler-axis-size distribution; record any new anomalies
- [ ] **Notebook 03 (spectrogram EDA)** — visually compare control vs PD; review the PD−CTRL difference map
- [ ] **Email Ignacio López-Delgado / team** for: age/sex/UPDRS/medication-state table, known-bad files list, `.mat` generation script, and the contrast-enhancement formula behind `ce_*` (not described in the paper)
- [ ] **Decide whether `fisp_022/test1/trial4` is usable** — keep it (extra data) or exclude for consistency

### Notebook map (all under `thesis-parkinson-radar/notebooks/`)

| # | Notebook | Output | Timing |
|---|---|---|---|
| 01 | `01_data_inspection.ipynb` | Visual sanity check on one file | Week 1 Day 1 |
| 02 | `02_dataset_overview.ipynb` | `outputs/metrics/inventory.csv`, duration / Doppler-size distributions | Week 1 Day 2–3 |
| 03 | `03_eda_spectrograms.ipynb` | Sample grid, mean spectrogram per group, PD−CTRL difference map | Week 1 Day 3–4 |
| 04 | `04_quality_control.ipynb` | `outputs/metrics/qc_report.csv`, `excluded_files.csv` | Week 2 Day 1 |
| 05 | `05_feature_eda.ipynb` | `outputs/metrics/trial_features.csv`, Mann–Whitney U, PCA projection | Week 2 Day 2–3 |
| 06 | `06_preprocessing.ipynb` | `outputs/preprocessed/{*.npy, manifest.csv}` | Week 2 Day 4–5 |
| 07 | `07_baselines.ipynb` | `outputs/metrics/baseline_{svm,rf}_folds.csv`, ROC plot | Week 3 |
| 08 | `08_cnn.ipynb` | `outputs/metrics/cnn_folds.csv` (LOSO-CV) | Week 4 |
| 09 | `09_resnet.ipynb` | `outputs/metrics/resnet_folds.csv` | Week 4 |
| 10 | `10_gradcam.ipynb` | Grad-CAM figures under `outputs/figures/` | Week 6 |
| 11 | `11_data_forensics.ipynb` | Empirical answers to open questions (CE formula, anomalies, symmetry, fingerprint) → fills `thesis-parkinson-radar/FINDINGS.md` | Week 2 — pre-supervisor-email artefact |
| 12 | `12_eda_statistics.ipynb` | Rigorous EDA: MWU+FDR+Cliff's δ, duration-confound audit, correlation, PCA → `outputs/figures/eda_*.png`, `reports/eda_summary.json` | **executed 2026-06-11** |
| 13 | `13_baseline_models.ipynb` | Classical LOSO-CV baselines (LogReg/SVM/RF) on 5 feature sets incl. duration-only control → `baseline_*.png`, `reports/baseline_summary.json` | **executed 2026-06-11** |

> **Notebooks 12–13 wrap the runnable modules `src/eda.py` and `src/baseline.py`** (importable, headless-runnable: `.venv/bin/python -m src.eda`). Results are real, not placeholders — see `RESULTS.md`.

> **`FINDINGS.md`** at the project root is the structured placeholder for everything Notebook 11 produces. It contains `_TBD_` slots for each finding plus a draft email to the dataset owner with the same placeholders. Designed so a fresh Claude session can run the notebook and fill it in cold (instructions are at the top of the file).

### Week 2 — Data pipeline
- [ ] **`data_loader.py` — load a trial**
  - `load_trial(path) -> dict` returning `{ce_foot, ce_torso, t_axis, doppler_axis, param}`
  - Helper to iterate all `fisc_*` / `fisp_*` subjects → list of (subject_id, group, test, trial, path)
- [ ] **`preprocessing.py` — windowing and normalization**
  - Split time axis into 2–5 s windows with configurable stride
  - Per-window log-scale + Z-score normalization
  - Resize windows to fixed input size (e.g. 224×224 for ResNet-18)
- [ ] **Quality-control pass on all `.mat` files**
  - Try-load every file, log any that fail or have anomalous shape, NaNs, or near-zero SNR
  - Persist an `excluded_files.csv` and a one-paragraph QC report
- [ ] **Cache preprocessed windows**
  - Save windowed arrays + metadata as `.npy` + `.parquet` under `outputs/preprocessed/`
  - Training reads from cache, never re-opens 109 GB of `.mat` files
- [ ] **`dataset.py` — PyTorch `Dataset`** indexed by (subject_id, window_idx), returning `(tensor, label)`

### Week 3 — Baseline models  ✅ DONE (executed 2026-06-11, see `RESULTS.md`)
- [x] **Handcrafted features** (`features.py`) — duration-invariant set; cached in `outputs/metrics/trial_features.csv`
- [x] **Train LogReg / SVM / Random Forest with LOSO-CV** (`src/baseline.py`)
  - Subject-level LOSO (58 folds); trial probabilities averaged → one score per held-out subject
  - 5 feature sets compared incl. a **duration-only control** (the confound floor)
- [x] **Report AUC-ROC + bootstrap CI, F1, sensitivity, specificity, balanced acc** → `reports/baseline_summary.json`, `outputs/metrics/baseline_auc_grid.csv`
- [x] **Sanity checks** — label-permutation null (p-value); LOSO guarantees no subject overlap
- **Result:** raw subject-level AUC ≈ **0.62** (`shape_clean`) vs 0.61 duration-only floor. **Confound-control follow-ups** (`src/confound_analysis.py`, executed) show the signal is **weak but real, not just duration**: duration-matched subset (post-match p=0.95) holds at 0.62–0.64; duration-residualized features rise to 0.66–0.68; test2-only clean features (0.589) beat the test2 duration floor (0.536). This is the honest floor the deep models must beat; it still strongly motivates the windowed CNN. **Diagnostics** (`src/diagnostics.py`) further show: errors are group-balanced (11 control / 9 PD of 20), the matched result is caliper-robust (AUC 0.61–0.62 across 0.25–2.0 s), and the learning curve is still rising at n=50 (+0.13 from n=10) → more subjects would help. Full discussion in `RESULTS.md` §3b–3c.

### Week 4 — Deep learning models
- [ ] **Custom CNN from scratch** (`models.py:CNN`)
  - 3–4 conv blocks + global average pool + binary head; small enough to train on Mac
- [ ] **ResNet-18 with ImageNet pretraining** (`models.py:ResNet`)
  - Replicate 2-channel input to 3 channels (or adapt first conv); freeze early blocks, fine-tune later ones
- [ ] **Training loop with LOSO-CV** (`train.py`)
  - Mixed precision, early stopping per fold, per-fold checkpoint and metric logging
  - If LOSO is too slow on Mac, switch to GAPS workstation via SSH (see Section 6 warning)
- [ ] **Hyperparameter tuning**
  - Learning rate, weight decay, window length, augmentation strength

### Week 5 — Ablations
- [ ] **Channel arrangement:** foot-only vs torso-only vs 2-channel stack vs custom recombination (using `all_doppler_time` + `idx_*_max_snr`)
- [ ] **Input representation:** `ce_*` vs `|stft_*|` vs log-magnitude STFT
- [ ] **Window length:** 2 s vs 3 s vs 5 s
- [ ] **Test protocol effect:** test1-only vs test2-only vs both (does the chair / no-chair start change discrimination?)
- [ ] **Demographic subsets:** if age table available, repeat on age-matched subgroup to control for age as confound

### Week 6 — Interpretation and results consolidation
- [ ] **Grad-CAM on best model** (`interpret.py`)
  - Aggregate maps across true positives to reveal discriminating Doppler/time regions
- [ ] **Compare against clinical literature** — slower cadence, reduced toe-clearance, asymmetry: do the heatmaps align?
- [ ] **Consolidate final tables and figures**
  - LOSO-CV metric tables (per model, per ablation), ROC curves, Grad-CAM examples

### Week 7 — Thesis draft (part 1)
- [ ] Introduction — motivation, clinical context, radar background, contribution statement
- [ ] Methodology — pipeline, models, evaluation protocol, dataset description
- [ ] Results — tables and figures from weeks 3–6

### Week 8 — Thesis draft (part 2) and review
- [ ] Discussion — limitations (subject count, single-site, dataset cleaning caveat), comparison to wearable/camera approaches
- [ ] Conclusions and future work — prodromal extension (Phase 2), multi-site validation, longitudinal monitoring
- [ ] Final review with supervisor; revisions

---

## 5. Open Questions

### Resolved by `shared_ziad/readme.txt`
- ~~`fisc_XXX` → group mapping~~ → `fisc` = control, `fisp` = PD, `fis` = prodromal (not uploaded)
- ~~PD patient count~~ → **26** (`fisp_012`–`fisp_052`); plus 33 controls
- ~~`stft_data.mat` contents~~ → see variable table in Section 2; foot/torso STFTs are pre-combined, with `ce_*` magnitude versions also provided
- ~~`test1` vs `test2`~~ → TUG with chair vs TUG without chair, each repeated 3 times (`trial1`–`trial3`)

### Resolved by reference-paper mining (2026-05-28)
- ~~STFT parameters~~ → 50 ms Hann window, 1-sample hop, ~1.6 kHz slow-time rate, ~20 Hz Doppler resolution. Captured in `src/radar_params.py`.
- ~~Doppler band of interest~~ → ±500 Hz displayed; torso ≤ 200 Hz; feet 200–500 Hz.

### Resolved by Notebook 11 forensics (executed 2026-05-29)
- ~~FFT size used in the STFT~~ → 320 bins everywhere (Notebook 11 § 2, zero outliers across 348 files)
- ~~Known bad files~~ → Notebook 04 produces `excluded_files.csv` from the QC pass
- ~~Doppler symmetry~~ → median ratio 1.005, direction folded
- ~~Subject fingerprint vs noise~~ → strong fingerprint, LOSO essential (median within/between ratio 0.374)
- ~~Duration-confound source~~ → concentrated in test1 chair-stand (+10.1%); test2 only +3.4%
- ~~Hidden variables / param contents~~ → all 16 variables in all files; `param` confirms paper RF setup with `BW=1.38 GHz` / `f₀=23.5 GHz` (slight rounding differences); two undocumented variables found (`all_fig_savename`, `all_fig_caption`)
- Partial — `fisp_022/test1/trial4` (clean retry, recommend keep) and `fisp_048/test2/trial2` (clean deletion, proceed with 5 files): characterised; team still asked to confirm intent
- Partial — `ce_*` formula: best element-wise fit is dB but R² 0.46–0.95 — not a global function; adaptive per-trial. Awaits Ignacio's confirmation of the actual operation.

### Resolved by second PDF re-scan (2026-05-29 evening)
- **Confirmed**: the paper does **not** describe any contrast-enhancement / log / dB / normalisation / equalisation step (0 hits in exhaustive grep). `ce_*` is a dataset-specific addition, not a paper-defined step.
- **New nugget**: per Fig. 6 caption, "*prior to the combination, the Doppler-time matrix of node 2 is flipped with respect to the Doppler axis*" — a deliberate operation that collapses motion direction in the combined torso signal. Notebook 11 § 6 verifies this empirically.

### Still open (for 4-person meeting / dataset owner)
1. **Age / demographic table** — each subject's age, sex, and (for PD) UPDRS score and time-since-diagnosis. Needed for any age-matched analysis and for the G1 (young) vs G2 (older) control split if that remains in scope.
2. **Medication state at recording** — were PD subjects ON or OFF medication? Not stated in the paper either. Big confound for gait features.
3. **Contrast-enhancement formula behind `ce_*`** — now confirmed to be entirely outside the paper. Notebook 11 § 1 will propose a best-fit element-wise formula (or report it's adaptive); Ignacio needs to confirm.
4. **Generation script** — is the MATLAB/Python script that produced these `.mat` files available?
5. **Two anomalies in v1 release** — `fisp_022/test1/trial4` (extra) and `fisp_048/test2/trial2` (missing): Notebook 11 characterises both; team still needs to confirm intent (duplicate/retry/error).
6. **Prodromal data timing** — when will `fis_*` subjects be uploaded? Affects whether the transfer-learning phase fits the 8-week timeline.

---

## 6. Local Environment Setup (Mac)

All development runs locally on Mac. The full 109 GB dataset is already downloaded.

### Python environment

```bash
# Create a dedicated conda environment
conda create -n thesis python=3.10
conda activate thesis

# Install dependencies
pip install torch torchvision scipy numpy matplotlib scikit-learn h5py jupyter
```

### First step — inspect a .mat file

The dataset's `.mat` files are MATLAB v5, not v7.3. Use `scipy.io.loadmat`
(no `h5py` needed; arrays come out as `(doppler, time)` directly).

```python
import scipy.io as sio
import numpy as np

path = '/Users/ziad.boussedra/Desktop/Master Thesis/shared_ziad/fisc_005/test1/trial1/stft_data.mat'

mat = sio.loadmat(path, squeeze_me=True)
for key, value in mat.items():
    if key.startswith('__'):
        continue
    if hasattr(value, 'shape'):
        print(f"  {key:22s}  shape={value.shape}  dtype={value.dtype}")
    else:
        print(f"  {key:22s}  {value}")

ce_foot  = mat['ce_foot']          # (doppler, time), real, magnitude after CE
ce_torso = mat['ce_torso']         # (doppler, time), real
t_axis   = mat['t_axis_target']    # 1D, seconds
doppler  = mat['doppler_axis']     # 1D, signed Hz; 320 bins over ±800 Hz

print(f"ce_foot:  {ce_foot.shape}  doppler bin spacing ≈ {np.median(np.diff(doppler)):.1f} Hz")
```

The project's `src.data_loader.load_trial(path)` wraps this and returns
a clean dict with `t_axis`, `doppler`, `ce_foot`, `ce_torso`.

### Project structure (recommended)

```
thesis-parkinson-radar/
├── data/                   # Symlink to shared_ziad/ (never committed)
├── src/                    # Library code (importable; modules are headless-runnable)
│   ├── config.py           # Central paths, feature groupings, experiment defaults
│   ├── data_loader.py      # Load .mat files (scipy v5); iterate subjects/tests/trials
│   ├── radar_params.py     # Reference-paper STFT/Doppler constants
│   ├── qc.py               # Quality-control checks
│   ├── stats.py            # MWU + BH-FDR, Cliff's delta, subject-bootstrap CIs
│   ├── eda.py              # EDA figures + statistics  (python -m src.eda)
│   ├── preprocessing.py    # Windowing, normalization, resizing → .npy cache
│   ├── channels.py         # Foot/torso channel arrangement
│   ├── features.py         # Handcrafted features (duration-invariant)
│   ├── baseline.py         # Classical LOSO-CV ML  (python -m src.baseline)
│   ├── dataset.py          # PyTorch Dataset over preprocessed .npy windows
│   ├── models.py           # SmallCNN + ResNet-18 fine-tune
│   ├── train.py            # Deep LOSO-CV training loop
│   └── interpret.py        # Grad-CAM
├── notebooks/              # 01–13 — see notebook map in Section 4
├── outputs/
│   ├── preprocessed/       # Cached windowed .npy + manifest.csv (deep track)
│   ├── models/             # Saved checkpoints (per LOSO fold)
│   ├── figures/            # eda_*.png, baseline_*.png, Grad-CAM maps
│   └── metrics/            # trial_features.csv, baseline_auc_grid.csv, …
├── reports/                # Generated summaries: eda_summary.json, baseline_summary.json
├── .venv/                  # Lightweight analysis env (gitignored; no torch)
├── subjects.csv            # subject_id, group, age, sex, notes
├── excluded_files.csv      # Output of QC pass (Notebook 04)
├── RESULTS.md              # Real results + thesis-direction decision
├── FINDINGS.md             # Notebook-11 forensics + draft email to data owner
├── requirements.txt        # Full stack incl. torch (deep track, conda env)
├── requirements-analysis.txt  # Lightweight stack (classical track, .venv)
├── .gitignore
└── README.md
```

**Two execution tracks.** The *classical track* (EDA + feature baselines) runs
in the local `.venv` with no torch and no `.mat` access — it works off the
cached `trial_features.csv`, so it is fully reproducible on any laptop. The
*deep track* (windowing → CNN/ResNet → Grad-CAM) needs the conda `thesis` env
with torch and the 106 GB dataset, ideally on the GAPS GPU workstation.

### .gitignore — keep data off GitHub

```
# Data — never commit
*.mat
data/
outputs/models/

# Python
__pycache__/
*.pyc
*.pyo
.DS_Store
venv/
.env

# Jupyter
.ipynb_checkpoints/
```

### Memory management tips for large .mat files

Since files are ~245 MB each and the full dataset is 109 GB, never load all subjects at once:

```python
# Load one trial at a time, extract what you need, release memory
import gc
import h5py
import numpy as np

def load_trial(filepath):
    """Return the two contrast-enhanced spectrograms and the axes."""
    with h5py.File(filepath, 'r') as f:
        return {
            'ce_foot':  np.array(f['ce_foot']),
            'ce_torso': np.array(f['ce_torso']),
            't_axis':   np.array(f['t_axis_target']).squeeze(),
            'doppler':  np.array(f['doppler_axis']).squeeze(),
        }

# Process trial by trial — never hold more than one in memory
for trial_path in all_trial_paths:
    trial = load_trial(trial_path)
    # window, normalize, save preprocessed .npy under outputs/preprocessed/
    del trial
    gc.collect()
```

Pre-process and save windowed spectrograms as smaller `.npy` files once — then train from those instead of reloading `.mat` files every run. This will make iteration much faster.

---

> ⚠️ **If your Mac gets stuck during full training:**
> The BYO/GAPS lab has GPU workstations available remotely via SSH (credentials from Camilo: `zboussedra@gaps_domain.ssr.upm.es`). Connect via UPM VPN first (`UPMvpn` — download from `upm.es/UPM/ServiciosTecnologicos/vpn`). Reserve a machine via Google Calendar before use (ask Amber). Best option: WS3 or WS4 (RTX 3090, 64–128 GB RAM).

---

## 7. Key Contacts

| Person | Role | Contact |
|---|---|---|
| Juan Ignacio Godino Llorente (Nacho) | Thesis supervisor | `juiggo@gmail.com` |
| Ignacio Esteban López-Delgado (Nacho) | Dataset owner / radar hardware | `ie.lopez@upm.es` |
| Jesús Grajal | Research group PI | `jesus.grajal@upm.es` |
| Camilo | Lab researcher / server account | GAPS lab |
| Amber | Lab researcher / calendar reservations | GAPS lab |

---

*Last updated: 2026-06-11 (pass 3, Fable 5) — added `src/diagnostics.py` (executed): per-subject error analysis (66% acc, errors group-balanced 11 ctrl/9 PD), caliper-robustness sweep (matched AUC 0.61–0.62 stable across 0.25–2.0 s → not a caliper fluke), and a subject-level learning curve (AUC 0.58→0.72 from n=10→50, still rising → data-limited, more subjects help). `RESULTS.md` §3c. Pass 2 (same day) — added `src/confound_analysis.py` and ran the decisive confound-control tests the first pass deferred: per-test split, duration residualization, and duration-matched (1:1 caliper) subset. **Revised headline:** the classical signal is weak but **genuinely gait, not just trial length** — it survives duration matching (AUC 0.62–0.64 at post-match p=0.95) and rises under residualization (0.66–0.68). The residualized `shape_clean` model is **permutation-significant: AUC 0.676, p=0.022** (n=1000), vs the raw-feature p=0.092 — i.e. removing duration *unmasks* a real signal, flipping the pass-1 "~chance" reading. `RESULTS.md` §3b has the full table. Pass 1 (same day) added the classical track — `config/stats/eda/baseline` + notebooks 12–13, EDA (MWU+FDR+Cliff's δ, PCA) and LOSO baselines in a torch-free `.venv` — and re-read the full reference paper (no classifier; torso↔heel-strike coupling breaks down in PD = mechanistic basis). Prior 2026-05-29 Notebook-11 forensics + FINDINGS.md email still stand.*

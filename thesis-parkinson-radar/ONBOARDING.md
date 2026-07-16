# Onboarding — Parkinson's Detection from Radar Gait

Read this first. It explains what the project is, the science behind it, the
data, the code, what's been done, and what to do next. Everything here is
grounded in the current files (not aspirational).

---

## 1. The one-paragraph summary

We are building a machine-learning system that decides **"Parkinson's disease
(PD) vs healthy control"** from **radar micro-Doppler spectrograms of a person
walking**. A UPM lab already built and validated the radar hardware and produced
the spectrograms; **our thesis contribution is the classifier on top** — the
lab's paper measures gait *parameters* but never classifies disease. The work so
far has cleaned/understood the data, run a rigorous classical-ML baseline (it
finds a weak-but-real PD signal), and scaffolded the deep-learning pipeline
(CNN/ResNet) that is the intended core model.

---

## 2. Why this matters (clinical motivation)

- PD screening today = a specialist subjectively scoring gait (e.g. the
  Timed-Up-and-Go test) in a clinic. Slow, subjective, not scalable.
- Radar is **contactless, camera-free (privacy), wearable-free, continuous**. A
  radar box in a hallway at home could flag gait changes years earlier than a
  clinic visit.
- If a model can read PD-specific gait from radar, it enables passive, at-home,
  privacy-preserving screening. That is the vision the thesis serves.

---

## 3. The reference paper (read it once, don't over-index on it)

**"Radar Network for Gait Monitoring: Technology and Validation"** — López-Delgado
et al., *IEEE TBME*, Jan 2026. PDF at repo root:
`../Radar_Network_for_Gait_Monitoring_Technology_and_Validation (1).pdf`.

What it actually is:
- A **hardware + signal-processing validation** paper. They built a 24 GHz FMCW
  radar network, compared 6 configurations and 4 algorithms against a Vicon
  motion-capture ground truth, and showed which setup best extracts **gait
  parameters** (stride/step/stance/swing/double-support times, stride/step
  distance & velocity, foot max velocity).
- **Best configuration = C5: two feet radars + one torso radar.** Feet radars are
  best for *temporal* parameters (esp. in PD/impaired subjects); torso radars are
  best for *spatial* parameters.
- Cohort in the paper: **8 subjects** (3 young healthy G1, 2 older healthy G2, 3
  PD G3). Note: **this is NOT our dataset** — ours is a separate, larger release.

What it does **not** do (→ our contribution):
- No disease classification. No ML classifier at all. We add that.

The single most useful scientific nugget for us: the paper reports that the
**torso-velocity ↔ heel-strike coupling that holds in healthy people breaks down
in PD subjects**. That is a mechanistic reason a classifier *should* be able to
separate PD from control using foot/torso Doppler structure. Also: PD-relevant
markers it highlights are **foot maximum velocity** and **gait asymmetry**.

Key radar/STFT numbers (captured in `src/radar_params.py`): 23 GHz carrier,
1.4 GHz bandwidth, 625 µs chirp, 50 ms Hann STFT window with 1-sample hop,
~20 Hz intrinsic Doppler resolution. (The dataset files store a 4× zero-padded
Doppler axis → 5 Hz bin spacing; see §5.)

---

## 4. The dataset

Location: `../shared_ziad/` (a 106 GB local copy; `data/` in the repo is a
symlink to it). Authoritative description: `../shared_ziad/readme.txt`.

**Subjects (58 total):**
| Group | Folder prefix | Count |
|---|---|---|
| Healthy controls | `fisc_*` | 33 |
| Parkinson's patients | `fisp_*` | 25 |
| Prodromal (pre-symptomatic) | `fis_*` | 0 — Phase 2, not uploaded yet |

**Structure per subject:** 2 tests × 3 trials = 6 recordings.
- `test1` = Timed-Up-and-Go **with** a chair (stand → walk → turn → walk → sit).
- `test2` = TUG **without** a chair (start/end standing).
- `trial1/2/3` = three repeats of each.
- Each trial folder has one `stft_data.mat` (~250 MB). **348 files total.**

**Two known anomalies** (both benign, confirmed in Notebook 11): `fisp_022` has an
extra `test1/trial4` (a clean retry — keep); `fisp_048` is missing `test2/trial2`
(a clean deletion — proceed with 5 files).

**What's inside each `.mat`** (from readme + our forensics):
- `ce_foot`, `ce_torso` — **contrast-enhanced magnitude spectrograms**
  (Doppler × time), foot radars and torso radars already SNR-combined. **These
  are our default model input.**
- `stft_foot`, `stft_torso` — the complex STFTs (pre-enhancement).
- `all_doppler_time`, `all_ce_time` — per-radar versions (for custom fusion).
- `t_axis_target`, `doppler_axis` — the axes.
- `param` — capture config (confirms 23.5 GHz / 1.38 GHz actual capture, date).

**No demographics in the data.** Age, sex, UPDRS severity, and PD medication
state are NOT in the files — they must be requested from the lab (see §9).

---

## 5. Hard-won facts about the data (don't relearn these the hard way)

These were discovered empirically and are baked into the code + memory:

1. **`.mat` files are MATLAB v5, not v7.3.** Use `scipy.io.loadmat`, not `h5py`.
   Arrays come out already as `(doppler, time)` — no transpose.
2. **Doppler axis = 320 bins over ±800 Hz → 5 Hz spacing** (a 4× zero-padded
   FFT; finer than the paper's 20 Hz intrinsic resolution, no new info). Uniform
   across all 348 files.
3. **Duration confound (the big one).** PD trials run **longer** than control
   trials — significantly in test1 (+10%, chair stand-up), weakly in test2
   (+3%). Any feature that sums energy over time leaks trial length and looks
   "discriminative" for the wrong reason. This shaped the whole analysis design.
4. **`ce_*` contrast-enhancement is NOT in the paper.** It's a dataset-specific
   step; reverse-engineering (Notebook 11 §1) shows it's an **adaptive per-trial**
   normalisation (a dB-like stage + rescale), not a single global formula. Ask
   the lab for the exact operation.
5. **Spectrograms are Doppler-symmetric** (median +/− energy ratio ≈ 1.005) —
   motion direction is folded out, so you can halve the Doppler axis as an
   ablation.
6. **Strong subject fingerprint** (within-subject feature distance ≈ 0.37× the
   between-subject distance). → **Leave-One-Subject-Out CV is mandatory**;
   k-fold that mixes a subject's trials across train/test would massively
   over-estimate performance.

---

## 6. Project structure

```
Master Thesis/
├── shared_ziad/                     # the 106 GB dataset + readme.txt
├── Radar_Network_..._Validation.pdf # the reference paper
├── Thesis_Proposal_Summary_....pdf  # 1-page abstract
├── project_summary_and_setup.md     # the master plan (methodology, work plan, findings)
├── academic-research-skills/        # cloned Claude-Code plugin (paper-writing skills)
└── thesis-parkinson-radar/          # ← THE CODE PROJECT
    ├── ONBOARDING.md    ← you are here
    ├── RESULTS.md       ← real results + thesis-direction decision (read 2nd)
    ├── FINDINGS.md      ← data forensics + draft email to the data owner
    ├── README.md        ← repo usage / two-track setup
    ├── data/            → symlink to ../shared_ziad
    ├── src/             # 16 library modules (see below)
    ├── notebooks/       # 01–13 (see below)
    ├── outputs/         # figures/, metrics/, models/, preprocessed/
    ├── reports/         # machine-readable JSON summaries of each analysis
    ├── subjects.csv     # 58 subjects; age/sex/updrs columns awaiting the lab
    ├── .venv/           # lightweight analysis env (no torch) — classical track
    ├── requirements.txt / requirements-analysis.txt
    └── excluded_files.csv
```

### `src/` — two tracks

**Foundation:** `config.py` (paths, feature groups), `data_loader.py`
(load `.mat`, iterate trials), `radar_params.py` (paper constants), `qc.py`
(quality control).

**Classical track (built + executed, runs on any laptop, no GPU/data):**
`stats.py`, `eda.py`, `baseline.py`, `confound_analysis.py`, `diagnostics.py`.
Works off the cached `outputs/metrics/trial_features.csv`.

**Deep track (implemented, NOT yet run — needs torch + GPU + the 106 GB):**
`preprocessing.py` (windowing → `.npy` cache), `channels.py`, `features.py`,
`dataset.py` (PyTorch), `models.py` (SmallCNN + ResNet-18), `train.py`
(LOSO-CV loop), `interpret.py` (Grad-CAM).

### `notebooks/` — analysis, numbered in workflow order
01 inspect · 02 dataset overview · 03 spectrogram EDA · 04 QC · 05 feature
extraction · 06 preprocessing (deep) · 07 window baselines (legacy) · 08 CNN ·
09 ResNet · 10 Grad-CAM · 11 data forensics · **12 EDA+stats** · **13 baselines**.
(11–13 are executed and wrap the `src` modules; 06/08/09/10 await the GPU run.)

---

## 7. What's been done vs what remains

| Phase | Status |
|---|---|
| Understand data, QC, fix format/confound gotchas | ✅ done |
| Rigorous EDA + statistics (MWU/FDR/Cliff's δ, PCA, duration audit) | ✅ done |
| Classical baselines (LogReg/SVM/RF, subject-LOSO, permutation, bootstrap) | ✅ done |
| Confound control (per-test, duration-residualized, duration-matched) | ✅ done |
| Diagnostics (per-subject errors, caliper robustness, learning curve) | ✅ done |
| **Windowed `.npy` cache** (preprocessing over all `.mat`) | ⏳ not run — needs data + ~30 min |
| **CNN / ResNet training** (LOSO) | ⏳ not run — needs torch + GPU |
| **Grad-CAM interpretation** | ⏳ not run — after models |
| Demographics-controlled (age-matched) analysis | ⏳ blocked on lab data |
| Thesis writing | ⏳ not started (skills cloned for it) |

---

## 8. Results so far (the honest story)

All from the classical track, subject-level LOSO-CV (full detail in `RESULTS.md`):

- **Raw** trial-feature AUC ≈ **0.62**; a **duration-only** model gives 0.61.
  Taken alone this looks like "≈ chance" (permutation p = 0.092).
- **But confound control shows the signal is weak-yet-real, not just duration:**
  - Duration-**matched** subset (groups made duration-identical, p=0.95):
    AUC **holds at 0.62–0.64** — doesn't collapse to chance.
  - Duration-**residualized** features: AUC **rises to 0.68**, permutation
    **p = 0.022 (significant)**. Removing duration *unmasks* real signal.
  - **test2-only** (no chair confound): clean features 0.589 vs duration floor
    0.536 — signal above the floor.
  - Errors are **group-balanced** (11 control / 9 PD of 20 wrong).
  - Learning curve **still rising at n=50** → more subjects would help.

**Takeaway:** handcrafted features carry a small but genuine PD signal. The
windowed CNN is expected to do better for two concrete reasons: (1) fixed-length
windows remove the duration confound **by construction**, and (2) a 2-D CNN sees
micro-Doppler *shape* (heel-strike minima, foot-max velocity, foot/torso
asymmetry) that scalar summaries discard.

---

## 9. Open questions for the lab (draft email in `FINDINGS.md`)

Contact: **Ignacio López-Delgado** (`ie.lopez@upm.es`), dataset owner.
Supervisor: **Juan Ignacio Godino-Llorente "Nacho"**.

1. **Age / sex / UPDRS / medication-state table** per subject — needed to rule
   out "PD vs older" being an age effect, and to interpret severity. *(Highest
   priority — gates the credible version of every result.)*
2. **The `ce_*` contrast-enhancement formula / script.**
3. Confirm the two file anomalies (`fisp_022` extra, `fisp_048` missing).
4. When does the **prodromal (`fis_*`) Phase-2 data** arrive?

---

## 10. How to run things

**Classical track (works now, no GPU):**
```bash
cd thesis-parkinson-radar
python3 -m venv .venv && .venv/bin/pip install -r requirements-analysis.txt
.venv/bin/python -m src.eda                # EDA + statistics
.venv/bin/python -m src.baseline           # LOSO baselines + permutation
.venv/bin/python -m src.confound_analysis  # per-test / residualized / matched
.venv/bin/python -m src.diagnostics        # per-subject / caliper / learning curve
```
Outputs → `outputs/figures/*.png`, `outputs/metrics/*.csv`, `reports/*.json`.

**Deep track (needs the conda env with torch + the dataset, ideally GAPS GPU):**
```bash
conda create -n thesis python=3.10 -y && conda activate thesis
pip install -r requirements.txt
jupyter lab      # run notebooks 06 → 08 → 09 → 10
```

**Golden rule for the deep track:** re-run the same confound checks
(duration-matched + test2-only) on the CNN, not just the baseline — that's how
you prove the model learns gait, not trial length.

---

## 11. Where to read next

1. `RESULTS.md` — the results and the recommended thesis direction.
2. `project_summary_and_setup.md` (repo root) — full methodology + week-by-week
   work plan + the empirical-findings table.
3. `FINDINGS.md` — data forensics + the ready-to-send email to the lab.
4. `src/config.py` and `src/radar_params.py` — the two files that define the
   project's constants and conventions.

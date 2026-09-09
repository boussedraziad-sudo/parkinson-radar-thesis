# Results & Thesis Direction

*Generated 2026-06-11 from real analysis runs on the cached trial feature table
(`outputs/metrics/trial_features.csv`, 348 trials / 58 subjects). Reproduce with
`.venv/bin/python -m src.eda` and `.venv/bin/python -m src.baseline`. Figures in
`outputs/figures/eda_*.png` and `baseline_*.png`; machine-readable summaries in
`reports/`.*

---

## 1. What was actually run

| Stage | Module | Status |
|---|---|---|
| EDA + statistics | `src/eda.py`, `src/stats.py` | ✅ executed, 7 figures |
| Classical baselines (LOSO-CV) | `src/baseline.py` | ✅ executed, AUC grid + ROC + permutation |
| Confound-control (per-test / residualized / matched) | `src/confound_analysis.py` | ✅ executed, 4 figures |
| Diagnostics (per-subject / caliper sweep / learning curve) | `src/diagnostics.py` | ✅ executed, 3 figures |
| Windowed `.npy` cache | `src/preprocessing.py` | ⏳ not run (needs full `.mat` pass) |
| CNN / ResNet (LOSO) | `src/models.py`, `src/train.py` | ⏳ not run (needs torch + GPU) |

The classical track runs end-to-end on this laptop in a lightweight venv
(numpy/pandas/scipy/scikit-learn/statsmodels — no torch, no 106 GB access).
The deep track needs the conda `thesis` env with torch and, realistically, the
GAPS GPU workstation.

---

## 2. Headline finding — the duration confound dominates the trial-level signal

PD trials are systematically **longer** than control trials, and the gap is
concentrated in test1 (the chair stand-up TUG):

| Test | PD − control median duration | Mann–Whitney p | Cliff's δ |
|---|---|---|---|
| test1 (chair TUG) | **+10.1 %** | 0.0016 | 0.28 (small–medium) |
| test2 (no chair) | +3.4 % | 0.14 (n.s.) | 0.13 (negligible) |

Consequently, the trial-level features that *look* most discriminative are the
duration-confounded energy sums:

| Feature | Cliff's δ (PD−ctrl) | FDR p | Confounded? |
|---|---|---|---|
| torso_torso_band_energy | +0.46 (medium) | 4.5e-12 | **yes** |
| foot_total_energy | +0.45 (medium) | 6.2e-12 | **yes** |
| torso_total_energy | +0.44 (medium) | 1.6e-11 | **yes** |
| foot_foot_band_energy | +0.43 (medium) | 3.7e-11 | **yes** |

24 of 25 features are "significant" after BH-FDR — but significance here is
mostly trial-length leakage, not gait. A duration-leakage audit shows even
several *nominally* duration-invariant features correlate with trial length
(e.g. `torso_std` r=−0.65, `foot_mean` r=−0.42) because longer PD trials
contain more low-velocity standing/turning frames that shift per-pixel stats.

---

## 3. Classical baseline — honest numbers (subject-level LOSO-CV)

Leave-one-subject-out, trial probabilities averaged to one score per held-out
subject, Logistic Regression (balanced):

| Feature set | # feats | Subject AUC | 95% CI (subj bootstrap) | Sens | Spec |
|---|---|---|---|---|---|
| confounded_only | 8 | 0.675 | [0.53, 0.81] | 0.56 | 0.64 |
| all_features | 25 | 0.633 | [0.49, 0.78] | 0.60 | 0.64 |
| duration_invariant | 17 | 0.627 | [0.48, 0.78] | 0.64 | 0.61 |
| **shape_clean** (|r_dur|<0.3) | 10 | **0.621** | [0.47, 0.77] | 0.64 | 0.67 |
| **duration_only** (control) | 1 | **0.610** | [0.45, 0.75] | 0.40 | 0.70 |

**Reading it:** trial length *alone* gives AUC 0.61. The confound-controlled
gait features (`shape_clean`) reach only 0.62 — i.e. the honest, non-duration
gait signal in these handcrafted summaries is **marginal** (~0.01 AUC above the
duration floor, CI spans 0.5). RBF-SVM and Random Forest are no better
(0.58–0.63). A 1000-shuffle label-permutation test on the headline model
(`logreg | shape_clean`) gives **p = 0.092** — i.e. the confound-controlled
classical signal is **not significantly above chance** at α=0.05.

This raw number on its own *looks* like a near-negative result — but the
permutation was run on features still entangled with duration. The
confound-control follow-ups (next section) show that is misleading.

---

## 3b. Confound-control tests — the signal is weak but REAL

`src/confound_analysis.py` (run: `.venv/bin/python -m src.confound_analysis`)
runs three independent ways of separating gait from trial length. They agree:
once duration is properly controlled, a modest signal **survives**, and the raw
permutation p actually *under-states* it because duration partly suppresses the
gait features.

**(a) Per-test split** — the confound lives in test1 (chair stand-up):

| feature set | both | test1 | test2 |
|---|---|---|---|
| duration_only | 0.610 | **0.646** | **0.536** |
| shape_clean | 0.621 | 0.628 | **0.589** |

In test2 (no chair), duration alone is essentially chance (0.536) yet the clean
gait features still reach 0.589 — i.e. that ~0.05 is real, non-duration signal.

**(b) Duration residualization** — linearly regress every feature on duration,
classify on the residuals:

| feature set | raw AUC | residualized AUC |
|---|---|---|
| shape_clean | 0.621 | **0.676** |
| duration_invariant | 0.627 | 0.663 |
| all_features | 0.633 | 0.642 |

Removing duration **raises** AUC — duration was acting as a partial suppressor,
not the source of the signal.

**(c) Duration-matched subset** — greedy 1:1 caliper matching (270 trials, 58
subjects) makes the groups duration-identical (post-match Mann–Whitney
**p = 0.95**). The baseline does **not** collapse:

| feature set | matched AUC |
|---|---|
| duration_invariant | 0.638 |
| shape_clean | 0.624 |
| all_features | 0.632 |

If the apparent signal were pure duration leakage, AUC here would fall to ~0.5.
It holds at ~0.62–0.64.

**Permutation on the residualized model — the capstone.** A 1000-shuffle
label-permutation on `logreg | shape_clean | duration-residualized` gives
observed AUC **0.676, p = 0.022** — **significant at α=0.05**. Contrast with the
raw-feature permutation (p = 0.092, n.s.): once duration is removed, the gait
signal *is* significantly better than chance. The confound was masking real
signal, not creating it. (`reports/confound_permutation.json`.)

**Feature importance** (`confound_feature_importance.png`): in-sample RF
permutation importance is ≈0 for every feature — expected for an overfit RF on
correlated features; it confirms the signal is weak and *distributed*, not
carried by one feature. Lean on the matched/residualized evidence, not this.

**Revised conclusion.** There is a modest but **statistically genuine**
PD-vs-control gait signal in the trial-level features — duration-residualized
AUC 0.676, permutation **p = 0.022** — not merely trial length. It is weak at the
handcrafted-feature level, which is exactly the gap the windowed CNN should
close. The earlier "p = 0.092, ~chance" reading was an artifact of testing
duration-entangled features; proper confound control flips it to significant.

## 3bb. 🔴 CRITICAL — an acquisition-batch confound invalidates the "weak but real" reading

*Discovered and independently verified 2026-07-29. This supersedes the optimistic
conclusion in §3b. Batch labels cached in `outputs/metrics/acquisition_batch.csv`.*

**What was found.** The 348 `.mat` files contain **exactly two distinct Doppler
axes**, differing in the 4th decimal place:

| Batch | `doppler_min` | Trials | Subjects |
|---|---|---|---|
| **A** | −794.8713989257812 | 203 | 34 |
| **B** | −794.872802734375 | 145 | 24 |

This is an **acquisition/session signature** — a different processing or hardware
configuration. **All 58 subjects are pure with respect to batch** (every trial of
a subject comes from the same batch), so batch is a *subject-level* property.

**Why it is fatal to the current headline.** Batch is confounded with diagnosis:

| Batch | control | PD | % PD |
|---|---|---|---|
| A | 24 | 10 | **29.4 %** |
| B | 9 | 15 | **62.5 %** |

> ### 🔑 **Batch membership ALONE gives subject-level AUC = 0.664**
> That is **higher than every feature-based model in the entire baseline grid** —
> higher than the 0.621 `shape_clean` headline and the 0.610 `duration_only` floor.

**The decisive test — re-run the exact LOSO baseline *within* each batch:**

| Feature set | pooled | batch A | batch B |
|---|---|---|---|
| `all_features` | 0.633 | 0.604 | **0.467** |
| `duration_invariant` | 0.627 | 0.567 | **0.496** |
| **`shape_clean`** | **0.621** | **0.537** | **0.541** |
| `duration_only` | 0.610 | **0.642** | **0.659** |

**The spectrogram-derived gait features collapse to chance inside a batch.** The
apparent "weak but real gait signal" of §3b was, to a large extent, the model
detecting **which acquisition batch a subject came from**.

**The one thing that survives — and it matters.** `duration_only` does *not*
collapse; it **strengthens** within batch (0.642 / 0.659), and PD trials are
longer inside *both* batches (A: 8.93→9.74 s; B: 7.81→8.91 s). So **walking
slowly is a genuine, batch-robust PD marker** — arguably bradykinesia — while the
handcrafted *spectral-shape* features are not.

**Why no existing control caught it.** Duration matching, duration
residualization and the test2-only split **all leave batch fully intact**. Batch
is orthogonal to every confound tested so far.

**It also explains the error pattern.** The classifier's prediction matches batch
membership for ~55/58 subjects; all 9 batch-B controls are misclassified as PD,
and 9 of the 10 batch-A PD subjects are misclassified as control. That is why
§3c's errors are "confident" rather than borderline, and why the subject-score
distribution is bimodal — the bimodality in `eda_pca.png` (PC1 = 62 % of variance)
**is the batch split**, not the disease.

### Honest caveats
- **Within-batch n is small** (34 and 24 subjects). "Chance-level" is also
  consistent with *"a weak signal that is undetectable at this n."* Do not
  over-claim a negative.
- **Batch may not be purely technical.** With no demographics table, the two
  batches could be two recruitment campaigns differing in **age** — which the
  literature flags as the dominant confound (radar separates young vs elderly at
  94.9 %). Then this is a *clinical* confound, not a hardware artifact. Either
  way it is uncontrolled.

### Required actions
1. **Report `batch_only` (AUC 0.664) as a second null floor**, alongside
   `duration_only` (0.610), in every results table.
2. **Report within-batch AUC as the headline number**, not the pooled one.
3. **Ask the data provider** (Ignacio López-Delgado, `ie.lopez@upm.es`) what the
   two batches are: capture dates, sessions, hardware/firmware or processing
   changes, and whether recruitment differed. `param.DestinationPath` in the
   `.mat` files may carry capture dates — extract them.
4. Add **batch** to the confound battery for **every** model, including the CNN.

---

## 3c. Diagnostics — robustness & error analysis

`src/diagnostics.py` (run: `.venv/bin/python -m src.diagnostics`) stress-tests
the result and characterises the errors. Three committee-facing answers:

**(a) Errors are not group-biased.** At threshold 0.5 the `shape_clean` model is
correct on 38/58 subjects (66%); the 20 errors split **11 control / 9 PD** — no
systematic bias toward either class. Per-subject scores in `diag_per_subject.csv`
/ `.png`.

**(b) The duration-matched result is robust, not a caliper artifact.** Sweeping
the matching caliper from 0.25 s to 2.0 s, matched-subset AUC stays in a tight
**0.613–0.624** band (254–278 trials retained). The "signal survives duration
matching" claim does not depend on the caliper choice.

| caliper (s) | 0.25 | 0.50 | 0.75 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|
| AUC | 0.613 | 0.617 | 0.624 | 0.624 | 0.622 | 0.619 |

**(c) The model is data-limited, not representation-saturated.** A subject-level
learning curve (stratified subsampling, 30 repeats/point) rises monotonically
and is **still climbing at the right edge**: AUC 0.58 (10 subjects) → ~0.72
(50 subjects), a **+0.13** gain. The endpoint is high-variance (few test
subjects remain), so don't over-read 0.72 — but the upward trend is the robust
signal: **more subjects would materially help.** This is a concrete,
data-backed argument for the Phase-2 / prodromal cohort and for not over-fitting
conclusions to the current n=58.

## 4. Why the deep-learning route is expected to help (and is the thesis core)

Two independent reasons, both now evidence-backed:

- **Fixed-length windows weaken (but do NOT remove) the duration confound.** The
  preprocessing pipeline (`src/preprocessing.py`) cuts every trial into 3 s
  windows. A window has the same length whether it came from a 9 s or a 19 s
  trial, so duration cannot leak *within* a window.
  > 🔴 **Correction (2026-07-29).** An earlier version of this section claimed
  > windowing removes the confound "by construction." **That is false, and it was
  > verified empirically:** longer trials produce *more* windows — control **27.3**
  > vs PD **30.9** windows per subject (ratio 1.13) — and **window count alone
  > yields subject-level AUC 0.621**, i.e. the same level as the classical
  > baseline. Windowing decorrelates sample *length* from label, **not the number
  > of samples per subject**.
  > **Required control:** cap windows per subject at K (or weight the loss by
  > `1/n_windows`), and report the **0.621 window-count floor** as the deep
  > track's null floor, exactly as `duration_only`=0.610 is the classical floor.
- **Spatial micro-Doppler structure is discarded by the summaries.** The
  reference paper (López-Delgado et al.) shows the PD signal lives in the
  *shape* of the foot/torso Doppler signatures over the gait cycle — heel-strike
  minima, foot maximum velocity, foot/torso asymmetry — and explicitly reports
  that the torso-velocity↔heel-strike coupling **breaks down in PD subjects**.
  A 2-D CNN over the (Doppler × time) window can see that structure; a handful
  of scalar summaries cannot.

---

## 5. Recommended thesis plan (revised)

**Framing.** The reference paper *measures gait parameters*; this thesis
*classifies PD vs control directly from micro-Doppler spectrograms* — a
genuinely new contribution (confirmed: the paper contains no classifier).

**Methodology spine (in priority order):**

1. **Confound-controlled evaluation throughout.** Always report `duration_only`
   and `shape_clean` baselines next to any model. Prefer test2 (no-chair) for
   the cleanest gait comparison, and/or a duration-matched subset. *(This is now
   the single most important methodological point and should be stated up front
   in the thesis.)*
2. **Window-level CNN** on `[ce_foot, ce_torso]` 2-channel 224×224 windows,
   subject-level LOSO, subject score = mean window probability. This is the core
   model.
3. **ResNet-18 transfer learning** as the stronger comparator.
4. **Grad-CAM** to check the CNN attends to gait-relevant Doppler bands
   (foot 200–500 Hz, torso ≤200 Hz) rather than artifacts.
5. **Ablations that matter:** test1-vs-test2 (does removing the chair-stand
   confound change AUC?), foot-only vs torso-only vs both, `ce_*` vs raw
   `|stft_*|`, window length.

**Open items that gate stronger results** (ask the team — see `FINDINGS.md`
draft email): age/sex/UPDRS/medication table (for age-matched analysis and to
separate PD signal from ageing), and the `ce_*` formula.

**Honest expectation.** The confound-controlled classical floor is ~0.62–0.68
(it survives duration matching and residualization — so it is real, just weak).
With 58 subjects, target a *credible* deep-model subject-level AUC with a
reported CI and permutation test — a well-validated 0.7–0.8 that survives the
same confound checks (duration-matched + test2-only) is a far stronger thesis
result than an inflated number that doesn't. Run the matched/residualized checks
on the CNN too, not just the baseline.

---

## 6. Reproduce

```bash
cd thesis-parkinson-radar
python3 -m venv .venv && .venv/bin/pip install -r requirements-analysis.txt
.venv/bin/python -m src.eda          # EDA + statistics → outputs/figures/eda_*.png
.venv/bin/python -m src.baseline     # LOSO baselines  → outputs/figures/baseline_*.png
# notebooks 12 & 13 wrap the same functions for interactive use
```

# Project handbook — understand everything from scratch

*The single document that explains the whole project: the data, the EDA, the
pipeline, why AUC, how every number was produced, what's broken, and what "done"
means. Written 2026-07-29, grounded in the actual code and the actual outputs.*

**Companion documents:** `WORK_PLAN.md` (the 4-week schedule) ·
`RESULTS.md` (raw numbers) · `METHODOLOGY_EXPLAINED.md` (engineering deep-dive) ·
`LITERATURE_REVIEW.md` + `RELATED_WORK.tex` (the field).

---

# Part 1 · The problem and the data

## 1.1 What we are building

**One sentence:** given a radar recording of a person walking, decide whether
that person has Parkinson's disease or is a healthy control.

This is a **binary classifier**. Input = a radar "movie of movement". Output = a
probability between 0 and 1.

**Why it's interesting:** current gait assessment needs either wearable sensors
(must be worn, alters natural gait) or cameras (privacy-invasive, lighting
dependent). Radar is contactless, works in the dark, through clothing, and sees
no faces. If it works, it could passively monitor people at home.

**Why it's novel:** the literature review (15 verified papers) found **no
published work that classifies Parkinson's vs control from radar micro-Doppler
with subject-independent validation.** The paper that produced our dataset
(López-Delgado et al. 2026) *measures gait parameters* — it contains **no
classifier**. That gap is the thesis.

## 1.2 The physics — how a walking person becomes a picture

```
1. Radar transmits a 23 GHz FMCW chirp        (radar_params.F_CARRIER_HZ)
2. Wave bounces off a moving body
3. Motion shifts the returned frequency        ← the Doppler effect
4. Faster motion  →  bigger frequency shift
5. STFT over time  →  SPECTROGRAM
```

A **spectrogram** has:
- **horizontal axis = time**
- **vertical axis = Doppler frequency** = *speed of whatever is moving*
- **brightness = how much of the body was moving at that speed, at that instant*

So it is **not a photograph**. It is a chart of *how fast each part of the body
was moving, moment by moment*.

The conversion between the two is exact
([radar_params.py:84](src/radar_params.py)):
```
velocity (m/s) = Doppler (Hz) × wavelength / 2      wavelength ≈ 1.3 cm
```

Which gives the two bands that matter:

| Body part | Typical speed | Doppler | Constant in code |
|---|---|---|---|
| **Torso** | ~1 m/s (steady) | ~150 Hz | `TORSO_BAND_HZ = (0, 200)` |
| **Feet** | 3–4 m/s (swinging) | up to ±500 Hz | `FOOT_BAND_HZ = (200, 500)` |

Your torso glides at a near-constant speed; your feet stop dead on the ground and
then swing fast. That is why feet produce the big bright arcs far from the centre
and the torso produces the steady bright band near the centre.

## 1.3 What is actually inside a `.mat` file

348 files, ~305 MB each, 106 GB total, MATLAB **v5** format (so `scipy.io.loadmat`
works — *not* `h5py`).

| Variable | What it is |
|---|---|
| `ce_foot` | contrast-enhanced magnitude spectrogram, **foot** radars |
| `ce_torso` | contrast-enhanced magnitude spectrogram, **torso** radar |
| `stft_foot` / `stft_torso` | the raw **complex** STFT (before enhancement) |
| `doppler_axis` | 320 bins spanning ±800 Hz |
| `t_axis` | time axis |

**"ce" = contrast-enhanced** — the brightness was rescaled per recording to make
patterns visible. ⚠️ **The formula is not in the paper and we don't have the
script.** This is an open provenance gap (ask the data owner).

Two important measured facts:
- **Doppler bins are spaced 5 Hz**, but the true resolution from the 50 ms STFT
  window is **20 Hz** — so the data is **4× zero-padded**. Bins are interpolated,
  not independent.
- Arrays come out as **(doppler, time)** — no transpose needed.

## 1.4 The cohort

| | |
|---|---|
| Subjects | **58** — 33 control (`fisc_*`), 25 Parkinson's (`fisp_*`) |
| Protocol | **Timed-Up-and-Go (TUG)**: stand, walk 3 m, turn, walk back, sit |
| Conditions | `test1` = **with chair** · `test2` = **without chair** |
| Repeats | 3 trials each → ~348 recordings |
| Trial length | **median 8.7 s** (min 7.0, max 18.5) |
| Anomalies | `fisp_022` has an extra trial; `fisp_048` is missing one |

**58 subjects is small.** That single fact drives almost every methodological
decision below.

---

# Part 2 · The EDA — what we did and what we found

## 2.0 Why EDA first, and why it mattered here

Exploratory Data Analysis = look hard at the data *before* modelling, so you learn
what it actually contains and — crucially — **what could fool a model**.

In this project EDA wasn't a formality. **It changed the entire thesis.** It found
a confound that made the obvious approach invalid, and it is now the strongest
part of the contribution.

## 2.1 Step 1 — Inventory (`outputs/metrics/inventory.csv`)

Walk every `.mat` file and record: subject, group, test, trial, file size, path,
**duration**, number of Doppler bins.

**Found:** 348 files, consistent 320 Doppler bins, and the two anomalies above.

## 2.2 Step 2 — Quality control (`src/qc.py` → `qc_report.csv`)

Per-trial sanity checks: is the array finite? is it non-empty? is the duration
plausible (`TRIAL_DURATION_S_MIN/MAX`)? is the Doppler axis the expected size and
range? are there dead/saturated frames?

**Purpose:** never let a corrupt file silently become a "finding". Excluded files
are recorded in `excluded_files.csv` so exclusions are auditable.

## 2.3 Step 3 — Feature extraction (`src/features.py`)

For the classical track we compress each spectrogram into ~25 numbers. Every
feature is computed **twice** — once for `foot`, once for `torso`.

### Every feature, in plain language

| Feature | What it measures physically | How it's computed |
|---|---|---|
| `mean` | overall reflected energy | `spec.mean()` |
| `max` | the single fastest/strongest moment | `spec.max()` |
| `std` | how much the signal varies | `spec.std()` |
| **`centroid_hz`** | **the "average speed" of the body** — centre of mass of the Doppler distribution | `Σ(|f| × energy) / Σ(energy)` |
| **`bandwidth_hz`** | **spread of speeds** — is motion uniform or varied? | std-dev around the centroid |
| **`entropy`** | how *spread out / disordered* the speed distribution is | `−Σ p log p` over the normalised Doppler marginal |
| `torso_band_mean` | mean energy in **0–200 Hz** (torso speeds) | mean over the band mask |
| `foot_band_mean` | mean energy in **200–500 Hz** (foot speeds) | mean over the band mask |
| **`foot_to_torso_band_ratio`** | **how much fast (foot) motion vs slow (torso) motion** | `foot / (foot + torso)` |
| **`env_cv`** | **rhythmicity** — coefficient of variation of the energy-over-time envelope | `env.std() / env.mean()` where `env = spec.sum(axis=0)` |
| `foot_torso_mean_ratio` | cross-channel balance | `foot_mean / torso_mean` |

**The clinically motivated ones** are `centroid_hz` (bradykinesia → slower →
lower centroid), `foot_to_torso_band_ratio` (reduced foot swing → smaller ratio),
and `env_cv` (gait variability → a hallmark of parkinsonian gait).

### The duration-invariance design rule

`features.py` deliberately uses **means, ratios and normalised shapes** — never
raw sums. Its docstring states the rule:

> *any energy-summed feature (`spec.sum()`) would leak duration*

Because a **sum** grows with recording length, but a **mean** does not. This is
the direct consequence of the discovery in §2.5.

## 2.4 Step 4 — The statistics, and *why each test was chosen*

All in [src/stats.py](src/stats.py). Every choice has a reason:

| Test | Why this one and not the obvious alternative |
|---|---|
| **Mann–Whitney U** | Non-parametric. The usual t-test assumes normally distributed data; gait features are **skewed** and n is small. Mann–Whitney only assumes you can rank the values. |
| **Cliff's delta** | An effect size, i.e. *how big* is the difference. A p-value alone is nearly meaningless at n=58 — with enough data everything is "significant". Cliff's δ = `P(a>b) − P(a<b)`, thresholds: <0.147 negligible, <0.33 small, <0.474 medium, else large. |
| **Benjamini–Hochberg FDR** | We test ~25 features at once. Testing many things guarantees some false positives by luck. BH controls the *false discovery rate*. Chosen over **Bonferroni**, which is over-conservative and would hide real effects. |
| **Spearman correlation** | Rank-based, so it catches *monotonic* (not just linear) relationships — used for the duration-leakage audit. |
| **Subject-level bootstrap** | Confidence intervals must resample **subjects, not trials** — because trials from one person are correlated. Resampling trials would give falsely narrow CIs. |
| **Label-permutation test** | The honest null: shuffle the labels, re-run *everything*, see how often chance beats you. |

## 2.5 🔴 The discovery that changed the thesis — the duration confound

**Parkinson's subjects take longer to do the walk.**

| Test | PD − control median duration | Mann–Whitney p | Cliff's δ |
|---|---|---|---|
| `test1` (with chair) | **+10.1 %** | 0.0016 | 0.28 (small–medium) |
| `test2` (no chair) | +3.4 % | 0.14 (n.s.) | 0.13 (negligible) |

**Why this is dangerous:** a model can "detect Parkinson's" by secretly measuring
*how long the recording is* — learning nothing about gait.

And that is exactly what was happening. The features that looked *most*
discriminative were all **energy sums**, which grow with length:

| Feature | Cliff's δ | FDR p | Confounded? |
|---|---|---|---|
| `torso_torso_band_energy` | +0.46 (medium) | 4.5e-12 | **yes** |
| `foot_total_energy` | +0.45 (medium) | 6.2e-12 | **yes** |
| `torso_total_energy` | +0.44 (medium) | 1.6e-11 | **yes** |
| `foot_foot_band_energy` | +0.43 (medium) | 3.7e-11 | **yes** |

**24 of 25 features were "significant" after FDR correction — and it was mostly
trial length.**

Worse, the leakage-audit found that even *nominally* duration-invariant features
correlate with length (`torso_std` r = −0.65, `foot_mean` r = −0.42), because
longer PD trials contain more slow standing/turning frames that shift per-pixel
statistics.

**This is the single most important finding in the project so far**, and it is
why the methodology (not the accuracy) is the contribution.

## 2.6 The subject fingerprint

Notebook 11 compared **within-subject** vs **between-subject** distance. The
distance is Euclidean in the **standardised feature space**, not on the raw
spectrogram pixels.

Recomputed from the current `trial_features.csv` over the **17
duration-invariant** features only (`tools/make_fingerprint_figure.py`, seed 0,
10 comparison subjects per person; numbers cached in
`outputs/metrics/subject_fingerprint.json`):

| Quantity | Median |
|---|---|
| within-subject distance | **1.811** |
| between-subject distance | **4.593** |
| ratio within/between | **0.378** |
| subjects with within < between | **58 / 58** |

A person's own trials are **~2.5× more alike** than anyone else's, i.e. **your
walk is a fingerprint** — and because only duration-invariant features are used,
that fingerprint is in *gait shape*, not in *elapsed time*.

> Earlier figures reported 2.04 / 5.23 / 0.374. Those used all 21 feature columns
> the extractor returned at the time, including the duration-growing ones. The
> conclusion is unchanged (still 58/58) but the narrower feature set is the one
> to quote.

**Consequence:** random train/test splitting is invalid. It would let the model
recognise *people*. This is why **LOSO is mandatory** (§4.5).

## 2.7 The figures produced, and what each showed

| Figure | Shows | Finding |
|---|---|---|
| `eda_class_balance.png` | subjects/trials per group | 33 vs 25 — mild imbalance → use `class_weight="balanced"` |
| `eda_duration_confound.png` | duration distributions by group | **the confound, visually** |
| `eda_duration_audit.png` | feature ↔ duration correlations | which features leak length |
| `eda_effect_sizes.png` | Cliff's δ per feature | energy features dominate — for the wrong reason |
| `eda_top_feature_distributions.png` | distributions of top features | heavy overlap → weak separability |
| `eda_correlation.png` | feature correlation matrix | features are highly redundant |
| `eda_pca.png` | 2-D projection | **groups do not visibly separate** — an early warning the signal is weak |
| `11_within_vs_between.png` | subject similarity | the fingerprint → LOSO required |
| `11_duration_by_test.png` | duration by test | the confound is concentrated in test1 |
| `11_ce_mapping_fit.png` | `ce` transform probing | enhancement is **adaptive per trial** — provenance unknown |
| `11_fisp022_trial4.png` | the extra trial | a clean retry, not a duplicate → keep it |
| `11_fisp048_test2.png` | the missing trial | a clean deletion → proceed with 5 files |
| `baseline_auc_grid.png` | AUC for every model × feature-set | the headline comparison (§5.1) |
| `baseline_roc.png` | ROC curves by feature set | exposes the confound visually |
| `confound_per_test.png` | AUC split by test | test2 is the clean test |
| `confound_residualized.png` | raw vs duration-removed AUC | removing duration **raises** AUC |
| `confound_duration_matched.png` | before/after matching | groups made duration-identical |
| `confound_feature_importance.png` | RF permutation importance | ≈0 everywhere → signal is **weak and distributed** |
| `diag_per_subject.png` | per-subject scores | errors are **not** group-biased |
| `diag_caliper_sweep.png` | AUC vs matching caliper | result is not a caliper artifact |
| `diag_learning_curve.png` | AUC vs #subjects | **still climbing** → data-limited |

---

# Part 3 · The pipelines, end to end

## 3.1 The classical pipeline (done, and it runs on your laptop)

```
.mat file
   │ scipy.io.loadmat                      data_loader.py
   ▼
ce_foot, ce_torso  (320 × time)
   │ compute ~25 summary numbers           features.py
   ▼
one row per trial  →  trial_features.csv   (348 rows)
   │ merge trial durations                 config.load_trial_table()
   ▼
LOSO: train on 57 subjects, test on 1      baseline.loso_subject_scores()
   │ average trial probs → 1 score/subject
   ▼
58 subject scores  →  AUC, sens, spec, CI  baseline.subject_metrics()
```

Three classifiers are compared ([baseline.py:50](src/baseline.py)):
**Logistic Regression** (the headline — simple, linear, interpretable),
**SVM-RBF**, and **Random Forest**. All with `class_weight="balanced"`, all
preceded by median imputation and standardisation in an sklearn `Pipeline` so
scaling is fit on training data only (no leakage).

Five **feature sets** are compared, and this is the clever part
([baseline.py:74](src/baseline.py)):

| Set | Contents | Purpose |
|---|---|---|
| `all_features` | everything | the naive approach |
| `duration_invariant` | means/ratios/shapes | the intended clean set |
| `shape_clean` | invariant **AND** \|Spearman r with duration\| < 0.3 | **the honest set** — data-driven |
| `confounded_only` | just the energy sums | shows how much duration alone buys |
| **`duration_only`** | **just `duration_s`** | 🔑 **the control** — "can you beat a stopwatch?" |

That last row is the key scientific control. Without it you cannot tell whether a
model learned gait or learned length.

## 3.2 The deep pipeline (built, **not yet run**)

```
.mat  →  ce_foot, ce_torso
   │ cut into fixed-length windows          preprocessing.window_indices()
   ▼  (3 s window, 1.5 s stride — 50% overlap)
window (320 × ~time-bins)
   │ log1p compression + z-score            preprocessing.normalise_window()
   │ bilinear resize → 224×224              preprocessing.resize_2d()
   ▼
stack [foot, torso] → (2, 224, 224)         ← like RGB, but foot/torso
   │ save .npy + a row in manifest.csv
   ▼
WindowDataset reads the manifest            dataset.py
   ▼
SmallCNN (~50k params) or ResNet-18         models.py
   ▼
LOSO training                               train.py
   ▼
average window probs → 1 score/subject → AUC
```

**Why cache to `.npy`:** the 106 GB of `.mat` files is opened **exactly once**
(measured: 1.4 min for all 348). Everything afterwards reads a **0.68 GB** cache.
This is what makes the whole project feasible on a laptop.

**The manifest CSV is the single source of truth** — subject, group, label, test,
trial, window index, time span, file path. LOSO becomes a one-line filter, and
every experiment is auditable.

---

# Part 4 · Evaluation — why AUC, and how we got it

This is the part to understand deeply, because every number in the thesis is one
of these.

## 4.1 Start with the confusion matrix

The model outputs a **probability**. Pick a threshold (say 0.5) and you get four
outcomes:

|  | predicted control | predicted PD |
|---|---|---|
| **actually control** | TN (true negative) | FP (false positive) |
| **actually PD** | FN (false negative) | TP (true positive) |

From which:
- **Sensitivity** (recall) = `TP / (TP + FN)` — *of all real patients, how many did we catch?*
- **Specificity** = `TN / (TN + FP)` — *of all healthy people, how many did we correctly clear?*

These trade off. Lower the threshold → catch more patients (↑sensitivity) but
also raise more false alarms (↓specificity).

## 4.2 The ROC curve

**Every possible threshold** gives one (sensitivity, specificity) pair. Plot them
all — sensitivity on the y-axis, `1 − specificity` on the x-axis — and you get the
**ROC curve** (`baseline_roc.png`).

- A **diagonal line** = random guessing.
- A curve **bowing toward the top-left** = a good model.

## 4.3 What AUC actually is

**AUC = the Area Under that ROC Curve.**

| AUC | Meaning |
|---|---|
| 0.5 | coin flip — useless |
| 0.62 | slightly better than chance |
| 0.70–0.80 | respectable for a hard medical problem |
| 0.90+ | excellent (or you have a confound…) |

**The most intuitive definition:** *pick one random patient and one random healthy
person. AUC is the probability the model gives the patient the higher score.*

So AUC 0.62 means: **62% of the time, the model correctly ranks the patient above
the healthy person.** Chance is 50%.

Computed by `sklearn.metrics.roc_auc_score` over the 58 subject scores
([baseline.py:166](src/baseline.py)).

## 4.4 Why AUC and not accuracy

1. **Threshold-free.** Accuracy depends entirely on where you draw the cutoff.
   AUC evaluates *all* cutoffs, so it measures the model, not the cutoff.
2. **Imbalance-robust.** With 33 control vs 25 PD, always predicting "control"
   gives **57% accuracy** while being completely useless. AUC would correctly
   read 0.5.
3. **Field standard** — makes the thesis comparable.

> ⚠️ **Comparison trap:** most papers report **accuracy (%)**; we report **AUC**.
> They are *different scales*. "0.80 AUC" is **not** worse than "95% accuracy" —
> they cannot be directly compared. Say this explicitly in the thesis.

**But AUC is never reported alone.** `subject_metrics`
([baseline.py:117](src/baseline.py)) always returns AUC + bootstrap CI,
balanced accuracy, F1, sensitivity, specificity, and the confusion matrix —
because for a *screening* tool, sensitivity matters enormously.

## 4.5 How the AUC is actually produced — LOSO, step by step

```
for each of the 58 subjects S:
        train the model on the other 57 subjects
        predict a probability for each of S's trials/windows
        subject_score[S] = mean of those probabilities
AUC = roc_auc_score(true_labels[58], subject_score[58])
```

Three deliberate decisions inside that loop:

1. **Leave-one-SUBJECT-out, not leave-one-trial-out.** Because gait is a
   fingerprint (§2.6). If a subject's trials straddle the split, the model
   recognises the person. Implemented as sklearn's `LeaveOneGroupOut` with
   `groups = subject_id` ([baseline.py:102](src/baseline.py)).
2. **Subject-level aggregation.** The clinical question is "is *this person* ill?"
   — not "is this 3-second window ill?". Averaging also prevents subjects with
   more trials from dominating.
3. **Metrics computed over 58 numbers**, not thousands of trials — so the sample
   size is honest and the CIs aren't artificially tight.

**Who invented LOSO?** No single person — it's standard leave-one-out
cross-validation (Stone 1974, Geisser 1975) with the *subject* as the group. It's
the norm in biomedical ML. What is *ours* is enforcing it here; much of this
literature does not, which is why their numbers are higher and ours are honest.

## 4.6 Uncertainty — the bootstrap CI

A single AUC from 58 people could be luck. So:

> resample the 58 **subjects** with replacement, recompute AUC, repeat 2000×,
> take the 2.5th and 97.5th percentiles.

That's the 95% CI ([stats.py:107](src/stats.py)). Note it resamples **subjects,
not trials** — trials within a person are correlated, and resampling trials would
give falsely narrow intervals.

## 4.7 Significance — the permutation test

The most honest possible null:

```
repeat 1000 times:
      randomly shuffle the PD/control labels among subjects
      re-run the ENTIRE LOSO pipeline
      record the AUC achieved on nonsense labels
p = fraction of shuffles that scored ≥ our real AUC
```

([baseline.py:135](src/baseline.py))

**Correct interpretation of p = 0.022:** *if the labels were truly random, only
~2.2% of shuffles would reach an AUC this high.*
**Wrong (but common):** "there's a 2.2% chance the result is due to luck." Don't
write that in the thesis.

> ⚠️ **This is affordable for the classical model and NOT for a deep one.** At
> ~1.7 h per deep LOSO, 1000 permutations ≈ **69 days**. See `WORK_PLAN.md` Day 11
> for the replacement (bootstrap CI + a reduced permutation on SmallCNN only).

---

# Part 5 · The results so far, read honestly

## 5.1 The classical baseline — the AUC grid

Subject-level LOSO, logistic regression:

| Feature set | # feats | **AUC** | 95% CI | Sens | Spec |
|---|---|---|---|---|---|
| `confounded_only` | 8 | **0.675** | [0.53, 0.81] | 0.56 | 0.64 |
| `all_features` | 25 | 0.633 | [0.49, 0.78] | 0.60 | 0.64 |
| `duration_invariant` | 17 | 0.627 | [0.48, 0.78] | 0.64 | 0.61 |
| **`shape_clean`** | 10 | **0.621** | [0.47, 0.77] | 0.64 | 0.67 |
| **`duration_only`** 🔑 | 1 | **0.610** | [0.45, 0.75] | 0.40 | 0.70 |

**How to read this — it's the whole story in five rows:**
- **A stopwatch alone scores 0.610.**
- The "clean" gait features reach only **0.621** — barely above the stopwatch.
- The *best* naive score (0.675) comes from the **most confounded** features.
- **Every CI includes or nearly includes 0.5.**

Naive conclusion: *the signal is mostly trial length.* And a permutation test on
the raw features agreed — **p = 0.092, not significant.**

## 5.2 The confound battery — and the twist

Three independent controls, all in `confound_analysis.py`:

**(a) Split by test.** The confound lives in test1:

| feature set | both | test1 | test2 |
|---|---|---|---|
| `duration_only` | 0.610 | 0.646 | **0.536** |
| `shape_clean` | 0.621 | 0.628 | **0.589** |

In test2, duration alone is essentially chance (0.536) — yet clean gait features
still reach **0.589**. That gap is *real, non-duration signal*.

**(b) Residualization** — regress each feature on duration, keep the residual:

| feature set | raw | **residualized** |
|---|---|---|
| `shape_clean` | 0.621 | **0.676** |
| `duration_invariant` | 0.627 | 0.663 |

**Removing duration *raises* AUC.** Duration was acting as a **suppressor** —
partially masking the gait signal, not creating it. This is the twist.

**(c) Duration matching** — pair each PD trial with a control trial of near-equal
length (270 trials; post-match Mann–Whitney **p = 0.95**, i.e. groups are now
duration-identical):

| feature set | matched AUC |
|---|---|
| `duration_invariant` | 0.638 |
| `shape_clean` | 0.624 |

If the signal were pure length leakage, this would collapse to ~0.5. **It holds at
0.62–0.64.**

**(d) The capstone.** Permutation test on the *residualized* model:
**AUC 0.676, p = 0.022 — significant.**

> **That was the conclusion — until the batch confound was found. Read §5.2b.**

## 5.2b 🔴 The acquisition-batch confound — this changes the conclusion

*Found and independently verified 2026-07-29. Batch labels cached in
`outputs/metrics/acquisition_batch.csv`.*

The `.mat` files contain **exactly two distinct Doppler axes**, differing in the
4th decimal place — an acquisition/session signature:

| Batch | `doppler_min` | Trials | Subjects | control | PD | % PD |
|---|---|---|---|---|---|---|
| **A** | −794.8713989257812 | 203 | 34 | 24 | 10 | **29.4 %** |
| **B** | −794.872802734375 | 145 | 24 | 9 | 15 | **62.5 %** |

**All 58 subjects are pure w.r.t. batch** — so batch is a *subject-level*
property, and it is **confounded with diagnosis**.

> ### 🔑 Batch membership ALONE gives subject-level **AUC = 0.664**
> Higher than **every** feature model — higher than the 0.621 headline and the
> 0.610 duration floor.

**The decisive test — re-run LOSO *inside* each batch:**

| Feature set | pooled | batch A | batch B |
|---|---|---|---|
| `all_features` | 0.633 | 0.604 | **0.467** |
| `duration_invariant` | 0.627 | 0.567 | **0.496** |
| **`shape_clean`** | **0.621** | **0.537** | **0.541** |
| `duration_only` | 0.610 | **0.642** | **0.659** |

**The spectral-shape gait features collapse to chance within a batch.** Much of
the "weak but real signal" was the model detecting *which acquisition batch a
subject came from*.

**What survives — and it's the interesting part.** `duration_only` does **not**
collapse; it **strengthens** (0.642 / 0.659), and PD trials are longer inside
*both* batches (A: 8.93→9.74 s; B: 7.81→8.91 s). So **walking slowly is a
genuine, batch-robust PD marker** — plausibly bradykinesia — while the
handcrafted spectral-shape descriptors are not.

**Why nothing caught it:** duration matching, residualization and test2-only
**all leave batch fully intact**. It is orthogonal to every control run so far.

**It explains two earlier puzzles:**
- The PC1 bimodality in `eda_pca.png` (62 % of variance) **is the batch split**,
  not the disease.
- The errors are "confident" rather than borderline because the classifier's
  prediction tracks batch for ~55/58 subjects.

**Caveats — state them honestly:**
- Within-batch n is small (34 / 24). "Chance" is also consistent with *a weak
  signal undetectable at this n.* Don't over-claim a negative either.
- Batch may not be purely technical — the two batches could be two recruitment
  campaigns differing in **age**, the confound the literature flags hardest.

**Required actions:** report `batch_only` (0.664) as a **second null floor**
beside `duration_only`; make **within-batch AUC the headline**; add batch to the
confound battery for every model including the CNN; and **ask the data owner**
what the two batches are (capture dates may be in `param.DestinationPath`).

## 5.3 Diagnostics — is the result robust?

**(a) Errors aren't biased.** 38/58 correct (66%); the 20 errors split
**11 control / 9 PD** — no systematic bias.

**(b) Not a caliper artifact.** Sweeping the matching caliper 0.25 s → 2.0 s keeps
AUC in a tight **0.613–0.624** band.

**(c) 🔑 Data-limited, not method-limited.** The learning curve rises
monotonically — **AUC 0.58 at 10 subjects → ~0.72 at 50** — and is **still
climbing at the right edge**. More subjects would materially help. *(Don't
over-read the 0.72 endpoint; it's high-variance. The trend is the finding.)*

**(d) Feature importance ≈ 0 everywhere** → the signal is **weak and distributed**
across features, not carried by one. Lean on the matched/residualized evidence,
not on this plot.

---

# Part 6 · Known problems — the honest list

## 6.1 🔴 Code bugs (must fix before any training)

1. **Early stopping is degenerate.** In LOSO the val set is one subject → all
   labels identical → AUC hard-coded to `0.5` every epoch → **the model trains for
   1 epoch** and restores epoch-0 weights. Reproduced empirically. The run still
   finishes in 4.5 min *looking* successful.
2. **Model selection on the test subject** — textbook leakage; exactly what the
   Related Work chapter criticises others for.
3. **No `torch.save` anywhere** → Grad-CAM can never run.
4. **No grouped-CV splitter** exists.
5. `num_workers=0` — the dataloader costs *more* than the GPU compute.

## 6.2 🔴 Reproducibility break

`trial_features.csv` **cannot be regenerated** by today's `features.py`: 9 CSV
columns don't exist in the code (`foot_total_energy`, `torso_total_energy`,
`*_band_energy`, `*_env_std`, `foot_torso_total_ratio`) and the code emits 5
columns not in the CSV (`*_band_mean`, `foot_torso_mean_ratio`). *Verified.*
The cached table came from an older version. **Must be repaired before any number
is defended.**

## 6.3 🔴 The window-count confound (a correction)

Earlier documents claimed fixed-length windowing removes the duration confound
"by construction." **That is false — verified:**

```
windows/subject:  control 27.3   PD 30.9   (ratio 1.13)
AUC from WINDOW COUNT ALONE = 0.621
```

Longer trials produce **more windows**. Windowing decorrelates sample *length*,
not sample *count*. **Fix:** cap windows per subject, or weight the loss by
`1/n_windows`; report **0.621 as the deep track's null floor**, just as
`duration_only` = 0.610 is the classical floor.

## 6.4 Open scientific risks

| Risk | Status |
|---|---|
| **🔴 Acquisition-batch confound** | ⛔ **the biggest one** — batch alone scores **0.664**, beating every model; gait features collapse to chance within batch (§5.2b). Must become a standard control. |
| **Age confound** | ⛔ unresolved — same radar separates young/elderly at **94.9%**; we have **no age data**. Top priority ask. And the two batches may *be* two age-different cohorts. |
| **`ce` provenance** | ⛔ formula unknown, adaptive per trial |
| **Normalisation vs bradykinesia** | per-window z-scoring **deletes absolute amplitude** — but reduced amplitude *is* a PD symptom. Run the ablation. |
| **Time-flip augmentation** | plays the walk backwards; gait isn't time-symmetric. Justify or drop. |
| **Selection optimism** | all choices tuned on the same 58 subjects that are then reported. Mitigate by **pre-registering** the final config. |

---

# Part 7 · Map of the repository

## Modules (`src/`)

| Module | Role | Run? |
|---|---|---|
| `config.py` | paths, feature groups, seeds, `load_trial_table()` | — |
| `radar_params.py` | radar/STFT constants from the paper; Doppler↔velocity | — |
| `data_loader.py` | `.mat` reading, `iter_trials`, `load_trial` | ✅ |
| `qc.py` | quality control | ✅ |
| `features.py` | the ~25 handcrafted features | ✅ |
| `stats.py` | Mann–Whitney, Cliff's δ, BH-FDR, bootstrap | ✅ |
| `eda.py` | EDA figures + tables | ✅ |
| `baseline.py` | LOSO classical baselines, permutation, AUC grid | ✅ |
| `confound_analysis.py` | per-test, residualization, matching, importance | ✅ |
| `diagnostics.py` | per-subject errors, caliper sweep, learning curve | ✅ |
| `preprocessing.py` | windowing → `.npy` cache + manifest | ⏳ |
| `dataset.py` | `WindowDataset` (+ channel modes, augmentation) | ⏳ |
| `models.py` | `SmallCNN`, `resnet18_finetune` | ⏳ |
| `train.py` | `train_one_fold`, `loso_cv` 🔴 **buggy** | ⏳ |
| `interpret.py` | Grad-CAM | ⏳ |
| `channels.py` | channel utilities | ⏳ |

## Notebooks — canonical order

| # | Notebook | Purpose | Executed? |
|---|---|---|---|
| 01 | data_inspection | first look at a `.mat` | no outputs |
| 02 | dataset_overview | inventory | no outputs |
| 03 | eda_spectrograms | visual EDA | ✅ |
| 04 | quality_control | QC report | ✅ |
| 05 | feature_eda | feature distributions | no outputs |
| **11** | **data_forensics** | **the confound + fingerprint + anomalies** | ✅ **richest** |
| 12 | eda_statistics | statistical tables | no outputs |
| 13 | baseline_models | LOSO baselines | no outputs |
| 06 | preprocessing | build the window cache | ⏳ never run |
| 07 | baselines | (overlaps 13) | ⏳ |
| 08 | cnn | SmallCNN | ⏳ never run |
| 09 | resnet | ResNet-18 | ⏳ never run |
| 10 | gradcam | interpretability | ⏳ never run |

⚠️ Several notebooks have **no saved outputs** even though the equivalent `src/`
module *was* run headless — the real results live in `outputs/` and `reports/`,
not in the notebooks. **07 overlaps 13, and 05 overlaps 12.** Consider retiring
the duplicates so there is one canonical path.

## Where results live
- `outputs/figures/` — 29 figures
- `outputs/metrics/` — `trial_features.csv`, `inventory.csv`, `qc_report.csv`, `baseline_auc_grid.csv`, `diag_*.csv`
- `reports/` — `baseline_summary.json`, `confound_summary.json`, `confound_permutation.json`, `diagnostics_summary.json`, `eda_summary.json`

---

# Part 8 · What "done" means

## 8.1 Done for the *implementation*
Everything in `WORK_PLAN.md` §6A — bugs fixed, cache built, both models
LOSO-evaluated on one **pre-registered** config each, all five ablations run,
Grad-CAM produced, confound battery + bootstrap CIs complete, and **a clean
checkout reproduces every headline number**.

## 8.2 Done for the *thesis*
Everything in `WORK_PLAN.md` §6B — six chapters, English abstract **and Spanish
resumen**, full front/back matter, ODS/SDG mapping, budget table, official
template, similarity check, **tutor's visto bueno**, deposited.

## 8.3 Done for the *science* — decide this now

**Success is not a number. Success is a number you can defend.**

| Outcome | Verdict |
|---|---|
| **> 0.70 AUC**, survives duration-matched + test2-only + window-count control | **Strong thesis** |
| **0.62–0.70**, survives the battery | **Solid thesis** — beats the floor honestly; n=58 explains the ceiling |
| **≈ 0.62** (the classical floor), battery honest, Grad-CAM sensible | **Acceptable thesis** — the contribution is the *first subject-independent radar PD classifier* + the confound-controlled methodology |
| Any high number that **collapses** under the battery | **Report the collapse.** That IS the finding — and it's the most valuable thing you could contribute to this literature |

The learning curve (0.58 → 0.72, still climbing) already tells you the ceiling
here is **data, not method**. That is a legitimate, defensible, publishable
conclusion — and it's an argument for the Phase-2 cohort.

**The one-line version to remember:**
> *We are not trying to get a big number. We are trying to find out how much real
> signal exists — and to be the first to measure it honestly.*

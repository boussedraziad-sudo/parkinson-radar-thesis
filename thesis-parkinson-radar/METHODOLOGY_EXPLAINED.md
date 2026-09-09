# Methodology & engineering, explained

*Companion to `WORK_PLAN_2WEEKS.md`. Written 2026-07-29 after reading every
module in `src/`. Covers: the science, the engineering choices, the problems
found in the code, and why the plan is ordered the way it is.*

---

# Part 0 — Two bugs to fix before you train anything

Found while reading `src/train.py`. **Fix these on Day 1 or Week 1 produces
meaningless numbers.**

## Bug 1 — early stopping is degenerate: the model trains for exactly 1 epoch

In `train_one_fold` ([train.py:80-93](src/train.py)):

```python
auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.5
```

In LOSO the validation set is **one held-out subject**. That subject is either
PD or control — so **every validation window has the same label**. Therefore
`len(set(labels)) == 1`, and `auc` is **always exactly 0.5**, every epoch.

Trace what that does:

| Epoch | `auc` | `auc > best_auc`? | Effect |
|---|---|---|---|
| 0 | 0.5 | 0.5 > 0.0 ✅ | save weights, `patience_left = 5` |
| 1 | 0.5 | 0.5 > 0.5 ❌ | `patience_left = 4` |
| 2–4 | 0.5 | ❌ | 3 → 2 → 1 |
| 5 | 0.5 | ❌ | 0 → **break** |

Then `model.load_state_dict(best_state)` restores the **epoch-0** weights.

**Result: you train 6 epochs, throw away 5 of them, and evaluate a model that
learned for one epoch.** Your CNN and ResNet would look barely better than
chance and you'd conclude "deep learning doesn't work here" — when in fact it
never trained.

## Bug 2 — early stopping peeks at the test subject

Even if Bug 1 were fixed by using a metric that works on one class (e.g. loss),
the deeper problem stands: `train_one_fold` selects the best epoch using the
**held-out subject** — the very subject whose score you then report. That is
**test-set leakage**: the stopping decision is informed by test labels.

## The fix (do this Day 1)

Three-way split inside each fold:

```
58 subjects
├── 1 subject   → TEST      (held out; touched only once, at the end)
├── ~6 subjects → INNER VAL (early stopping + epoch selection)
└── ~51 subjects→ TRAIN
```

The inner-val set contains **both classes**, so AUC is meaningful and early
stopping works. The test subject is never used for any decision. Concretely:

- change `loso_cv` to carve an inner-validation group out of `train_subj`
  (stratified by class, e.g. 3 control + 3 PD),
- pass that as `val_ds` to `train_one_fold`,
- after training, run the restored best model on the held-out subject and use
  **those** probabilities for the reported score.

## Also worth fixing while you're in there

| Issue | Where | Why it matters |
|---|---|---|
| `num_workers=0` | [train.py:60-61](src/train.py) | Data loads on the main thread. On a 10-core M1 Max use `num_workers=4-6` + `persistent_workers=True`. **This is a large speedup across 58 folds.** |
| Time-flip augmentation | [dataset.py:60-61](src/dataset.py) | `arr[..., ::-1]` reverses the **time** axis — i.e. walking backwards. Gait is not time-symmetric (heel-strike → toe-off). Justify it or drop it. |
| No fixed seed per fold | [train.py](src/train.py) | Results won't be exactly reproducible. Seed per fold. |

---

# Part 1 — The methodology (the science)

## 1.1 What the task actually is

Given a radar recording of someone walking, decide: **Parkinson's patient or
healthy control?** One binary decision per person.

The physical chain:

```
person walks → radar emits 23 GHz FMCW chirps → reflections return
             → each moving body part shifts the frequency (Doppler effect)
             → STFT over time  → SPECTROGRAM (time × Doppler-frequency)
             → contrast enhancement ("ce") → what we actually receive
```

Doppler frequency maps directly to **radial speed**
(`doppler_to_velocity_m_s` in [radar_params.py:84](src/radar_params.py)):

| Body part | Speed | Doppler | Band in our data |
|---|---|---|---|
| Torso | ~1 m/s | ~150 Hz | `TORSO_BAND_HZ = (0, 200)` |
| Feet | ~3–4 m/s | up to ±500 Hz | `FOOT_BAND_HZ = (200, 500)` |

So a spectrogram is literally a **picture of how fast each part of the body was
moving, at each instant**. Bright pixels near the centre = torso; bright
excursions far from centre = feet swinging.

We get **two channels per recording** — `ce_foot` (from foot-aimed radars) and
`ce_torso` (torso-aimed radar), because the network uses separate nodes
(`N_FOOT_RADARS = 2`, `N_TORSO_RADARS = 1`).

## 1.2 Why LOSO — the single most important methodological choice

**The threat:** micro-Doppler encodes *identity* extremely well. The literature
recognises individuals from gait radar at 93–97%. If any recording from subject
X appears in training while another recording of X is in test, the model can
memorise "this is X, and X is a patient" — scoring high without learning
anything about Parkinson's. We confirmed this fingerprint exists in our own data:
using only the 17 duration-invariant features, a subject's own trials sit a median
distance of **1.811** apart against **4.593** to other people's, and this holds for
**58 / 58 subjects** (`11_within_vs_between.png`,
`outputs/metrics/subject_fingerprint.json`).

**The defence — Leave-One-Subject-Out:**

```
for each of the 58 subjects:
      train on the other 57 → test on this one → record their score
finally: compute AUC over the 58 subject scores
```

Every test subject is someone the model has literally never seen. This is
called **subject-independent** validation. It produces *lower* numbers than
random splitting — that is the point: the lower number is the honest one.

**Provenance:** LOSO isn't anyone's invention; it's standard leave-one-out
cross-validation (Stone 1974, Geisser 1975) applied with the *subject* as the
group. It's `LeaveOneGroupOut` in scikit-learn ([baseline.py:102](src/baseline.py))
and hand-rolled in `loso_cv` for the deep track. What is *ours* is the decision
to enforce it here — much of this literature does not.

## 1.3 Subject-level aggregation

One subject has many recordings, and (after windowing) many windows. The model
scores each window. We then **average all of a subject's window probabilities
into a single score**, and compute metrics over the 58 subject scores — not over
thousands of windows.

Why: the clinical question is "is *this person* a patient?", not "is this 3-second
window patient-ish?". Also, scoring at window level would make the sample size
look artificially huge and the confidence intervals artificially tight.

Implemented at [train.py:143](src/train.py) (`subj_prob = np.mean(fold["val_probs"])`)
and [baseline.py:112](src/baseline.py) for the classical track.

## 1.4 Confounds — the central scientific problem

A **confound** is anything that differs between groups but isn't the disease,
that the model can exploit. We found three.

### Confound A — trial duration
PD subjects take longer. Strong in `test1` (+10%, significant — it includes the
chair stand-up), weak in `test2` (+3%, n.s.).

**Evidence of the danger:** a classifier using *nothing but recording length*
scores **AUC 0.61**. Our naive feature model scored 0.62. So the naive model was
almost entirely a stopwatch.

Four defences (in `confound_analysis.py`):
1. **Residualization** — regress each feature on duration, keep the residual
   ([confound_analysis.py:81](src/confound_analysis.py)).
2. **Duration-matched subset** — pair each PD trial with a control trial of
   near-identical length (0.75 s caliper), discard the rest
   ([confound_analysis.py:130](src/confound_analysis.py)).
3. **test2-only** — analyse the test where the confound is weak.
4. **Fixed-length windowing** — for the deep track, chop everything into
   identical 3-second windows so duration is **structurally absent**. This is
   the cleanest defence, and it's why windowing is Day 1.

### Confound B — age *(unresolved — we have no age data)*
The same 24 GHz micro-Doppler separates young from elderly at **94.9%**
(Hayashi 2021). PD cohorts skew older. **We cannot currently rule out that part
of our signal is an age detector.** This is the top ask of the supervisor and,
absent an answer, becomes a headline limitation.

### Confound C — subject identity
Handled by LOSO (§1.2).

## 1.5 Statistical honesty

- **Permutation test** — shuffle the PD/control labels hundreds of times,
  re-run the whole pipeline, build a distribution of "scores achievable by
  chance". Our real score sat in the top 2.2% → **p = 0.022**
  ([baseline.py:135](src/baseline.py)).
  *Correct reading:* "if labels were random, only ~2.2% of shuffles would score
  this high" — **not** "2.2% chance the result is luck."
- **Subject-level bootstrap CIs** — resample *subjects* (not trials) to get
  honest error bars ([baseline.py:121](src/baseline.py)).

## 1.6 Which metric, and why AUC

**AUC** (Area Under the ROC Curve): 0.5 = coin flip, 1.0 = perfect. It's
threshold-free and robust to the 33/25 class imbalance, and it's the field
standard — so it's the headline. But it is reported *alongside*:

| Metric | Meaning | Why report it |
|---|---|---|
| Sensitivity | % of patients correctly found | a screening tool must not miss patients |
| Specificity | % of healthy correctly cleared | avoid false alarms |
| Balanced accuracy | imbalance-corrected accuracy | single intuitive number |
| Confusion matrix | the raw 2×2 counts | shows *which* group fails |

All computed in `subject_metrics` ([baseline.py:117](src/baseline.py)).

> ⚠️ **Careful when comparing to papers:** most report **accuracy (%)**, we report
> **AUC**. They are different scales — 0.80 AUC is not "worse than their 95%".

## 1.7 Two model tiers, and why both

| Tier | What | Status |
|---|---|---|
| **Classical** | Hand-designed features (band energies, spectral shape) → logistic regression / SVM / random forest | ✅ done — AUC 0.62 raw, 0.68 duration-adjusted (p=0.022), 0.62–0.64 duration-matched |
| **Deep** | The spectrogram image itself → CNN / ResNet-18 | ⚠️ built, never run |

The classical tier exists to be the **floor**. Without it, a deep AUC of 0.70
sounds impressive; with it, you know whether 0.70 beat a simple model or not.
This is a proper scientific control, and most papers in this niche skip it.

---

# Part 2 — The engineering choices

## 2.0 The data journey

```
shared_ziad/<subject>/<file>.mat        106 GB, 348 files, MATLAB v5
        │  scipy.io.loadmat        (data_loader.py)
        ▼
ce_foot, ce_torso                       (320 Doppler bins × N time bins)
        │  slice into 3 s windows   (preprocessing.window_indices)
        ▼
window                                  (320 × ~time-bins-in-3s)
        │  log1p + z-score          (preprocessing.normalise_window)
        │  bilinear resize          (preprocessing.resize_2d)
        ▼
(224 × 224) per channel
        │  stack foot + torso       (preprocessing.process_trial)
        ▼
(2, 224, 224) float32  →  saved as .npy  +  a row in manifest.csv
        │  WindowDataset reads manifest  (dataset.py)
        ▼
PyTorch tensor → SmallCNN / ResNet-18 (models.py) → train (train.py)
```

The **manifest CSV is the single source of truth** — it records, for every
window: subject, group, label, test, trial, window index, time span, and file
path. That's what makes LOSO a one-line filter
(`manifest[manifest.subject_id.isin(train_subjects)]`) and makes every
experiment auditable.

**Why cache to `.npy`:** the 106 GB of `.mat` files are opened exactly **once**.
Every subsequent experiment reads small pre-computed arrays. Without this, 58
LOSO folds × many epochs would re-decode MATLAB files endlessly. This single
decision is what makes the two-week plan possible on a laptop.

## 2.1 Windowing — 3 s, 1.5 s stride

```python
window_s: float = 3.0     # ~2-3 gait cycles
stride_s: float = 1.5     # 50% overlap
```
([preprocessing.py:37-38](src/preprocessing.py))

**Four reasons:**
1. **Weakens the duration confound.** Every sample is exactly 3 s, so "PD walks
   longer" cannot be read off *within* a window.
   > 🔴 **Correction — an earlier version of this document claimed windowing kills
   > the duration confound "by construction." That is FALSE, and I verified it:**
   > longer trials yield *more* windows — control **27.3** vs PD **30.9** windows
   > per subject (ratio 1.13), and **window count alone gives AUC 0.621** — the
   > same level as the classical baseline being chased. Windowing decorrelates
   > sample *length* from label, **not the number of samples**.
   > **Must do:** cap windows per subject at K (or weight the loss by
   > `1/n_windows`), and report the 0.621 window-count floor as the deep track's
   > null floor, just as duration-only 0.61 is the classical floor.
2. **Data amplification.** 348 recordings is far too few to train a CNN. At 3 s
   with 50% overlap, each recording yields many windows → thousands of samples.
3. **Fixed tensor shape.** Networks need constant input size.
4. **3 s ≈ 2–3 gait cycles** — long enough to contain the rhythm the disease
   affects, short enough to be plentiful.

**The cost — be honest about it in the thesis:** overlapping windows from one
recording are highly correlated, so the *effective* sample size is much smaller
than the window count. LOSO protects the *validity* (no subject spans the
split), but don't claim "we have 5,000 independent samples."

## 2.2 Normalisation — `log1p` then per-window z-score

```python
x = np.log1p(np.maximum(arr, 0))
return (x - x.mean()) / (x.std() + 1e-6)
```
([preprocessing.py:64-67](src/preprocessing.py))

- **`log1p` (log compression):** radar power spans orders of magnitude; the torso
  return dwarfs the feet. A log squashes that range so faint but informative
  foot excursions aren't numerically invisible. `log1p` = `log(1+x)`, safe at 0.
- **Per-window z-score:** puts every window at mean 0, std 1 — what optimisers
  and BatchNorm expect.

**Why per-window and not a global mean/std?** A global statistic computed over
the whole dataset would be computed over *training and test data together* —
a subtle **data leak**. Per-window normalisation is self-contained and provably
leak-free. Good choice.

> ⚠️ **The trade-off you must discuss in the thesis.** Per-window z-scoring
> **removes absolute amplitude**. But reduced movement amplitude
> (**bradykinesia**) is a *defining symptom of Parkinson's*. We may be deleting
> some of the very signal we're hunting — and the literature explicitly warns
> that aggressive normalisation can discard the pathology.
> **Mitigation:** run one ablation with `normalise="log_minmax"` or `"none"`
> (both already supported) and report the comparison. Cheap, and it turns a
> weakness into a methodological contribution.

## 2.3 Resize to 224×224

ResNet-18 was pretrained on 224×224 ImageNet photos; matching that size lets the
pretrained filters work as intended.

**Cost:** the Doppler axis is squeezed 320 → 224 (≈30% of frequency resolution
gone), and time is stretched/squashed to 224. Given the true intrinsic Doppler
resolution is 20 Hz while bins are spaced 5 Hz (4× zero-padded — see
[radar_params.py:39-43](src/radar_params.py)), we're mostly discarding
interpolated detail rather than real information. Defensible — but say so
explicitly rather than leaving it silent.

## 2.4 Two channels, like RGB

`np.stack([foot_w, torso_w], axis=0)` → shape `(2, 224, 224)`
([preprocessing.py:112](src/preprocessing.py))

A colour photo is `(3, H, W)` — red, green, blue stacked. Here it's
`(2, H, W)` — foot, torso. The CNN sees both simultaneously at every pixel
location and can learn **relationships between them**.

That matters scientifically: the reference paper found that the normal coupling
between torso velocity and heel-strike **breaks down in Parkinson's**. A
2-channel model can represent exactly that; two separate single-channel models
could not.

`WindowDataset.channel_mode` ([dataset.py:44-56](src/dataset.py)) supports
ablations — `"foot"`, `"torso"`, `"both"`, `"diff"` — so you can *measure* how
much each channel contributes. **Run this ablation; it makes a great thesis
table.**

## 2.5 Two architectures

### SmallCNN — the from-scratch control
~50k parameters, 3 conv blocks, global average pooling
([models.py:16](src/models.py)).

- **Small on purpose.** With 58 subjects, a big network memorises. Fewer
  parameters = less overfitting.
- **BatchNorm** after each conv → stable, faster training.
- **Global average pooling** instead of a big flatten→dense layer: drastically
  fewer parameters and some translation invariance (a gait event matters
  wherever in the window it occurs).
- **Dropout 0.3** before the classifier.

### ResNet-18 — transfer learning
([models.py:40](src/models.py))

Pretrained on ImageNet (millions of photos). Early layers of *any* vision
network learn generic edge/texture/gradient detectors — and a spectrogram has
edges and textures. So we reuse them rather than learning from scratch on tiny
data. This is the standard answer to small datasets, and it's exactly what
Seyfioğlu 2018 and Park 2016 established for micro-Doppler specifically.

**Two adaptations were required:**

**(a) 3-channel → 2-channel input.** ImageNet's first conv expects RGB. The code
averages the three pretrained RGB filters and copies that average into both our
channels:
```python
avg = old.weight.mean(dim=1, keepdim=True)   # (64,1,7,7)
new.weight.copy_(avg.repeat(1, in_channels, 1, 1))
```
([models.py:63-64](src/models.py)) — this preserves the learned *spatial* filter
shapes instead of random-initialising them.
*Minor note:* summing over 2 channels instead of 3 scales activations by ~2/3;
the following BatchNorm absorbs it, so it's harmless here.

**(b) Freezing.** `freeze_until="layer3"` leaves only `layer4` + the classifier
trainable ([models.py:69-76](src/models.py)). With ~58 subjects, fine-tuning all
11M parameters would overfit instantly. Freezing the generic early layers and
adapting only the last, most task-specific block is the right call — and it
trains much faster.

## 2.6 Augmentation

```python
if rand < 0.5:  arr = arr[..., ::-1]              # time flip
if rand < 0.7:  arr += N(0, 0.02)                 # Gaussian noise
```
([dataset.py:58-64](src/dataset.py))

Noise injection: sensible, mimics receiver noise, improves robustness.

**Time flip: question it.** The last axis is time, so this plays the walk
backwards. Gait is *not* time-symmetric — heel-strike→toe-off is directional,
and PD gait asymmetry is clinically meaningful. Either justify it (micro-Doppler
of steady walking is quasi-periodic) or disable it and compare. **Cheap ablation,
and a reviewer will ask.**

## 2.7 Why LOSO is expensive, and the 5-fold trick

LOSO = **58 complete training runs per experiment**. Multiply by CNN + ResNet ×
several confound variants and you have far more compute than a fortnight allows
if done naively.

**Strategy:**
- **Develop and tune with 5-fold grouped CV** (5 runs, minutes) — for debugging,
  learning rates, epochs, architecture choices.
- **Run full 58-fold LOSO once per final configuration**, overnight, for the
  numbers that go in the thesis.

Grouped 5-fold keeps whole subjects inside a fold, so it's still
subject-independent — just coarser. This is a legitimate, documented
development/reporting split.

---

# Part 3 — Why the plan is ordered the way it is

## The dependency chain

```
Fix train.py bugs
      ↓
Preprocessing (Day 1)         ← everything downstream needs the .npy cache
      ↓
CNN 5-fold (Day 2)            ← cheap; proves the pipeline works end-to-end
      ↓
CNN LOSO overnight (Day 3)    ← expensive; only after the cheap version works
      ↓
ResNet 5-fold → LOSO (Day 4)
      ↓
Grad-CAM (Day 5)              ← needs a *trained* model to explain
      ↓
Confound battery (Day 6)      ← needs final models
      ↓
Results → Discussion → Intro/Abstract (Days 7-9)
      ↓
Assemble & send (Day 10)
```

Three ordering principles:

1. **Cheap before expensive.** Never launch a 58-fold run before a 5-fold run has
   proved the code works. A bug found after 6 hours costs a day; found after 6
   minutes it costs nothing.
2. **Long jobs run overnight.** Launch LOSO at end of day, read it next morning.
   Your working hours go to thinking, the machine's night hours to computing.
3. **Write while the machine works.** Training is the perfect time to write —
   the Methods chapter describes exactly what's running. By end of Week 1 the
   entire Methods chapter exists, so Week 2 only needs results-dependent
   chapters.

## Why Abstract is written last

An abstract summarises what you actually found. Written first, it becomes a
promise you then have to keep. Written on Day 9, it's a factual summary.

## Why the age analysis is isolated to Day 9

It's the only piece that depends on Ignacio replying. Every other task is
self-contained. If the metadata arrives → drop it in on Day 9. If it never
arrives → it becomes a documented limitation and nothing else in the plan
shifts. **The plan cannot be blocked by an unanswered email.**

## What "done" means if the deep model disappoints

If CNN/ResNet don't beat the 0.62–0.68 classical floor after honest confound
control, **that is a legitimate result, not a failure.** The thesis contributes:
the first subject-independent radar PD classifier, a confound-controlled
methodology, and an honest measurement of how much signal is really there. The
literature is full of inflated numbers from leaky validation; a careful, lower,
*correct* number is a genuine contribution — and precisely why Q8 to the
supervisor asks him to confirm this framing in advance.

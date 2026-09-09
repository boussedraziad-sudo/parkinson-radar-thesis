# The data, explained — what the model actually sees

*Companion to `HANDBOOK.md`. This document is about **the data itself**: how to
read a spectrogram, what the foot and torso channels each capture, what is
physically visible in a real recording, and — in plain language — what every
confound-control term means.*

**Key figure:** `outputs/figures/guide_data_explained.png` — a real control and a
real PD subject, both channels, matched for duration (8.0 s vs 8.1 s).

---

# Part 1 · How to read one spectrogram

A spectrogram here is **not a picture of a person**. It is a chart with three
dimensions:

```
        ↑ Doppler frequency (Hz)  =  SPEED of whatever is moving
        │                            (+ = toward radar, − = away)
        │        brightness = HOW MUCH of the body
        │                     is moving at that speed, right now
        └──────────────────────────→  time (s)
```

**The vertical axis is speed.** The conversion is exact:

```
velocity (m/s) = Doppler (Hz) × λ / 2 ,   λ ≈ 1.3 cm at 23 GHz
```

| Doppler | Speed | What moves that fast |
|---|---|---|
| ~50 Hz | 0.33 m/s | slow torso drift |
| ~150 Hz | ~1 m/s | **torso, normal walking pace** |
| ~300 Hz | ~2 m/s | thigh / lower leg mid-swing |
| ~500 Hz | ~3.3 m/s | **foot at peak swing** |

So the two horizontal reference lines in the figure are physiology, not decoration:
- **±200 Hz (white)** — above this, motion is too fast to be the torso → it's limbs
- **200–500 Hz (orange)** — the **foot band**

---

# Part 2 · What is actually visible in a real recording

Look at `guide_data_explained.png`. Three things are immediately readable, and
they're worth being able to point to in your defence.

### 2.1 The turn is visible — and it splits every recording in two

In **all four panels**, the energy is **below zero** for the first half and
**above zero** for the second half. That is not noise, it is the **TUG turn**:

- **Negative Doppler** = walking **away** from the radar
- The sign flip = **the moment the subject turns around**
- **Positive Doppler** = walking **back toward** the radar

Control turns at ≈ 6.2 s; the PD subject at ≈ 7.6 s. **Every trial contains two
walking bouts and one turn.** This matters: the turn is the slowest, most
unstable part of the walk, and turning difficulty is a classic parkinsonian
symptom. It's also the part most contaminated in `test1` by the chair stand-up.

### 2.2 The scalloped arcs are individual steps

In the **foot** panels, the bright band isn't smooth — it has repeating
**arches**. Each arch is one **foot swing**: the foot accelerates from rest,
reaches peak speed mid-swing, then decelerates to zero at heel-strike. The
**valleys between arches are the moments the foot is planted** (speed ≈ 0).

So you can literally **count steps** off the image, and the *spacing* of the
arches is the step rhythm — the thing gait analysis cares about.

### 2.3 Control vs PD — what differs in this example

| | Control (`fisc_048`) | PD (`fisp_024`) |
|---|---|---|
| Foot arcs | **crisp, tall, regular** — clean excursions to ±500 Hz | **less regular**, more fragmented |
| Background | dark, clean | **visibly noisier / speckled** |
| Torso band | tight, well-defined | broader, more diffuse |

> ⚠️ **Read that third row carefully — and be suspicious of the second.**
> The PD example has a much noisier background. That could be pathology… **or it
> could be the acquisition-batch difference** (§3.5). This is exactly the kind of
> "obvious visual difference" that turns out to be a recording artifact. **Never
> conclude from one pair of images.** This is why the confound battery exists.

---

# Part 3 · Foot vs torso — why two channels

The radar network has **2 foot-aimed nodes and 1 torso-aimed node**, mounted at
different heights (feet ≈ 0.15 m, torso ≈ 1.0 m). They see the same walk from
different angles, so they capture different physics:

| | `ce_foot` | `ce_torso` |
|---|---|---|
| Aimed at | ankles / feet | chest / trunk |
| Dominant energy | **200–500 Hz** (fast swings) | **0–200 Hz** (steady glide) |
| Visual character | **scalloped arcs** — periodic, spiky | **one smooth band** |
| Encodes | step timing, swing amplitude, asymmetry | walking speed, trunk steadiness |

**Why the model gets both stacked together** (like the R, G, B of a colour image):
so the network can learn **the relationship between them**. That's not decorative
— the reference paper's central finding is that **the coupling between torso
velocity and heel-strike timing breaks down in Parkinson's**. A single-channel
model cannot represent a *relationship*; a 2-channel model can.

---

# Part 4 · What the model actually receives

The raw spectrogram is not fed in directly. Each recording becomes many small
standardised tiles:

```
full recording        (320 Doppler bins × ~1000 time bins, ~9 s)
   │  cut into 3-second windows, sliding 1.5 s at a time
   ▼
window                (320 × ~330)
   │  log1p compression   → stop the loud torso drowning the faint feet
   │  z-score            → mean 0, std 1
   │  resize             → 224 × 224
   ▼
stack foot + torso    → (2, 224, 224)  ← ONE training example
```

**So the model never sees a whole walk.** It sees a **3-second snapshot** — about
2–3 steps — and must decide "PD-ish or not?" from that alone. The per-subject
answer is the **average over all their windows**.

**Two consequences worth stating in your thesis:**
1. The model **cannot** use total walk duration — it's not in a fixed-length
   window. *(But see §3.6 — window **count** still leaks it.)*
2. Per-window z-scoring **deletes absolute amplitude**. Since reduced amplitude
   (**bradykinesia**) is a defining PD symptom, we may be erasing signal. That's
   why the normalisation ablation is in the plan.

---

# Part 5 · Every confound-control term, in plain language

A **confound** is anything that differs between the two groups but *isn't the
disease*, which the model can exploit to look clever. Each control below removes
one — and each has a **null floor**: the score you'd get from the confound alone.

### 5.1 LOSO — Leave-One-Subject-Out
**The problem:** your walk is a fingerprint. Measured: within-subject distance
2.04 vs between-subject 5.23, and **58/58 subjects** are more similar to
themselves than to anyone else. If a subject's trial 1 is in training and trial 2
in testing, the model recognises *the person*, not the disease.

**The control:** train on 57 subjects, test on the 58th, repeat 58 times. Every
test subject is someone the model has never seen.

### 5.2 `duration_only` — the stopwatch baseline
**The problem:** PD subjects take longer (test1: +10.1%, p=0.0016).

**The control:** build a classifier that uses **nothing but recording length**.
It scores **AUC 0.610**. That's the floor — *any* model must beat a stopwatch to
have learned anything about gait.

### 5.3 Duration-**matched** — the strictest fairness test
**In one sentence:** *pair each PD trial with a control trial that took almost
exactly as long, throw away everything unpaired, and re-run.*

Mechanically: for each PD trial, find the unused control trial whose duration is
closest; accept the pair only if the gap is under a **caliper** (0.75 s). Result:
270 trials, and afterwards the two groups' durations are **statistically
identical** (Mann-Whitney p = 0.95).

**Why it's strong:** it assumes nothing about *how* duration affects the features
— it just constructs a subsample where duration cannot possibly be the tell. If
the score survives, duration wasn't the explanation.
**Result:** 0.621 → **0.624**. Survived.

### 5.4 Duration-**residualized** — the mathematical version
**In one sentence:** *for each feature, fit a straight line predicting it from
duration, subtract that line off, and classify on what's left over.*

The leftover ("residual") is the part of the feature that duration **cannot**
explain. **Weaker than matching** because it only removes *linear* dependence.
**Result:** 0.621 → **0.676** — removing duration *raised* the score, meaning
duration was **masking** the gait signal, not creating it.

### 5.5 `test2`-only — use the naturally cleaner condition
`test1` includes standing up from a chair, where the duration gap lives (+10.1%).
`test2` is pure walking (+3.4%, not significant). Analysing `test2` alone is a
confound-free condition that costs nothing but sample size.
**Result:** duration alone drops to 0.536 (chance) while gait features hold 0.589.

### 5.6 Window-count control
**The problem I got wrong at first.** Fixed-length windows remove duration from
*inside* a window — but a longer trial yields **more windows**. Measured: control
**27.3** vs PD **30.9** windows per subject. **Window count alone → AUC 0.621**,
the same as the whole classical model.

**The control:** cap every subject at the same number of windows, or weight the
loss by `1/n_windows`.

### 5.7 🔴 Within-**batch** — the one that changed the conclusion
**The problem:** the `.mat` files contain **two slightly different Doppler axes**
— a fingerprint showing the cohort was recorded in **two separate acquisition
groups**. All 58 subjects are pure to one group, and the groups have very
different composition:

| Group | Subjects | control | PD | % PD |
|---|---|---|---|---|
| A | 34 | 24 | 10 | 29.4 % |
| B | 24 | 9 | 15 | 62.5 % |

**Batch membership alone → AUC 0.664** — better than every model built.

**The control:** run the whole analysis **inside group A only**, then **inside
group B only**. Within one group, batch is constant, so it cannot be the tell.

| Feature set | pooled | batch A | batch B |
|---|---|---|---|
| `shape_clean` | 0.621 | **0.537** | **0.541** |
| `duration_only` | 0.610 | **0.642** | **0.659** |

**The gait-shape features collapse to chance. Duration does not — it gets
stronger.** So slowness looks like a real, batch-robust PD marker; the
spectral-shape descriptors mostly were not.

> **Important nuance:** the *axis difference itself* (0.0014 Hz) is physically
> meaningless — it is a **label**, not a cause. What matters is that it reveals
> two groups with different PD proportions. And with only 34 and 24 subjects per
> group, "chance" is also consistent with *a weak signal too small to detect at
> this n*. Don't over-claim the negative either.

### 5.8 Permutation test — is it luck?
Shuffle the PD/control labels at random, re-run the **entire** pipeline, record
the AUC. Do it 1000 times. That builds the distribution of "scores achievable by
pure chance on this exact data."

**p = 0.022** means: *if the labels were random, only ~2.2 % of shuffles would
score this high.*
❌ It does **not** mean "there's a 2.2 % chance the result is luck." Don't write
that in the thesis — a signal-processing committee will notice.

### 5.9 Bootstrap CI — how precise is the number?
Resample the **58 subjects** (not the trials) with replacement 2000 times,
recompute AUC each time, take the 2.5th–97.5th percentiles. Resampling *subjects*
is essential because trials from one person are correlated; resampling trials
would give falsely narrow intervals.
**Result:** 0.621 with 95 % CI **[0.473, 0.770]** — the interval **contains
chance**, which is itself an honest and important statement.

---

# Part 6 · The summary table to keep in your head

| Control | Removes | Null floor | Result |
|---|---|---|---|
| **LOSO** | subject identity | — | mandatory; everything uses it |
| `duration_only` | — | **0.610** | the stopwatch |
| Duration-matched | duration (non-parametric) | 0.610 | 0.624 — survives |
| Residualized | duration (linear) | 0.610 | 0.676 — *improves* |
| `test2`-only | the chair stand-up | 0.536 | 0.589 — survives |
| Window-count | more windows per longer trial | **0.621** | to be applied |
| 🔴 **Within-batch** | acquisition group | **0.664** | **0.537 / 0.541 — collapses** |

**The one sentence for your defence:**
> *We measured four separate ways this dataset could fool a classifier, gave each
> one an explicit null floor, and reported performance against those floors rather
> than against chance.*

That sentence is the thesis. It is also exactly the "clarity" the coordinator
asked for — a committee from signal processing will immediately recognise it as
rigorous experimental design, even if they've never seen a UPDRS score.

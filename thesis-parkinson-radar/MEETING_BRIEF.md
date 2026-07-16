# Supervisor meeting brief

*Project: detecting Parkinson's disease from radar "gait movies." Prepared for the
meeting with Prof. Godino-Llorente (Nacho).*

---

## 0. How to use this document

- **Section 1** is a 60-second plain-language primer — read it first so the terms
  below feel familiar.
- **Section 2** is what we've found so far (things to *report*).
- **Section 3** is the questions. Each one is written as: *what we know → what I
  want to ask → why it matters → what we're currently assuming.* That last line
  matters: instead of an open question, it gives Nacho something concrete to
  agree with, correct, or redirect. You can literally read the **"My question"**
  line out loud.
- **Section 4** is what to put on screen during the call.
- **Section 5** is the single most important thing to leave the meeting with.

**Priority if the meeting runs short:** two things have long lead times and
should be *started today* — the **subject metadata (Q1)** and **GPU access
(Q13)**. After those, go **3A (data) → 3C (expectations) → 3B (methodology) →
3D (logistics).**

---

## 1. 60-second primer (so the jargon below makes sense)

- **The data are not photos of people.** A radar sends out a wave, it bounces off
  a moving body, and comes back slightly shifted in frequency depending on how
  fast each body part moves toward or away from the radar. We turn that into a
  picture called a **spectrogram** — time on one axis, speed on the other,
  brightness = how much of the body is moving at that speed. Think of it as a
  "movie of movement," not an image of a person. *(The technical name for this
  movement-frequency signal is **micro-Doppler** — you'll see that term in the
  literature.)*
- **Two views per recording.** One spectrogram focuses on the **feet**
  (`ce_foot`), one on the **torso** (`ce_torso`). "ce" = *contrast-enhanced*,
  i.e. the brightness has been rescaled to make the pattern easier to see.
- **The task.** Look at these movement-movies and decide: is this person a
  **Parkinson's patient** or a **healthy control**? That is our binary
  classifier.
- **The trap we keep fighting — "confounds."** A confound is something that
  differs between the two groups but *isn't the disease itself*, that the model
  can cheat off. Example: if patients simply take longer to do the walk, the
  model can "detect Parkinson's" by secretly measuring **time**, not gait. We
  found and controlled several of these.
- **Why we test the way we do — "LOSO."** Everyone's walk is as unique as a
  fingerprint. If the same person appears in both the practice set and the exam,
  the model recognises *the person*, not *the disease*, and the score looks great
  but is fake. So we train on everyone **except one subject**, test on that
  held-out subject, and repeat for all 58. This is **Leave-One-Subject-Out
  (LOSO)**. A result tested this way is called **"subject-independent"** — the
  honest kind.
- **How we score — "AUC."** A number from 0.5 (pure coin-flip, useless) to 1.0
  (perfect). 0.62 means "a bit better than a coin flip." 0.70–0.80 would be a
  respectable, trustworthy result for a hard medical problem with few subjects.
  *(Heads-up: papers in this field usually report **accuracy (%)** instead of AUC.
  They're related but not the same scale — don't read "0.80 AUC" as "worse than
  their 95%.")*

---

## 2. What we've found so far (to report)

### The headline
We understand the data, we've built a careful first ("baseline") model, and we've
done a full literature review. The signal that separates Parkinson's from healthy
is **real but weak, and it survives our cheating-checks** — meaning it's not
*only* the confounds. The main things standing between us and a stronger,
credible result are **missing information about the subjects (their age, etc.)**
and **the number of subjects (58)**.

### The dataset
- **58 people:** 33 healthy controls (`fisc…`) + 25 Parkinson's patients
  (`fisp…`).
- Each did a standard clinical walk (**Timed-Up-and-Go**) in **two versions** —
  one standing up from a chair (`test1`), one without the chair (`test2`) —
  **3 times each**, giving roughly **348 recordings**.
- Every recording produces **2 spectrograms** (feet + torso).
- Two small oddities: one patient (`fisp_022`) has an extra recording, one
  (`fisp_048`) is missing one. *(We'll just keep both and note them — no action
  needed from Nacho; see the FYI box in Section 3.)*

### What we learned about the data (all checked in code, not guessed)
1. **The duration confound.** Patients take longer to do the walk. This is
   strong in `test1` (the chair version, +10%, statistically real) and small in
   `test2` (+3%, not significant). → A naive model could "detect Parkinson's"
   just by measuring how long the recording is. We had to neutralise this.
2. **The "ce" brightness rescaling is done per-recording and we don't have the
   formula.** It isn't described in the reference paper. That's a provenance gap:
   we can't fully reproduce or trust the exact pixel values until we know how
   they were made.
3. **Strong personal fingerprint.** Confirms we *must* use LOSO (above); an
   easier test would have given us an impressive but fake score.

### Our first model (the "classical baseline," already run)
Plain-language version: we measured simple, hand-picked numbers from each
spectrogram (how much energy in each speed band, etc.) and trained a standard
classifier, tested with LOSO. Read these numbers as a *story*, not a list:
- **Simple features → AUC ≈ 0.62** (a bit better than chance).
- A predictor using **only recording length** already reaches **≈ 0.61** — the
  "cheating floor." So those simple features are *almost entirely just duration*.
- A **richer** feature set reaches **≈ 0.68**, but part of that is still duration
  leaking in.
- **The key test:** when we **statistically adjust out the effect of duration**
  from that richer set, we **still get ≈ 0.68** — and a *permutation test* (we
  reshuffle the patient/control labels thousands of times and re-score) says that
  **if the labels were random, only about 2.2% of shuffles would score this high
  (p = 0.022)**. So the 0.68 is unlikely to be a fluke and is *not* just
  duration.
- **The strictest check** — comparing only patients and controls **matched for
  walk duration** — drops to **≈ 0.62–0.64**: lower than 0.68, but still *above*
  the 0.61 duration-only floor. So a **real, if weaker, non-duration signal
  remains.**
- Mistakes are evenly split (11 control / 9 PD of 20) — not biased to one group.
  The "learning curve" is **still rising at 58 subjects** → **more data would
  likely help.**
- **Bottom line:** a weak-but-real Parkinson's signal in the gait, not just
  "they're slower." The honest range is roughly **0.62 (strictest) to 0.68
  (duration-adjusted)** — and that's the floor the fancy deep model must beat.

### The deep-learning model
Built and ready — **a convolutional neural network and a ResNet** (both standard
image-recognition networks; the ResNet uses **transfer learning**, meaning we
reuse a network already trained on millions of everyday photos and adapt it to our
spectrograms). **Not yet trained** — that needs a GPU machine and, ideally, the
missing metadata so we can control confounds properly.

### The literature (15 verified papers)
- **Nobody has published a subject-independent Parkinson's-vs-control classifier
  from radar micro-Doppler** (the movement-frequency signal from Section 1). Our
  project is genuinely novel.
- Our testing discipline (LOSO + confound checks) is **more rigorous than most
  papers in this niche**, many of which report flashy 90%+ *accuracy* using leaky
  testing.
- **Big warning from the literature:** the *same kind of radar* can tell **young
  from elderly** people at **94.9%** accuracy. Parkinson's patients are usually
  older than controls. So **if we don't control for age, we might build an
  age-detector wearing a Parkinson's-detector costume.** This is why age metadata
  is our top ask.

---

## 3. Questions & decisions for Nacho

> Each item: context → **My question** (say this) → why it matters → what we're
> assuming. He can answer "yes / no / do it this way instead."

> **FYI — things we'll just handle ourselves (no decision needed):** the two data
> oddities (`fisp_022` extra trial, `fisp_048` missing trial) — we'll keep both
> and document them. Mention only if he asks.

### 3A. Data & subject information — *highest priority, this gates everything*

**Q1 — Subject metadata (age is the must-have).**
- *Context:* We have the radar recordings but almost nothing *about* each person.
  The literature shows age alone is a huge confound.
- **My question (must-have):** "Can we get each subject's **age and sex** — who
  owns that data and when can I have it?"
- **My question (also useful if it exists):** "And if available, their Parkinson's
  **severity** — on either standard doctor's rating scale, **UPDRS-III** or
  **Hoehn & Yahr** (both are just standard severity ratings) — and whether they
  were **on or off their medication** during recording?"
- *Why it matters:* Without age especially, we cannot prove the model detects the
  *disease* rather than *being older*. This single item determines how strong a
  claim the thesis can make. *Get an unambiguous "yes" on age before moving on.*
- *Our assumption:* Age/sex exist in the records and can be shared anonymised;
  severity/medication may be harder.

**Q2 — Can we get more subjects (or a power analysis)?**
- *Context:* Our learning curve is **still climbing at 58 subjects** — the model
  keeps improving as we add people, so we're *data-limited*, not method-limited.
- **My question:** "Since performance is still improving with every subject we
  add, **can more participants be recruited, and by when?** If not, should I run a
  **power analysis** to estimate how many subjects we'd need to reach a defensible
  0.70–0.80?"
- *Why it matters:* This is possibly the single biggest lever on the final
  result, and only you can authorise recruitment.
- *Our assumption:* Recruitment may be limited; we'll report a power analysis
  either way.

**Q3 — The "ce" recipe, the raw spectrograms, and the reference pipeline.**
- *Context:* The spectrogram brightness was rescaled per-recording by a method
  not in the paper, and we have no external result to compare our 0.68 against.
- **My question:** "How exactly were the `ce_foot` / `ce_torso` images generated
  — is there a **script or formula**, and do we have the **raw (un-enhanced)**
  spectrograms? And did **López-Delgado 2026 report any classification numbers**
  on this data, with a **reference pipeline** we can reproduce and benchmark
  against?"
- *Why it matters:* We need this to trust/reproduce the data, the raw version may
  hold information the enhancement discards, and a reference number gives our
  result an anchor.
- *Our assumption:* A preprocessing script exists (with the López-Delgado team);
  the paper reports gait *parameters* but no classifier.

**Q4 — The prodromal / early-stage data.**
- *Context:* There's mention of a Phase-2 "prodromal" group (`fis_…`, people who
  may develop Parkinson's but aren't diagnosed yet).
- **My question:** "When (if at all) will the **prodromal (early / pre-diagnosis)
  data** arrive, and should the thesis plan around it?"
- *Why it matters:* It changes scope — a second, harder task, or a nice-to-have.
- *Our assumption:* Future/optional; we plan on the current 58 subjects and treat
  prodromal as a bonus.

### 3B. Methodology — *confirm we're on the right track*

**Q5 — Scope of the classification task.**
- **My question:** "Core task is **Parkinson's-vs-control**, agreed? And *if we
  get age*, is a **young-vs-older control** comparison worth adding as a sanity
  check (to show how much of our signal is age), or out of scope?"
- *Why it matters:* Defines exactly what we build and report; the young-vs-older
  check directly addresses the age-confound story.
- *Our assumption:* Binary Parkinson's-vs-control is the core deliverable; the
  age check is a valuable add-on if age arrives.

**Q6 — Which walk test is the 'main' one.**
- *Context:* `test1` has the chair (and the bigger duration confound); `test2` is
  cleaner walking.
- **My question:** "Should we treat **`test2` (no chair)** as the primary, cleaner
  benchmark and report `test1` as secondary?"
- *Why it matters:* Picking the cleaner test up front avoids a confounded headline
  number.
- *Our assumption:* `test2` primary, both reported.

**Q7 — What the headline number should be.**
- *Context:* This depends on Q1 (we don't have age yet). An age-matched subset of
  58 people would be *small*, which brings its own uncertainty (a "power" risk).
- **My question:** "*Assuming we get age* — would you lead the thesis with the
  **age-matched LOSO** number and report the full 58-subject LOSO alongside? Or,
  given how small a matched subset gets, prefer the **age-adjusted (residualized)
  full-cohort** number as the headline?"
- *Why it matters:* This is the number that goes in the abstract; we want your
  buy-in on the approach now.
- *Our assumption:* Report both; lead with whichever is more defensible once we
  see how small the matched subset is.

**Q8 — Our confound-control checklist.**
- *Context:* We stress-test every model with four checks: duration-matched,
  test2-only, age-matched, and a **label-shuffle** sanity check (the permutation
  test from Section 2).
- **My question:** "Do you agree we should run the **same set of checks
  (duration-matched, test2-only, age-matched, label-shuffle) on every model,
  including the deep network** — even when it lowers the numbers?"
- *Why it matters:* This rigour *is* our core contribution; want it endorsed.
- *Our assumption:* Yes — rigour over flashy numbers.

**Q9 — Extra methods worth adding.**
- *Context:* Two optional additions of different cost/value.
- **My question:** "Is it worth adding (a) an **LSTM** — a network that reads
  movement as a time-sequence — on the **velocity curves** (the speed-over-time
  line pulled from the spectrogram), like Hayashi 2021; and (b) **Grad-CAM** —
  heatmaps that colour *which part of the image the network actually used*? **If
  only one, which do you prioritise?**"
- *Why it matters:* Grad-CAM catches the network cheating on an artifact instead
  of gait (a documented failure in this field); the LSTM is a second modelling
  angle. Both cost time.
- *Our assumption:* Grad-CAM is important; the LSTM is nice-to-have.

### 3C. Scope & expectations — *align on what "success" means*

**Q10 — Realistic target.**
- **My question:** "With only 58 subjects, is a **well-validated 0.70–0.80 AUC**
  the right definition of success, rather than a headline 95%?"
- *Why it matters:* Sets expectations so a rigorous 0.75 isn't seen as
  'underperforming.'
- *Our assumption:* Yes; trustworthy beats flashy.

**Q11 — Is a rigorous, first-of-its-kind, possibly-weak result the accepted
framing?**
- **My question:** "Can we frame the thesis as **'the first subject-independent
  radar Parkinson's classifier (i.e. tested with LOSO, on people never seen in
  training) plus a confound-controlled methodology'** — and if the deep model,
  after honest confound control, only matches or barely beats the baseline, is
  that **rigorous weak/negative result** an acceptable thesis outcome?"
- *Why it matters:* Locks the thesis story *and* de-risks it in one decision — a
  lot of the value is the methodology regardless of the final number.
- *Our assumption:* Yes on both.

**Q12 — Publication & data rights.**
- *Context:* The data comes from a clinical collaboration (López-Delgado 2026)
  under ethics approval PD-RADAR-05/2024.
- **My question:** "Do you see a **paper** coming from this, and what would
  **authorship** look like? And are there any **restrictions on using or
  publishing** results from this dataset — an embargo until the dataset paper is
  out, a data-use agreement I must sign, or data owners who must be co-authors?"
- *Why it matters:* Affects how much polish/rigour to invest, and I shouldn't
  publish anything I'm not cleared to.
- *Our assumption:* Possible paper; we build to publishable standard and confirm
  rights before submitting anywhere.

### 3D. Logistics

**Q13 — Compute *(start this today — long lead time).*** "Is the **GPU
workstation (GAPS, via Camilo/Amber)** available for the deep-learning training
run, and can we start the access request now?" *(Nothing deep can be trained
without it.)*

**Q14 — Timeline.** "What are the **key deadlines and the next check-in**, and how
often should I send updates?"

**Q15 — Ethics.** "Am I **covered under the existing ethics approval
(PD-RADAR-05/2024)** as a thesis student working with this data?"

---

## 4. What to show him on screen (in this order)

1. **`LEARNING_GUIDE.html`** — the annotated spectrograms, for a 2-minute shared
   look at what the data actually is.
2. **The figures** — duration-confound plot, the baseline AUC grid, the
   confound-controlled (duration-matched / duration-adjusted) results, the
   learning curve.
3. **`RESULTS.md`** — the numbers and the "weak but real" conclusion.
4. **`RELATED_WORK.html`** — where we sit vs the literature + the age-confound
   warning.

## 5. The one thing to walk away with

**A commitment (or a concrete path) to the subject metadata table — above all,
each subject's age.** Everything about how strong a claim the thesis can make
depends on being able to prove we're detecting *Parkinson's* and not *age*. If
you get nothing else from the meeting, get this. *(Second closest: start the GPU
access request, since it's the other long-lead item.)*

---

### Quick checklist to tick off live
- [ ] **Age + sex** metadata — owner + date? *(must-have)* — severity/medication if it exists
- [ ] **More subjects** possible / by when — or run a power analysis?
- [ ] `ce` generation script + **raw spectrograms** — available? Reference-paper baseline to benchmark against?
- [ ] Prodromal data — timeline & whether to plan for it
- [ ] Primary test = `test2`? Binary task confirmed? Young-vs-older check if age arrives?
- [ ] Headline = age-matched vs age-adjusted LOSO? Confound checks on *all* models — agreed?
- [ ] Target 0.70–0.80 + rigorous negative result acceptable? Contribution framing OK?
- [ ] **GPU access** — start request now? Deadlines? Ethics coverage? Publication rights / embargo?

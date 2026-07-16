# Concepts & Notes — plain-language learning log

A beginner-friendly companion to `ONBOARDING.md`. This file answers the
"what does this actually mean?" questions, built up brick by brick. Nothing
here assumes radar or ML background.

---

## 0. Quick glossary (the words that keep coming up)

| Term | Plain meaning |
|---|---|
| **PD** | **Parkinson's Disease** — a brain disorder that slows and stiffens movement; it changes how people walk (smaller, slower, less regular steps). |
| **Control** | A **healthy** participant (no PD). We compare PD vs control. |
| **Radar** | A box that sends out a radio wave and listens to the echo bouncing off a moving body. |
| **Doppler shift** | How much the echo's frequency changes because the body part was moving. **More shift = faster movement.** (Same effect as a passing siren changing pitch.) |
| **Micro-Doppler** | The mix of many small Doppler shifts from different body parts (torso, arms, feet) all moving at different speeds at once. |
| **Spectrogram** | A picture of how those Doppler shifts change over time. This is the image we work with. |
| **STFT** | Short-Time Fourier Transform — the math that turns the raw echo into the spectrogram. |
| **ce** | **Contrast-enhanced** — the spectrogram after a brightness-boosting step so faint parts show up. `ce_foot`, `ce_torso`. |
| **trial / test** | One recording of a person walking (see §2). |
| **LOSO** | Leave-One-Subject-Out — the fair way we test the model (train on 57 people, test on the 1 left out, repeat). |

---

## 1. The dataset in numbers

- **58 participants total: 33 control (healthy) + 25 PD (patients).** Folder names: `fisc_*` = control, `fisp_*` = PD.
- **Each participant does 2 tests, each test repeated 3 times = 6 recordings ("trials") per person.**
  - `test1` = Timed-Up-and-Go **with a chair** (stand up → walk → turn → walk back → sit).
  - `test2` = Timed-Up-and-Go **without a chair** (start and end standing).
  - `trial1 / trial2 / trial3` = the three repeats of each test.
- **≈ 348 recordings total** (a couple of people have one extra or one missing — see `ONBOARDING.md`).
- So `fisp_012 / test2 / trial1, PD` means: **participant fisp_012, who has Parkinson's, doing the no-chair walk, first repeat.**

Each recording is stored as `stft_data.mat`, and inside it `ce_foot` and
`ce_torso` are the two spectrogram images we mostly use.

---

## 2. How to read a `ce_foot` spectrogram (the single most important skill)

A spectrogram is a **picture with three things**:

- **Horizontal axis = time** (seconds, left → right as the walk progresses).
- **Vertical axis = Doppler shift ≈ speed** of whatever reflected the wave.
  - Middle line (0 Hz) = not moving toward/away from the radar.
  - **Above 0 = moving toward the radar. Below 0 = moving away.** (The *sign* = direction.)
  - **Further from 0 (up or down) = faster.** (The *distance from the middle* = speed.)
- **Brightness/colour = intensity** — how much echo energy came back at that speed and time. Bright = strong reflection; dark = nothing moving at that speed then.

### What is "log(1+intensity)"?
Raw intensity values span a huge range (a few very bright pixels, lots of tiny
ones). If you plotted them directly, the image would be almost all black with a
couple of white dots. `log(1 + intensity)` **compresses that range** so both
strong and faint features are visible at once. The `1+` is just a trick so that
an intensity of 0 maps to 0 (log of 0 is undefined). **It's a display choice, not
new data** — like turning up the shadows on a photo.

### What are the "spikes"?
Each **spike / burst** is a **foot swinging through a step**. When you walk, your
planted foot is still (0 speed), but the swinging foot briefly moves *much* faster
than the rest of you — so it throws a burst of energy far from the centre line.
**One spike ≈ one step.** Between steps the foot is planted → no spike.

### "Steps are higher when walking where?"
The spike's **direction (up/down) tells which way you're walking**, and its
**height (distance from centre) tells how fast the foot moved**:
- Walking **toward** the radar → spikes appear **above** the centre (positive).
- Walking **away** → spikes appear **below** the centre (negative).
- In `fisp_012/test2/trial1`: first ~8 s the spikes are **below** (walking away),
  then a turn, then after ~8 s they jump **above** (walking back). That up/down
  flip is literally the person turning around.

### The everyday analogy
Think of a **music equalizer bar display, but for speed instead of pitch, scrolling
over time.** Slow, steady bars near the bottom = the trunk. Tall bars flashing
rhythmically = the feet kicking through each step. Read left-to-right and you're
watching a "speed movie" of the walk.

### What does the spike **energy** tell us?
- **How far the spike reaches** (up/down) = how fast the foot moved (its peak speed).
- **How bright the spike is** = how strong that fast motion's reflection was.
- **How regularly spikes repeat** = the walking rhythm/cadence.
So weak, short, irregular spikes suggest small, slow, uneven steps.

### Speed conversion (for intuition)
At this radar's wavelength (~13 mm), Doppler Hz → speed:
`speed = Doppler × 0.0065 m/s per Hz`.
- 200 Hz ≈ 1.3 m/s (typical torso / walking speed)
- 500 Hz ≈ 3.3 m/s (a fast foot swing)
- 800 Hz ≈ 5.2 m/s (the maximum the radar can measure here)

That's why the **torso lives near 0–200 Hz** (slow, steady) and the **feet reach
200–500 Hz** (fast swings).

---

## 3. Can you tell healthy vs PD by naked eye? (honest answer)

**Not reliably — and that's the whole reason we build a classifier.** But the
*things you'd look for* (all reduced/disrupted in PD, from the medical literature)
are:

1. **Foot-spike reach** — PD feet tend to swing **slower/smaller**, so spikes
   reach **less far** from the centre (lower peak speed).
2. **Rhythm regularity** — PD steps are often **less evenly spaced** (irregular
   spike timing).
3. **Left/right symmetry** — PD often makes one side weaker, so alternating
   spikes become **uneven**.
4. **Overall "busyness"** — shuffling gait produces smaller, fainter foot bursts.

⚠️ **Big caveat:** you **cannot** trust one image. Brightness and reach also
change with how close the person walked, their body size, radar noise, and the
contrast-enhancement step (§5). By eye you might guess; to actually *claim* it you
need the statistics (§6). Naked-eye reading builds intuition — it is not the
diagnosis.

---

## 4. The two footnotes explained

### (a) "`ce_foot` is feet-aimed, but you still see torso energy near 0"
There are radars aimed low (at foot height) and radars aimed high (at the torso).
`ce_foot` comes from the **foot-aimed** radar, so **fast foot spikes dominate** it.
But a radar beam is wide — it can't perfectly see *only* the feet, so the slow
torso still shows up as a band near 0 Hz. `ce_torso` (torso-aimed) is the mirror
image: the torso band dominates and the feet are fainter. Neither perfectly
isolates one body part; they just **emphasize** different ones. (That's also why
the reference paper uses *both* foot and torso radars.)

### (b) "`ce` = contrast-enhanced, raw `|stft_foot|` would look dimmer"
The fast foot motion reflects **much less energy** than the big slow torso, so in
the **raw** spectrogram (`|stft_foot|`) the foot spikes are faint and hard to see.
The **contrast-enhancement** step brightens the faint parts so the foot spikes
become clearly visible. Upside: you can see the gait. Downside: it also brightens
noise, and it does so **differently for each recording** (see §5) — which is a
trap we have to control for.

---

## 5. Why did the two example images look so different? (control "hot", PD "dark")

In the control-vs-PD figure, the control looked bright/"heated" everywhere and the
PD one looked darker with crisp arcs. That difference is **mostly not biology** —
it's three artifacts stacked up:

1. **Adaptive contrast enhancement.** `ce` rescales **each recording to its own
   brightness range**. The control recording had more background noise, so the
   enhancement lit the *whole* image up. The PD recording was cleaner, so it
   stayed dark with sharp foot arcs. → Absolute brightness is **not comparable**
   between two `ce` images.
2. **SNR / setup differences.** How close the person walked, body size, and radar
   signal quality all change overall brightness — nothing to do with PD.
3. **n = 1 per group.** One control and one patient. Individuals vary enormously;
   one pair proves nothing.

So "control looks hotter" ≠ "healthy people reflect more." It mostly means those
two particular recordings were normalized differently.

### The "scalloped teeth" (PD panel)
Each foot swing **accelerates then decelerates** (speeds up mid-swing, slows as it
lands), tracing an **arc that rises and falls** — a "tooth" or scallop shape. A row
of evenly spaced teeth = regular stepping. In that figure the PD panel happened to
show clean, well-separated teeth; the control panel was too washed-out by noise to
see them. (Again: a quirk of those two recordings, not a rule.)

---

## 6. "That number is partly an artifact" — and the levels of evidence

I had put "foot reach ≈ 775 Hz (control) vs 275 Hz (PD)" on the figure. **Don't
trust that number**, for three reasons — this is the kind of thing a thesis must
catch:

1. **Noise inflates it.** The "reach" metric finds the highest Doppler with strong
   energy. The control image had noise almost everywhere, so it reported a reach
   near the ±800 Hz *maximum the radar can measure* — regardless of real foot
   speed.
2. **Adaptive CE makes brightness non-comparable** across recordings (§5).
3. **n = 1.** One person each — can't generalize.

Clinically, you'd *expect* healthy feet to be faster than PD feet (PD =
**bradykinesia**, slowed movement). So the *direction* of the number isn't crazy —
but **this particular measurement is too unreliable to claim it.**

### Hierarchy of evidence (how much to trust each thing)
| Level | What it is | How much to trust |
|---|---|---|
| **One example pair** | look at 1 control + 1 PD image | **illustrative only** — builds intuition, proves nothing |
| **Group averages** | average many controls vs many PD, look at the difference | **moderate** — averages out per-recording noise |
| **Confound-controlled statistics** | subject-level LOSO AUC with duration matching + permutation test | **this is what you cite** — real, quantified, defensible |

Our actual quantified result lives at the bottom level: confound-controlled
subject-AUC ≈ **0.62–0.68** (significant, p = 0.022 after removing the duration
confound). That is the honest "there is a weak but real signal" statement — see
`RESULTS.md`. Everything visual (§2–§5) is for *understanding*, not for *claims*.

---

## 7. Earlier concepts (quick recap, with pointers)

- **Prodromal** = the *pre-symptomatic* early phase of PD (before diagnosis). Our
  dataset has **none yet** (Phase 2). We ignore it for now; it's future work.
- **"Strict data cleaning needed"** (dataset readme) = the v1 files may contain bad
  recordings; we run quality-control checks (`src/qc.py`) to catch corrupt/empty
  trials before training.
- **The reference paper vs this thesis**: the paper (López-Delgado 2026) *measures
  gait parameters* and validates the radar; it never diagnoses. **This thesis adds
  the PD-vs-healthy classifier.** See `ONBOARDING.md` §3.
- **Two tracks**: same spectrogram → (a) *classical* = summarize into features →
  simple model (done), (b) *deep* = feed windows to a CNN that learns for itself
  (pending, needs GPU). Both judged by the same LOSO test. See `ONBOARDING.md` §6.

---

*Read order for a newcomer: this file → `ONBOARDING.md` → `RESULTS.md`.*

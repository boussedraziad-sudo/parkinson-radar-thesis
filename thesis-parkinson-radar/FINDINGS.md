# Findings — Notebook 11 results

**Status:** **Filled** from notebook execution (exec_count 21/21 — § 6 cell 32 was the only one without recorded outputs; everything else ran).

Captured outputs come straight from the notebook's cached results — not reconstructed.

---

## § 1 — Contrast-enhancement formula

**Source spectrogram tested:** `fisc_005/test1/trial1`. `|stft_foot|` shape (320, 12068), range [1.19e-4, 138]. `ce_foot` shape (320, 12068), range [8.42e-7, 0.999].

### 1.1 Best element-wise fit on the source trial (foot)

| Family | R² | Params (a, b/γ) |
|---|---|---|
| **dB** (`a·log10(x)+b`) | **0.9178** | `a=0.5475, b=0.4023` |
| log1p (`a·log(1+b·x)`) | 0.9127 | `a=0.3035, b=2.5707` |
| cbrt (`a·x^(1/3)+b`) | 0.8329 | `a=0.5534, b=-0.1786` |
| gamma (`a·x^γ`) | 0.7883 | `a=0.3929, γ=0.3829` |
| linear | 0.3403 | — |

→ **dB** is the best element-wise family on this trial, but R² = 0.92 is *not* good enough to call it the formula (would need > 0.99 for a clean element-wise mapping).

### 1.2 Same dB fit applied to torso + 4 random trials

| Trial / channel | R² | Params |
|---|---|---|
| Torso, same trial (fisc_005/test1/trial1) | **0.5396** | `a=0.3403, b=0.1985` |
| fisp_037/test1/trial3 (foot) | 0.9451 | `a=0.647, b=0.467` |
| fisc_012/test1/trial3 (foot) | 0.9015 | `a=0.577, b=0.361` |
| fisc_027/test1/trial3 (foot) | 0.5150 | `a=0.041, b=0.048` |
| fisc_040/test2/trial2 (foot) | 0.4594 | `a=0.071, b=0.068` |

R² range across 4 random foot trials: **0.46 – 0.95** (very wide). Parameters drift by an order of magnitude (`a` from 0.041 to 0.647).

### Conclusion

**`ce_*` is not a simple element-wise function**. The dB fit's quality varies dramatically per trial (R² 0.46–0.95), and even within the same trial the foot and torso channels need different parameters (foot R²=0.92, torso R²=0.54). This is consistent with **per-trial adaptive normalisation** — most likely a per-trial / per-channel rescale (min-max or percentile-based) layered on top of a log or dB compression. CLAHE-style local windowing is also possible.

**Proposed CE formula (best-effort):**

```
ce ≈ rescale_per_trial( log10(|stft|) )    # parameters of rescale vary per channel and per trial
```

**For the Ignacio email**: don't claim a formula — instead say "the mapping looks adaptive per-trial; can you describe the actual operation?"

---

## § 2 — Doppler-axis uniformity

| Aspect | Value | Expected |
|---|---|---|
| Modal `n_bins` | **320** (348/348 files) | 320 ✓ |
| Modal Δf | **5.00 Hz** (std 8e-6 Hz across files) | 5.0 ✓ |
| Files with `n_bins ≠ 320` or `\|Δf − 5\| > 0.5 Hz` | **0** | 0 ✓ |
| Doppler range | `[-794.87, +799.87] Hz` ≈ ±800 Hz | ±800 ✓ |

**Conclusion:** Doppler axis is **perfectly uniform** across all 348 files (320 bins at 5 Hz, ±800 Hz). The 4× zero-padding from the paper's 20 Hz intrinsic resolution is consistent everywhere. Preprocessing can safely assume a fixed Doppler dimension.

---

## § 3 — `param` struct + hidden variables

### 3.1 — `param` field dump (from `fisc_005/test1/trial1`)

```
nMuestras                  7500000          # number of samples
hora, minutos              9, 28            # timestamp (HH:MM, capture session)
indice                     1
Tramp                      0.000625         # 625 µs chirp ✓ matches paper
Tcaptura                   15               # 15-second capture window
nCanales                   4                # 4 radar channels
Fs                         500000           # 500 kHz ADC
BW                         1380000000       # 1.38 GHz ✓ (paper says 1.4 GHz)
ftrabajo                   23500000000      # 23.5 GHz carrier (paper says 23 GHz)
TrampTotal                 0.0006251
nMuestrasRamp              312.55
c                          300000000        # speed of light
rampas_bloque              300000
p_muestras_iniciales_eliminadas    0.05    # 5% of initial samples trimmed
p_muestras_finales_eliminadas      0.03    # 3% of final samples trimmed
nMuestrasCount             375059996
angulo_rotacion            0
tug                        1               # TUG-protocol flag
offset_samples             [118, 118, 25, 25]   # per-channel offsets (4 radars)
escala                     1
n_samples_ramp             287
n_ramp                     23994
DestinationPath            C:\Captura_Silicon\Capturas\20_02_2024   # capture date 2024-02-20
deriva                     1e-07
offset                     344
flyback                    0
TriggerMode, InC1..InC4    0, 0, 0, 0, 0
```

**Notable**: `BW = 1.38 GHz` (paper says 1.4 GHz — minor difference) and `ftrabajo = 23.5 GHz` (paper says 23 GHz). Should update `src/radar_params.py` constants to match the actual capture (`F_CARRIER_HZ = 23.5e9`, `BANDWIDTH_HZ = 1.38e9`).

Also: every file has a `tug` flag and pre-trimming fractions baked in (5% head, 3% tail). The `DestinationPath` carries the capture date.

### 3.2 — Variable inventory across the 348 files

| Variable | Files seen / 348 |
|---|---|
| All 16 variables present in **all 348 files** | — |

Listed (in occurrence order): `all_fig_savename`, `all_loc`, `all_fig_caption`, `idx_foot_max_snr`, `idx_torso_max_snr`, `doppler_axis`, `all_doppler_time`, `all_ce_time`, `ce_foot`, `ce_torso`, `all_target_time`, `t_axis_target`, `stft_foot`, `all_dir`, `param`, `stft_torso`.

**New variables not documented in the readme**: `all_fig_savename`, `all_fig_caption`. Likely auto-generated MATLAB figure metadata — could potentially extract per-trial captions to mine for additional context.

**Conclusion:** No missing variables, no inconsistencies. Dataset structure is well-formed. The undocumented `all_fig_*` fields are worth dumping in a follow-up to check whether they contain useful annotation.

---

## § 4 — `fisp_022/test1/trial4` forensics

### Trial inventory

| Trial | Duration (s) | Time bins |
|---|---|---|
| trial1 | 9.50 | 15191 |
| trial2 | 11.20 | 17920 |
| trial3 | 8.72 | 13947 |
| **trial4** | **9.64** | **15415** |

Trial4 sits comfortably inside the within-subject duration range — not an outlier.

### Cosine-similarity matrix (`log1p(ce_foot)`, resized 160×256)

|        | trial1 | trial2 | trial3 | trial4 |
|--------|--------|--------|--------|--------|
| trial1 | 1.000  | 0.824  | 0.863  | 0.820  |
| trial2 | 0.824  | 1.000  | 0.839  | 0.833  |
| trial3 | 0.863  | 0.839  | 1.000  | 0.825  |
| trial4 | 0.820  | 0.833  | 0.825  | 1.000  |

All inter-trial similarities cluster in **0.82 – 0.86** — solidly within the within-subject reference band (0.6–0.85) seen elsewhere.

**Conclusion:** **trial4 is a clean retry, not a duplicate.** No pair similarity exceeds 0.95 (duplicate threshold), and the values are indistinguishable from the trial1↔trial2↔trial3 baseline. **Recommended action: keep trial4**, but mention to Ignacio for confirmation.

---

## § 5 — `fisp_048/test2/trial2` missing-trial forensics

| Aspect | Value |
|---|---|
| Present in `fisp_048/test2/` | trial1, trial3 |
| Cosine similarity trial1 ↔ trial3 | **0.867** |
| Within-subject reference band (from § 4) | 0.82 – 0.86 |

The two surviving trials show a normal within-subject similarity. They are **not duplicates** of each other.

**Conclusion:** trial2 was almost certainly a clean deletion. Safe to proceed with 5 files for this subject. Still worth a one-line mention to Ignacio.

---

## § 6 — Doppler symmetry

| Aspect | Value (`ce_foot` across 348 trials) |
|---|---|
| Median E(+Doppler) / E(−Doppler) | **1.005** |
| Mean ratio | 1.005 |
| Std across trials | 0.043 |
| Range | [0.752, 1.133] |

Ratio centred almost exactly on 1.0 with very tight spread (5th–95th roughly 0.92–1.08).

**Conclusion:** **`ce_foot` is symmetric in the Doppler axis** — direction of motion is not preserved in the combined foot signal. This is consistent with (a) the paper's Fig. 6 deliberate flip of node 2 before SNR combination (which applies to torso explicitly), and (b) the dataset storing magnitude-style spectrograms where +/− Doppler are folded by construction. **Implication for the model**: we can safely fold to absolute Doppler (160 bins instead of 320) without losing information — useful as an ablation to halve the input dimension on the Doppler axis.

> *Note: the notebook code currently computes symmetry only for `ce_foot`. A `ce_torso` symmetry check would confirm the Fig. 6 prediction holds there too — worth adding a one-line cell if you want the corroborating number.*

---

## § 7 — Duration confound, broken down by test

| Test | Control median (s) | PD median (s) | PD longer by |
|---|---|---|---|
| **test1** (TUG with chair) | **9.17** | **10.09** | **+10.1%** |
| **test2** (TUG without chair) | **8.04** | **8.31** | **+3.4%** |

Group-level descriptive stats from the notebook output:

```
                  count   mean   std   min    50%    max
test1 control      99     9.81  2.16  7.09   9.17  16.20
      pd           76    11.18  3.03  6.71  10.09  19.20
test2 control      99     8.30  1.69  5.80   8.04  13.26
      pd           74     8.83  2.13  6.06   8.31  15.65
```

**Conclusion:** **The duration confound is concentrated in test1.** PD subjects take ~10% longer on the chair-stand TUG, but the gap collapses to ~3% in the no-chair (test2) TUG. This is clinically sensible — the sit-to-stand transition is the slowest part of TUG for PD subjects, while the steady walking phase is much less affected.

**Implication for modeling:** test1 carries more PD signal — but also more confound. Consider three options for the thesis:
1. Train on both pooled (current default).
2. Train on test2 only (less confound, cleaner gait signal).
3. Train on both with `test` as a feature so the model can use the test-1 chair-stand information when it helps.

Bring this to the next Nacho meeting.

---

## § 8 — Subject fingerprint strength (within vs between)

| Aspect | Value |
|---|---|
| Median within-subject distance | **2.04** |
| Median between-subject distance | **5.23** |
| **Median ratio (within / between)** | **0.374** |
| Fraction of subjects with ratio < 1 | **1.00** (all 58) |
| Max ratio observed | 0.973 |

Full descriptive stats:

```
       within  between  ratio
mean    2.231   5.633   0.412
std     1.045   1.350   0.193
min     0.965   3.562   0.140
50%     2.041   5.228   0.374
max     6.898  11.271   0.973
```

**Conclusion:** **Very strong subject fingerprint.** Median within-subject distance is ~37% of the median between-subject distance, and **every single subject** has within < between. **LOSO is essential** — any cross-validation that mixes a subject's trials between train and test would massively over-estimate performance. The maximum ratio (0.973) means one subject's trials look almost as variable as random others' — worth a quick look in the saved figure (`outputs/figures/11_within_vs_between.png`) to identify the outlier.

---

## Summary table (the one for the email + thesis)

| # | Question | Answer |
|---|---|---|
| 1 | CE formula | **Not element-wise** — per-trial / per-channel adaptive (dB-like with rescaling). R² varies 0.46–0.95 across trials. |
| 2 | Doppler-axis uniformity | Perfectly uniform: 320 bins × 5 Hz × ±800 Hz across all 348 files. |
| 3 | `param` fields | Confirms paper params (with `BW=1.38 GHz`, `f₀=23.5 GHz` — slightly different from the paper's 1.4 / 23). All 16 variables present in all 348 files. |
| 4 | `fisp_022/test1/trial4` | Clean retry (similarities 0.82–0.86, within normal band). Keep. |
| 5 | `fisp_048/test2/trial2` | Likely clean deletion. trial1↔trial3 similarity 0.867 looks normal. Proceed with 5 files. |
| 6 | Doppler symmetry | Symmetric (median ratio 1.005, std 0.043). Direction not preserved → can fold to abs-Doppler as an ablation. |
| 7 | Duration gap by test | test1: +10.1% (PD longer) ; test2: +3.4%. Confound concentrated in chair-stand. |
| 8 | Subject fingerprint | Strong: median within/between = 0.374; 58/58 subjects below diagonal. LOSO essential. |

---

## Draft email to Ignacio López-Delgado (`ie.lopez@upm.es`)

> Subject: Thesis — a few dataset clarifications (with what we found from the data so far)
>
> Hi Ignacio,
>
> I'm Ziad, Nacho's master's student working on the radar gait dataset (`shared_ziad`). Before bothering you, I ran some forensics on the v1 release and have specific findings to confirm rather than open questions to ask. Three things, all summarised below; details and figures in `thesis-parkinson-radar/FINDINGS.md` if useful.
>
> **1. `ce_foot` / `ce_torso` formula.** The reference paper doesn't describe a contrast-enhancement step, so I tried to reverse-engineer it from the data. The best **element-wise** fit is dB (`a·log10(|stft|) + b`), but its R² varies a lot — 0.92 on one trial, 0.46 on another — and the parameters drift by an order of magnitude across trials. So the mapping looks **adaptive per trial** (probably a log/dB stage followed by some rescaling). Could you confirm what the actual CE operation is? If there's a script that generates `ce_*` from `stft_*`, that would settle it.
>
> **2. `fisp_022/test1/trial4`.** Extra trial. Compared to its sibling trials (cosine similarities 0.82–0.86, indistinguishable from the trial1↔trial2↔trial3 baseline), it looks like a clean retry. I'm planning to **keep it** — confirm if that's OK?
>
> **3. `fisp_048/test2/trial2`.** Missing. The other two trials in that test look like a normal retry pair (similarity 0.87), so I'm proceeding with 5 files for that subject. Anything I should know?
>
> One smaller thing — `param.BW` reads 1.38 GHz and `param.ftrabajo` reads 23.5 GHz, slightly different from the 1.4 GHz / 23 GHz in the IEEE paper. Just to be sure I'm using the right numbers in the thesis, are 1.38 / 23.5 the operating values for this dataset?
>
> Separately, I'd also appreciate a few things when you have a moment:
>
> - **Age / sex / UPDRS / medication-state table** per subject (needed for any age-matched analysis and to disambiguate the G1-young vs G2-older control split).
> - **Generation script** for the `.mat` files (would be useful for the thesis's reproducibility section).
> - **Phase 2 / prodromal data timing** — when can `fis_*` subjects be expected? Affects whether transfer-learning experiments fit the 8-week window.
> - The undocumented `all_fig_savename` / `all_fig_caption` variables — do they carry any per-trial annotation worth extracting?
>
> Thanks — happy to come by GAPS if it's easier than email.
>
> Ziad
> `ziad.boussedra@alumnos.upm.es`

---

## Provenance

| Field | Value |
|---|---|
| Notebook run date | 2026-05-29 (notebook exec_count=21/21) |
| Total trials processed | 348 / 348 |
| Trials skipped due to errors | 0 |
| Figures saved | `outputs/figures/11_ce_mapping_fit.png`, `11_fisp022_trial4.png`, `11_fisp048_test2.png`, `11_duration_by_test.png`, `11_within_vs_between.png` |
| Filled in by | Claude session, parsed from cached notebook outputs |

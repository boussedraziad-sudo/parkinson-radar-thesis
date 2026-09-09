# Progress tracker

*Live status. Last updated 2026-09-09. Companion to `WORK_PLAN.md`.*

**1 day to submission (10 Sep) · defence window 23–30 Sep**

---

## ✅ Supervisor's annotated draft resolved (9 Sep) — 24 comments, 80 pp

- Comments pulled programmatically from the PDF annotation layer
  (`ZIAD_THESIS_FIRST_DRAFT_reviews.pdf`); every one applied, see
  `REVIEW_FINDINGS.html` §6.
- Terminology: **micro-Doppler signature** defined in §1.1 and used in the
  abstracts, Ch1, Ch2, Ch5 and flagged Ch3 spots; "spectrogram" kept only for
  the STFT object itself.
- New **Table 5.7** (future work, methods suggested at review) with 13 new
  references: GAIN attention-guided masking, RRR / explanation regularisation,
  counterfactual training, adaptive background thresholding, DCAE bottleneck +
  t-SNE/UMAP, speech-pretrained encoders, AR Doppler estimation, window length
  tuned by AUC.
- Coupling figure redrawn with both traces on the signature at true Doppler.
- §2.6 renamed "Limitations of the state of the art in the field"; accuracies
  of reviewed works added; intra/inter-patient variability sentence; Gregorio
  Marañón collaboration + Vicon reference; p-value equation; all bullets
  capitalised; GMR in acknowledgements.
- Compile: **80 pages, 0 overfull, 0 undefined, bibtex clean (37 refs)**.

## ✅ Read-through round 4 applied (5 Sep)

Full report DONE and polished through four read-through rounds: **78 pages, 0
overfull, 0 undefined refs**. This round:

- **Figure 3.2 regenerated** — signal-chain box texts rebroken and boxes
  widened; nothing overflows (`tools/make_setup_figure.py`).
- §3.3 opener no longer forward-references §3.4/§3.7; fold-definition
  paragraph moved between Figure 3.5 and Table 3.3.
- §3.4 deletions: "mean over all of its pixels" sentence and the whole
  mean-vs-sum paragraph (the shape_clean row of Table 3.7 keeps the fact).
- §3.5 Statistical Analysis **kept** (p, δ, corrected p appear in nearly every
  Ch4 exhibit) with a one-line justification added at its top.
- Classical baseline: "repeated once per fold" → "once for each subject in
  turn"; "(RBF)" dropped from the SVM row.
- §3.7.1 restructured figure-first; Figure 3.12 resized 0.9\textwidth so it
  sits on the same page as its heading.
- **Hayashi decision closed**: 97.8 % AlexNet verified as hayashi2021's own
  comparison (their §4.3, own Grad-CAM attributes it to background noise);
  naming fixed to "Hayashi et al." (3 places).
- **Paired bootstrap declined** (user call) — softened wording is final.
- **Two-sided header question closed** (user call) — ignored, single-sided.
- Fresh 14-agent review swarm launched over the recompiled PDF; findings in
  `REVIEW_FINDINGS.html`.

Remaining decisions: "session-balanced" phrase in Future Work; annex language
convention (ask school). Remaining work: tracked content minors + bibliography
batch in `REVIEW_FINDINGS.html`.

---

## ✅ Week 0 — COMPLETE

| # | Item | Status |
|---|---|---|
| 1 | Email data owner (López-Delgado) — age/sex, `ce` script, anomalies | ✅ sent, awaiting reply |
| 2 | Email tutor — template, deadlines, ethics, scope | ✅ sent, awaiting reply |
| 3 | Deadlines from coordinator | ✅ **RECEIVED** — 10 Sep submit, 23–30 Sep defence |
| 4 | Obtain TFM template | ✅ official `plantilla_tft_etsit` + example thesis |
| 5 | Build report scaffold | ✅ `Ziad_Boussedra_TFM/`, English folder names |
| 6 | Confirm body language | ✅ English body + mandatory Spanish *Resumen* |
| 7 | **Install LaTeX** | ✅ **TeX Live 2026 working** (see PATH note below) |
| 8 | Degree + track on covers | ✅ MSTC / Signal Processing and ML for Big Data |
| 9 | **Port Related Work into the report** | ✅ **Chapter 2 complete — 2,140 words, 16 citations** |
| 10 | Decide: proceed without age/sex | ✅ locked → becomes a Limitation |
| 11 | Decide: data as-is, no data surgery | ✅ locked |

### Bonus work completed beyond the Week-0 plan
- **Bibliography repaired** — 3 classes of bug that would have printed wrongly:
  author separators (12 entries), research notes leaking into `journal` fields
  (5), affiliations inside `author` fields (3). Then normalised: volume/pages
  into real fields, 2 wrong entry types corrected.
- **References hyperlinked** — `unsrt` was silently dropping every URL. Switched
  to `unsrturl`; **17 live links** verified in the PDF.
- **Layout bugs fixed** — stretched paragraph gaps (`\raggedbottom`), 3 overfull
  lines, one over-long heading. Now **0 overfull boxes**.
- **Writing conventions locked** (apply to every future chapter): no em-dashes,
  **bold key terms** for skimmability, bullet/numbered lists instead of long
  prose blocks.

### Build status
```
26 pages · A4 · 0 errors · 0 overfull boxes · bibtex clean
16/16 citations resolved · 0 undefined references · 17 clickable links
```

**PATH note:** MacTeX did not create the usual `/Library/TeX` symlink, so add this
once:
```bash
echo 'export PATH="/usr/local/texlive/2026/bin/universal-darwin:$PATH"' >> ~/.zshrc && source ~/.zshrc
```

---

## Document completion

| Chapter / part | State |
|---|---|
| Ch 1 — Introduction & Objectives | ✅ complete, read-through applied |
| Ch 2 — State of the Art | ✅ complete, read-through applied |
| Ch 3 — Materials & Methods | ✅ complete, read-through applied (round 4) |
| Ch 4 — Results | ✅ complete, user read-through pending |
| Ch 5 — Discussion / Conclusions / Future Work | ✅ complete, user read-through pending |
| Abstract (EN) + Resumen (ES) | ✅ rewritten per feedback |
| Acknowledgements | ✅ scaffold in place (personalise before deposit) |
| Annex A (ethics/ODS) · Annex B (budget) | ✅ complete (360 h · M1 Max · 8.999,61 €) |
| Acronyms | ✅ done |
| Covers + committee page | ✅ done |

**All chapters written.** 80 pages, clean compile. What remains is the user's
read-through of Ch4–Ch5/annexes, the tracked minors, and the deposit logistics.

---

## ⏳ Open / awaiting others

| Item | Owner | Impact if never answered |
|---|---|---|
| Age / sex metadata | López-Delgado | Age confound stays uncontrolled → prominent Limitation |
| `ce` enhancement formula | López-Delgado | Provenance gap stated in Methods |
| **What distinguishes the two acquisition batches** | López-Delgado | Cannot say whether it is hardware or two recruitment campaigns |
| Ethics coverage confirmation | Tutor | Needed for Annex A |
| **Tutor's *visto bueno*** | Tutor | 🔴 **Required to deposit.** Escalate to coordinator by 11 Aug if silent. |

---

## 🔄 SUPERSEDED 21 Aug: batch analysis removed from the report (user decision)

The within-batch analysis was never confirmed with the supervisor, so the user
decided (21 Aug) to **remove it from the thesis and the results page** and
report pooled-only. The thesis verdict is reframed as: *no model clearly
exceeds the duration floor (0.610); every 95 % CI contains it; Grad-CAM and the
background-suppression experiment attribute the remaining score to
per-recording properties.* This verdict needs no batch claim and is fully
defensible. The batch/within-batch analyses stay **archived** (RESULTS.md,
outputs/runs/campaign_summary.json, this tracker) and one open question row in
the results page points at "the two recording sessions" for the supervisor
conversation. If the supervisor confirms the sessions matter, everything can be
restored from the archive in an afternoon. The block below is kept for the
historical record only.

## ✅ Decided: how the acquisition batch is handled *(superseded, see above)*

*Not an open question. `WORK_PLAN.md` already settles this (Day 3, Day 11, §10A,
§10C). Recorded here so the two documents agree.*

Two decisions, both already taken:

1. **Data stays as-is.** No data surgery, no file edits. The 0.0014 Hz Doppler
   difference is physically meaningless and is treated purely as a *label*.
2. **Batch is reported as a control.** It is a scheduled Day 3 task using the
   labels already cached in `outputs/metrics/acquisition_batch.csv`.

| Group | Subjects | control | PD | % PD |
|---|---|---|---|---|
| A | 34 | 24 | 10 | 29.4 % |
| B | 24 | 9 | 15 | 62.5 % |

Batch membership alone predicts diagnosis at **AUC 0.664**, higher than any model
built so far, and the classical baseline falls to chance inside a single batch
(0.621 → 0.537 / 0.541).

**What the plan commits to:**
- `batch_only` (0.664) is reported as a **second null floor**, beside
  `duration_only` (0.610).
- **Within-batch AUC is the reported headline**, with the pooled figure shown
  alongside (`WORK_PLAN.md` §10A). This follows from §10C, where every success
  tier is defined by within-batch AUC.
- Within-batch is the first item in the Week-3 confound battery (Day 11),
  because duration-matching, residualisation and test2-only all leave batch
  intact.
- The §10C success criteria are all expressed in terms of within-batch AUC.

---

## ✅ Decided: subject-independent LOSO — provenance, for the defence

*Verified 10 Aug 2026 against the proposal PDF (git blob, initial commit), the
source publication, and the project's own dated notes.*

- **Not in the proposal.** The one-page proposal abstract commits only to
  "robust classification models evaluated on clinical datasets" — no split
  strategy, no mention of subjects as a grouping unit.
- **Not in the source publication.** López-Delgado et al. contains no
  classifier; its "validation" means sensor-vs-Vicon accuracy.
- **It was forced by our own EDA (2026-05-29).** The subject-fingerprint
  measurement (within/between distance ratio 0.374, 58/58 subjects;
  `FINDINGS.md` §8) showed a walk identifies the walker, so any split that
  mixes a subject's recordings between train and test measures person
  re-identification, not disease. LOSO itself is textbook leave-one-out
  (Stone 1974) with the subject as the unit; what is ours is *enforcing* it.
- **Field context.** Of the 16 reviewed works, only one reports genuinely
  subject-wise splits (n=3, and its F1 dropped 89.8→83.2 under it); the rest
  use random hold-outs or don't specify. That gap is the stated contribution.

If asked "who decided this and when": *decided by the student during
exploratory analysis, before any model was trained, on measured evidence —
then confirmed with the supervisor via the July question sheet (Q8).*

---

## ✅ Decided: report structure stays IMRaD

*Settled 6 Aug 2026. Do not reopen.*

The report follows the **chapter skeleton of the ETSIT-UPM template**, unchanged:

| Ch. | Contents |
|---|---|
| 1 | Introduction and Objectives |
| 2 | State of the Art |
| 3 | **Materials and Methods** — everything methodological, EDA *and* ML |
| 4 | **Results** — §4.1 EDA findings, §4.2 onward modelling |
| 5 | Discussion, Conclusions and Future Work |

**The alternative considered and rejected:** splitting into *EDA + its results*,
then *ML + its results*. Rejected for three reasons:

1. **Shared methods.** LOSO, the five feature sets, the confound controls, the
   subject-level bootstrap and the permutation test all serve both halves. A
   split duplicates them or parks them arbitrarily.
2. **Comparability.** §4.2 onward must place a CNN, a logistic regression and the
   `duration_only` floor side by side; separate chapters break that.
3. **The defence.** The 20-minute Objectives → Activities → Results →
   Conclusions format maps one-to-one onto IMRaD.

**Known cost, accepted:** Methods forward-references Results (e.g. §3.1.2 points
to §4.1.2) because several design choices were forced by EDA findings the reader
has not met yet. Mitigated by §4.1 opening with a statement that its findings
constrained everything downstream, and by explicit subsection cross-references.

**Citation style:** plain numeric keys only. No page, section or figure locators
inside `\cite`.

---

## ✅ Deep-model campaign complete (9 Aug) — negative result, thoroughly confirmed

11 deep configurations, 4 model families, all leak-free nested LOSO. Full table:
`outputs/runs/campaign_summary.json`. **No configuration shows within-batch gait
signal.** The only apparent hit (abl_foot batch B 0.704) flipped batches under
the corrected v2 protocol → selection noise. v2 fixes (pre-reg `68c6284`)
removed the duration residue (corr +0.15 → +0.05) without changing the verdict.

## ✅ Random-split detour complete (13 Aug) — the inflation, measured

`classical_split/`: same ResNet-18 probe, same v2 cache, only the split changed.
11 trainings. Window-level random split **0.77** AUC, recording-level (the
literature's hold-out) **0.76**, vs LOSO 0.65 pooled / ~0.50 within-batch.
**Collapse test**: models scoring 0.78–0.90 on familiar subjects fall to
0.47–0.53 on 8 never-seen subjects (one lucky octet at 0.74 — small-n noise).
100 % of random-split test windows come from subjects seen in training.
Gallery shows predictions track background texture, not gait. Full page:
`classical_split/CLASSICAL_SPLIT.html`; results `classical_split/outputs/`.
→ **Placement decided 13 Aug: FIRST section of the Discussion chapter** — the
small proof of why the normal way of doing things was not followed. Cross-
referenced from the modelling document's validation section (§4).

## ✅ Grad-CAM + background-suppressed experiment (16 Aug)

Grad-CAM (58 folds, both v2 models): attention sits on background >500 Hz
(density 0.192 vs 0.118 torso) and turn windows (4-6x steady), NOT gait bands;
frozen probe shows zero PD/control attention asymmetry. → Results §4.6.
Follow-up pre-registered experiment `experiments/bg_suppressed/` (background
subtraction + crop to ±500 Hz, else v2-identical): pooled HELD (smallcnn 0.624,
resnet 0.686) → batch fingerprint leaks through more than texture; within-batch
bar NOT met (smallcnn 0.52/0.31, resnet 0.50/0.68 — the 0.68 is a one-model
one-batch bump, same pattern as foot-only-B that dissolved; frozen rules demand
a confirmation run before believing it). **Negative finding stands.**

## ✅ Decided 21 Aug: report placement of the interpretability work

Grad-CAM ("What the models attended to") and the background-suppression
experiment go in the **Discussion chapter** of the LaTeX report, alongside the
random-split demonstration (its first section), not in Results. Results §4.2+
carries: yardstick grid, the three models + two exploratory variants,
decisions/confusion section with the case gallery. The "Every question
answered" recap exists only in the HTML working documents, not the report.

## ✅ AlexNet experiment complete (22 Aug) — dataset vs methodology, settled

`experiments/alexnet/`: the field's architecture (Hayashi's AlexNet), same v2
cache, both protocols. Random recording split (their way): window AUC
0.70-0.86, **subject AUC up to 0.891**, 100 % of test subjects seen in
training. LOSO (our way): **0.694** [0.554, 0.833] — highest compliant pooled
number, but CI contains the duration floor (0.610), duration residue **+0.25**
(strongest of any compliant run; campaign models ~+0.05), single post-hoc
config. Verdict: the dataset yields literature-grade numbers under the leaky
protocol and honest-band numbers under LOSO → the gap is the methodology's
question, not the data or our models. Page: `experiments/alexnet/ALEXNET.html`.
DISCUSSION.html drafted same day (6 sections), awaiting user review.

**Placement decided (22 Aug):** AlexNet goes into the report as a subsection of
Discussion section 2 (the random-split demonstration), NOT into Results —
Results stays the pre-registered campaign only, and it is already proofread.
Added to DISCUSSION.html section 2 ("The same demonstration with the field's
own network": two-protocol table, alex_ladder figure, duration-residue
+0.25 vs +0.08, paired Δ +0.040 CI [−0.047, +0.134] recomputed at build time
from stored fold CSVs, seed 1337). LaTeX untouched; enters ch5 with the rest
of the Discussion after the user reads the HTML.

**✅ Chapter 5 WRITTEN INTO LATEX (22 Aug):** discussion.tex (Discussion §5.1
with protocol ladder, AlexNet, Grad-CAM both galleries, intervention test,
positive findings + Limitations §5.2), conclusions.tex (§5.3, five bold
conclusions), future_work.tex (§5.4, four steps). Five figures copied as
disc_*.png. Compiles 74 pp, 0 overfull, 0 undefined, all cites resolve.
Verified by a 3-agent pass (numbers: zero findings; naming unified to
"ResNet-18 probe" everywhere; tab:alexnet now referenced; bg-suppressed CIs
added to the table; ch3's Grad-CAM promise narrowed to "both image
networks"). Open items: conclusions.tex carries a note to map paragraphs to
O1..O5 once ch1 objectives exist; future work intentionally has NO
per-sensor-recordings step (user removed it 22 Aug) even though the
asymmetry limitation row mentions per-sensor recordings as the resolver.

**✅ USER READ-THROUGH ROUND 2 APPLIED (ch2 + ch3 + layout batch, 3 Sep):**
simpler words (nascent/hallmark/sparse/keying/pervasive/insidious/rigour/
coarse/corpus/consequential); LSTM named in §2.3; shortcut-learning +
"trustworthy vs diagnostic" sentences rewritten plainly; Table 2.1 → ≤3
bullets per cell; gap claim deduplicated (kept in ch1); §3.1.3 cohort
deviations made explicit (7-recording and 5-recording PD subjects); QC and
§3.2.3 shortened, "cheapest first" cut; Table 3.3 (bins) + paragraph deleted;
§3.2.4 direct take; heel strike glossed; NEW coupling figure (guide_coupling,
fisc_048 test2: foot-band valleys + trunk-speed peaks aligned — note: torso
ENERGY does not peak at heel strike, trunk SPEED does; §3.2.6 phrased
accordingly); §3.3 restructured (sentence → pipe_overview → stage table);
§3.3.2 Amplitude compression deleted (log1p lives in the stage table);
chapter refs hyperlinked (ch:intro/soa/methods/results/discussion); Closing
Remarks section after Future Work. ALSO applied the review's content majors:
ce channels = per-frame max-SNR combination of the two same-height nodes;
4-node rig (table + setup-figure annotation regenerated); 23 GHz = sweep
start of 24 GHz band, λ≈1.3cm ±3%, radial-velocity axis; Hann ≈30 Hz; BH
justification corrected; inner-loop seed rule + epoch metric stated;
post-hoc framing of the random-split test; classical settings named
(sklearn defaults, archived); grid-reading claim restricted to multi-feature
sets; SVM 0.287 footnote; "shown to exceed" wording; 0.2-AUC derivation
stated. LAYOUT: global no-hyphenation; all ch3-ch5 floats [H]; tab:keyworks
[p]; committee page fixed; fpsep balanced; figs 4.1/4.2 after first
paragraph; Table 4.4 one-line headers; annexA \section*; annexB caption
above. FIGURES regenerated with ≥7-8pt effective fonts by agents: setup,
all pipe_*, eda_class_balance/duration/violins (via src/eda.py), cs_collapse
(dark labelled connectors), both Grad-CAM galleries (same windows). NO
GENERATOR EXISTS for guide_data_explained.png and pipe_turn_v2.png — left
as-is, flagged. 78 pp, 0 overfull, 0 undefined.

**✅ USER READ-THROUGH ROUND 1 APPLIED (frontmatter + ch1, 22 Aug):** title
spelled out (no FMCW abbreviation; covers compacted so title/author/year fit
on one page — orphan "2026" pages gone, 74 pp); Summary + Resumen rewritten
(smaller, "difficulty walking" not "gait impairment", no cohort numbers, no
"pre-registered" jargon, simpler baseline wording, hedged priority claim);
§1.2 rewritten for flow; §1.3 contributions rewritten (hedge added, "frozen
before any result was known" deleted, no bare "that protocol"); §1.4 now a
nested bullet structure; **global no-hyphenation** (\hyphenpenalty=10000 +
widow/club penalties) — three headings that used to hyphenate got manual
breaks (§2.3, §2.5, annex A2); short LOF/LOT titles added to all 18 long
captions (one line per entry now); GDPR acronym typo fixed. 0 overfull,
0 undefined. Review worklist in REVIEW_FINDINGS.html still pending user's
decisions; user still reading ch2 onward.

**✅✅ REPORT COMPLETE FRONT TO BACK (22 Aug):** all remaining sections
transcribed into LaTeX after user review: abstract.tex (Resumen + Summary,
academic register, no result numbers), acknowledgements.tex (scaffold),
introduction.tex + objectives.tex (O1–O5, \label{sec:objectives};
conclusions.tex now tags each conclusion with the objectives it closes:
O2+O3 / O1+O5 / O5 / O1 / O4), annexA_ethics.tex (A1–A4 English content
under Spanish headings), annexB_budget.tex (template table with project
numbers: 360 h × 15 €/h; MacBook Pro 2021 M1 Max 32 GB at 3.200 € × 7/60
months = 373,33 €; GG 866,00; BI 398,36; fungible 400; subtotal 7.437,69;
IVA 1.561,92; TOTAL 8.999,61 € — table renumbered B.1). GDPR added to
acronyms. 76 pages, 0 overfull, 0 undefined, no placeholders anywhere.
Current action plan: REMAINING_WORK.md (rewritten 22 Aug).

**✅ REMAINING_SECTIONS.html drafted (22 Aug):** all unwritten report
sections in report order, awaiting user review before LaTeX transcription:
Resumen ES (~330 w) + Summary EN (~270 w) → frontmatter/abstract.tex;
acknowledgements scaffold; Ch1 Introduction (Motivation / Problem /
Contributions / Structure, cite keys marked inline: postuma2015,
mirelman2019, hausdorff2009, seifert2020, gurbuz2024, lopezdelgado2026,
hayashi2021, hoshiga2021, papanastasiou2020, ni2020) + Objectives O1–O5
(kept as the outline's O1–O5 so ch5 conclusions map back); §4.7: **DROPPED by user decision (22 Aug)** — placeholder removed from
results.tex, §3.9 "Duration controls" bullet reworded to the delivered
controls (fixed windows + duration_only floor + score-vs-window-count
correlation), ch2 safeguards list updated (duration-matched promise removed,
stale label-permutation promise also removed, null floors named instead);
the duration-matched/residualised/per-variant battery is now officially out
of scope; Annex A A1–A4 in English (language for annexes TBC with
school; A3 = overstated-clinical-claims analysis, uses thesis's own
numbers); Annex B budget computed in-script (360 h @ 15 €/h, PC
amortisation, GG 15%, BI 6%, IVA 21% → total ≈ 8.793 €; 12-ECTS hour
assumption flagged for user to confirm). Builder:
tools/build_remaining_html.py. Acronym list already complete.

**Citations for the ch5 transcription (user instruction, 22 Aug):** the AlexNet
subsection heading/content cites `\cite{hayashi2021}` (the field's architecture
and the distrusted 97.8% spectrogram CNN); the positive-findings paragraph
cites `\cite{fard2026}` (scores fell moving to leave-one-person-out) and
`\cite{hayashi2021}` (envelope model preferred over distrusted CNN); numeric
bracket style as everywhere else.

## ❓ For the supervisor: direction ambiguity in ce_foot

Under background-subtracted direction labels, **24/58 subjects have no window
with a single-direction foot signature** (all-or-nothing per subject). Likely
cause: the per-recording SNR combination of the two foot radars (paper flips one
node's Doppler axis before merging, p. 396–397). Question for Ignacio: is the
sign convention of `ce_foot` consistent across recordings, and is per-node data
(`all_ce_time`, `idx_foot_max_snr`) usable to recover left/right asymmetry?

---

## ▶️ Week 1 (Jul 30 – Aug 5) — ready to start, nothing blocking

**Reordered 2026-07-29: data before code.** The EDA is largely done but is not
reproducible and does not yet contain the batch or window-count findings. Days 1–2
close that gap; the code fixes move to Day 3, still ahead of the first training run.

| Day | Task |
|---|---|
| **1** | 🔴 **Data integrity** — repair `features.py`, regenerate `trial_features.csv`, re-run the whole analysis chain, diff every published number. Write Methods §3.1 |
| 2 | **Complete the data story** — integrate batch + window-count as first-class controls; decide windowing / normalisation / window-cap from the data. Write §3.2 + Results §4.1–4.2 |
| 3 | Fix the 4 code defects + `num_workers` + seeds; run preprocessing. Write §3.3 |
| 4 | SmallCNN smoke test → **pre-register config** → full LOSO (~18 min); `ce` vs `stft_abs`. Write §3.4 |
| 5 | Consolidate; **send Methods chapter to tutor** |

### Day 1 checklist — data integrity
- [ ] Restore the 9 missing feature columns in `features.py` behind an explicit flag
- [ ] Regenerate `trial_features.csv` from the `.mat` files (~1 min)
- [ ] Re-run `src.eda`, `src.baseline`, `src.confound_analysis`, `src.diagnostics`
- [ ] Diff against the published numbers: AUC grid, duration confound, permutation p
- [ ] Correct `RESULTS.md` for anything that moved, today
- [ ] Write Methods §3.1 (Dataset) into `chapters/ch3_methods/methods.tex`

### Day 3 checklist — the code blockers *(unchanged, just later)*
- [ ] `train.py` — replace held-out-subject early stopping with an inner-validation split (or a fixed epoch budget)
- [ ] `train.py` — assert `len(history) == epochs` so a silent 6-epoch run cannot recur
- [ ] `train.py` — add `checkpoint_dir` + `torch.save` per fold (Grad-CAM needs this)
- [ ] `train.py` — `num_workers=4, persistent_workers=True`
- [ ] `train.py` — seed every fold
- [ ] new `grouped_cv()` using `StratifiedGroupKFold`

---

## Standing risks

| Risk | Mitigation |
|---|---|
| Tutor unreachable | 🔴 *Visto bueno* is required to deposit. Escalation ladder: stated defaults by 1 Aug → second channel by 6 Aug → coordinator by 11 Aug. |
| Writing throughput | ~2.5 pages/day realistic; body is 50–60 pages. **Write every afternoon from Day 1.** Weeks 5–6 are revision, not drafting. |
| Mid-August in Spain | 15 Aug is a holiday and the surrounding weeks are peak vacation. **Send each chapter within 24 h of drafting**, not one batch. |
| Deep model may not beat the classical floor | Already accepted as a legitimate outcome — see `WORK_PLAN.md` §10C. |
| Committee unfamiliar with the domain | Coordinator's explicit warning. Explain confounds from first principles; lead with meaning, not numbers. |

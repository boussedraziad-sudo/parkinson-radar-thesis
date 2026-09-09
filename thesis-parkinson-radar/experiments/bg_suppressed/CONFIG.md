# Pre-registration: background-suppression experiment (bg_suppressed)

Frozen 2026-08-16, BEFORE any window of this cache was written or any model
trained. Nothing below changes after this point.

## Motivation

Grad-CAM on the v2 models showed attention on (a) the static background
texture that fills each recording's contrast-enhanced arrays and (b) the empty
band above 500 Hz, both of which are plausible carriers of the acquisition
batch fingerprint rather than gait. This experiment removes both carriers at
the input and re-runs the v2 protocol unchanged, to test whether any gait
signal remains once the background cannot be read.

## The two changes, and only these two

Applied per recording (per trial .mat), to each channel (ce_foot, ce_torso)
independently, BEFORE windowing:

1. Background subtraction. Compute the per-Doppler-bin background profile as
   the median over the time axis of the ce representation, clipped at zero:
   `x = max(max(ce, 0) - median_time(max(ce, 0)), 0)`. This is exactly the
   correction `positive_energy_fraction` (v2) applies when computing turn
   labels, promoted from label computation to the model input itself
   (see src/preprocessing.py).
2. Doppler crop to the gait band. Drop all Doppler rows with |f| > 500 Hz
   before resizing, so the empty high-frequency texture band the Grad-CAM
   exposed cannot be read. With the shared ~5 Hz bin spacing this keeps 201 of
   320 rows in both acquisition batches (verified on one trial per batch
   before freezing: batch B kept [-499.92, +499.92] Hz, batch A kept
   [-499.92, +499.92] Hz, 201 rows each).

The turn label `pos_frac_foot` is computed by the unchanged
`positive_energy_fraction` on the already background-subtracted, cropped
window with the cropped Doppler axis. It is cached for provenance only;
`TrainConfig.max_turn_frac` stays None, so no window is dropped.

## Everything else: identical to the v2 protocol

Preprocessing (mirrors PreprocConfig defaults, frozen 2026-08-06):

- representation "ce", window 3.0 s, stride 1.5 s (50 percent overlap)
- normalise "log" (log1p only; standardisation deferred to the Dataset)
- antialiased bilinear resize to 224 x 224 (src.preprocessing.resize_2d)
- 2 channels stacked (foot, torso), float32
- no trial exclusions (excluded_files.csv is empty, same as the v2 cache)
- manifest schema identical to v2 so fold_norm_stats and add_window_weights
  run unmodified

Training (src/train.py used as-is, no code changes):

- src.train.loso_cv, full 58-fold nested LOSO
- TrainConfig() defaults: epochs 20, lr 1e-3, weight_decay 1e-4, batch 32,
  patience 5, inner_val_subjects 8, channel_mode "both", augmentation on
  (Doppler flip, noise 0.10, mask 0.15), class_balance on effective mass,
  window_weighting on, max_turn_frac None, seed 1337, device mps
- fold-local standardisation from training subjects only (fold_norm_stats)
- fold seeds keyed to subject id (unchanged loso_cv behaviour)

Models (src/models.py used as-is):

- smallcnn_bg: SmallCNN(in_channels=2)
- resnet18_fc_bg: resnet18_finetune(in_channels=2, freeze_until="layer4")

## Outputs

All under experiments/bg_suppressed/:

- cache/ (windows + manifest.csv), outputs/{name}_folds.csv,
  outputs/{name}_summary.json, outputs/ckpt_{name}/,
  outputs/bg_suppressed_summary.json

## Metrics (definitions fixed here, matching the campaign summary exactly)

Verified before freezing by reproducing the published v2 numbers from
outputs/runs/*_folds.csv with these exact definitions:

- pooled subject AUC: roc_auc_score over per-subject mean window
  probabilities, all 58 subjects
- 95 percent CI: nonparametric bootstrap over subjects, 2000 resamples,
  percentile 2.5 / 97.5, numpy seed 1337
- within-batch AUC: same score restricted to batch A subjects (n=34) and
  batch B subjects (n=24) separately; batch labels from
  outputs/metrics/acquisition_batch.csv (constant per subject)
- corr_nwin: Pearson correlation of subject_prob with n_test_windows

## Reference values to compare against (outputs/runs/campaign_summary.json)

- smallcnn_v2: pooled 0.6158, batch_A 0.4208, batch_B 0.3481, corr_nwin 0.047
- resnet18_fc_v2: pooled 0.6545, batch_A 0.4583, batch_B 0.5407, corr_nwin 0.081
- batch label alone: pooled 0.664

## Interpretation rules, fixed in advance

- If pooled AUC stays near the v2 values while within-batch AUC stays at or
  below chance, the pooled number is still the batch fingerprint leaking
  through channels the suppression did not remove, and the negative
  within-batch finding stands.
- If pooled AUC drops toward 0.5, the v2 pooled numbers were carried largely
  by the background/texture carriers this experiment removed.
- If within-batch AUC rises materially above chance in BOTH batches and both
  models, that is the first evidence of a gait signal unmasked by background
  suppression; it would need a confirmation run before being believed.

## Procedure

1. Build the cache (deterministic, no fitting involved).
2. Smoke test: loso_cv on 2 subjects (first two, sorted order) per model to
   verify the pipeline end to end; results discarded, folds are identical to
   the full run's because fold seeds are keyed to subject ids.
3. Full 58-fold LOSO for smallcnn_bg then resnet18_fc_bg.
4. Write bg_suppressed_summary.json and report against the references.

Expected wall clock: cache ~10 min, smoke ~2 min, full runs ~30 min combined
(v2 references: 970 s + 866 s).

# Subject-Independent Classification of Parkinson's Disease from FMCW Radar Micro-Doppler Gait Signatures

Master's thesis, MSc in Signal Theory and Communications (MSTC), ETSIT-UPM, 2026.
Author: Ziad Boussedra. Supervisor: Juan Ignacio Godino-Llorente.

**Thesis (PDF):** [`Ziad_Boussedra_TFM/main.pdf`](Ziad_Boussedra_TFM/main.pdf)

## What is here

| Folder | Content |
|---|---|
| `Ziad_Boussedra_TFM/` | LaTeX source of the thesis (ETSIT template), figures and bibliography. Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`. |
| `thesis-parkinson-radar/src/` | The pipeline: data loading, quality control, windowing, handcrafted features, classical baselines, SmallCNN / ResNet-18 probe / Envelope-LSTM training under nested leave-one-subject-out validation, statistics, Grad-CAM. |
| `thesis-parkinson-radar/tools/` | Scripts that produce every figure and table in the thesis from the stored results. |
| `thesis-parkinson-radar/experiments/` | The post-hoc experiments of the discussion chapter: AlexNet under both validation protocols, background-suppressed inputs. |
| `thesis-parkinson-radar/classical_split/` | The collapse test: the field's random hold-out reproduced on this dataset and scored on excluded subjects. |
| `thesis-parkinson-radar/reports/`, `outputs/metrics/`, `outputs/figures/` | Aggregate results (AUCs, confidence intervals, group statistics) and the figures. |
| `thesis-parkinson-radar/*.md` | Working documents: literature review, methodology notes, results log, handbook. |

## What is not here, and why

- **The clinical recordings.** The 58-subject radar dataset belongs to the originating team (López-Delgado et al., UPM / Hospital General Universitario Gregorio Marañón) and stays under its ethics approval. The code expects it at `thesis-parkinson-radar/data/<subject>/<test>/<trial>/stft_data.mat`.
- **Per-subject derived files** (subject lists, per-recording features, per-fold predictions), model checkpoints and preprocessing caches: pseudonymised clinical identifiers and multi-gigabyte artefacts, both regenerable from the data with the code above.

## Reproducing

```bash
cd thesis-parkinson-radar
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.eda          # exploratory analysis and group statistics
.venv/bin/python -m src.baseline     # classical leave-one-subject-out baselines
.venv/bin/python -m src.train        # deep models under the nested protocol
```

The frozen, pre-registered training configuration is `src/train.py::TrainConfig`; every reported number comes from it unchanged.

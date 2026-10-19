# Cross-database evaluation and recalibration of a refitted mortality prediction model in ICU patients with heart failure

Reproduction code for the manuscript submitted to *Scientific Reports*.

This repository contains the analysis code, the model-ready public source data, the
derived result files, and the final figures used to produce every number, table and
figure reported in the manuscript. The analysis is a secondary study of three
publicly available, de-identified critical-care cohorts.

## Contents

- `scripts/` — analysis, figure-generation and audit code (Python; one Node script for the Word build).
- `data/` — model-ready cohorts exactly as released (see Data availability below).
- `outputs/` — derived numeric result files (JSON/CSV) that the figures and tables read from.
- `figures/` — final submission figures (PNG at 600 dpi plus vector PDF).
- `requirements.txt` — pinned Python environment.

## How to reproduce

From the repository root, with the pinned environment installed:

```bash
pip install -r requirements.txt
```

Run the analysis scripts in numeric order (they write into `outputs/`), then the figure
generator, then the consistency audit. A fixed random seed (2026) is set in the code, so
the reported numbers and figures are deterministic. The audit script re-derives each
reported figure and table from the stored results and returns a pass/fail status.

## Data availability

All source data are publicly available under CC0 licences via Dryad:

- MIMIC-III cohort: https://doi.org/10.5061/dryad.0p2ngf1zd
- MIMIC-IV derivation and eICU-CRD validation cohorts: https://doi.org/10.5061/dryad.tx95x6b18

No restricted-access request is required. The files in `data/` are the redistributed,
model-ready tables exactly as released.

## Notes and limitations

The analysis is retrospective. Predictor timing and hospital clustering could not be
independently verified in the redistributed files, and the within-eICU evaluations are
stay-level held-out assessments rather than independent hospital-level validation. The
results bound within-database evidence and do not support prospective or cross-hospital
deployment claims. Reporting follows the TRIPOD+AI guideline.

## License

Code is released for reproducibility of the manuscript; the underlying clinical data are
third-party public datasets under their own licences.

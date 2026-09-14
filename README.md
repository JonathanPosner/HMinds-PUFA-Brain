# Maternal PUFA intake and infant brain development — analysis code

Analysis code for:

> Posner J, Fay M, Michael A, Silva I, Abdala N, Coelho Milani AC, Oliveira Santana V,
> Shibuya Alves B, Wang Y, Duarte CS, Jackowski A. *Maternal prenatal polyunsaturated
> fatty acid intake and infant brain development: A prospective cohort study.*

Data are from Healthy MiNDS (Mother Influences on Child Neurobehavioral Development
Study), a prospective birth cohort in São Paulo and Guarulhos, Brazil.

## What is here

`main_analyses.py` reproduces the primary analyses reported in the manuscript:

| Section | Analysis | Reported as |
|---|---|---|
| 1 | Cohort descriptives | Table 1 |
| 2 | Subcortical volumes — MICE-imputed OLS, Benjamini–Hochberg FDR across 8 regions | Table 2 |
| 3 | Robust (Huber M-estimator) confirmation of FDR-surviving regions | Results text |
| 4 | PUFA × infant sex interaction | Results text |
| 5 | Subcortical–prefrontal functional connectivity — MICE + FDR across 32 pairs | Results text |
| 6 | Exploratory CBCL correlations and percentile bootstrap | Results text |
| 7 | Exploratory mediation (PUFA → right amygdala → CBCL Stress Problems) | Results text |

Sensitivity and supplementary analyses are not included. They reuse the same
functions on a filtered input frame.

## Data

**No participant data are in this repository, and none should be committed.**
`.gitignore` excludes `data/` and all `.csv` files for that reason.

The script expects one de-identified CSV, one row per mother–infant dyad, with the
column names listed in [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md). Nothing about the
file's contents is hard-coded; supply its location with `--data`.

De-identified data are available through the NIMH Data Archive (nda.nih.gov) under
collection C3811. Requests for additional de-identified data may be directed to the
corresponding author.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main_analyses.py --data path/to/analysis_dataset.csv --results-dir results/
```

Output is one CSV per analysis in `--results-dir`, plus a printed summary.

**Pin `lightgbm==4.5.0`.** miceforest 6.0.5 calls an internal LightGBM method whose
signature changed in 4.6, and a default install pulls 4.7.x, which fails immediately.
This is the most likely reason the pipeline will not run.

## Reproducibility notes

- Seeds are fixed: `123` for every MICE kernel, `1234` reset before each bootstrap block.
- 10 imputed datasets, 5 MICE iterations, random-forest MICE via `miceforest`.
- Estimates are pooled by Rubin's rules with the Barnard–Rubin degrees-of-freedom
  adjustment (`rubin_pool`).
- Predictor and outcome are both z-scored **inside each imputed dataset, after
  dropping rows missing that outcome**, so all reported β are fully standardized on
  the per-outcome analytic sample. Binary covariates stay on their 0/1 scale.
- Structural and connectivity tests are **separate FDR families** (8 and 32 tests).
- Confidence intervals in the published tables are β ± 1.96·SE (normal quantile).
- The CBCL correlations and the mediation model are exploratory: bivariate or
  unadjusted, and uncorrected for multiple comparisons.
- Outcomes are included in the imputation model, standard MICE practice, which means
  imputation is not independent of the outcome.

## License

MIT — see [`LICENSE`](LICENSE).

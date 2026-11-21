# RAMS: Resource-Adaptive Micronutrient Screening

This repository contains a curated set of the original Python scripts, locked modeling specifications, and aggregate result/audit files associated with the RAMS NHANES study.

## Contents
- `scripts/`: original stage 01–09 and RAMS Phase 2–5 Python scripts. Filenames preserved. The historical CatBoost repair script is omitted from this publication package.
- `configurations/`: original stage-05 manifests, temporal splits, feature lists, random seeds, hyperparameter search spaces, preprocessing and calibration specifications.
- `documentation/`: predictor and outcome definitions from the archived study.
- `results/`: selected aggregate results and statistical-audit summaries.

## Data
The underlying data are from the publicly available National Health and Nutrition Examination Survey (NHANES), CDC/NCHS: https://www.cdc.gov/nchs/nhanes/ . Download the relevant 2009–2018 and 2021–2023 components from the official source. The large intermediate joined files, analytic cohort extracts, and participant-level predictions are intentionally excluded from this package.

## Reproduction notes
The files are provided in their original form. Their execution from a fresh checkout has NOT been verified; some scripts may rely on local directory conventions or external inputs. Before describing this repository as fully reproducible, verify script dependencies, environment versions, file paths, and the end-to-end workflow.

## Software
The manuscript reports Python 3.13.1, scikit-learn 1.6.1, NumPy 2.2.2, pandas 2.2.3, and CatBoost, LightGBM and XGBoost. Exact versions for all additional dependencies should be captured from the original working environment.



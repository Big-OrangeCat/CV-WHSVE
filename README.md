# CV-WHSVE: PLR Drug-Use Screening Code

This repository contains code associated with the manuscript:

**Explainable Machine-Learning Framework for Non-Invasive Drug-Use Screening Based on Pupillary Light Reflex Dynamics**

This revision provides an executable, leakage-aware implementation of the
reported workflow. Real participant data, trained weights, predictions, and
generated scientific results are intentionally excluded.

## What is included

```text
src/
  cv_whsve/
    model_pool.py                       # 17 preprocessing-aware candidate pipelines
    ga_selection.py                     # GA subset selection from inner OOF probabilities
    nested_validation.py                # Nested selection, weighting, calibration and testing
    run_cv_whsve_with_shap.py           # Backward-compatible command-line entry point
    run_ablation_analysis.py           # Ablation/auxiliary model comparison script
    run_stacking_meta_comparison.py    # Stacking meta-classifier comparison script
  baselines_1d/
    train_1d_mobilenet_v3.py
    train_tsm_1d.py
    train_micro_bilstm.py
  tabular_deep/
    compare_utils.py
    compare_tabtransformer.py
    compare_tabm.py
    compare_tabr.py
    run_compare_3x.py
  run_repro_3x.py                       # Ten-seed mean and sample-SD analysis
scripts/
  generate_synthetic_data.py            # Non-clinical example-data generator
  smoke_test.py                          # Fast end-to-end executable check
  drug_category_metrics.py               # Metrics from an author-supplied category map
  shap_padding_sensitivity.py            # Controlled median-padding sensitivity analysis
data/
  synthetic_plr.csv                     # Simulated example only; no participant data
legacy_original_scripts/
  # selected traceable original scripts, renamed to ASCII filenames
```

## Data availability and synthetic example

The original participant-level PLR data and urine-screen labels are sensitive
human-participant data and are not included. The bundled
`data/synthetic_plr.csv` contains simulated curves solely to verify that the
code executes. It must not be used to reproduce, replace, or substantiate the
manuscript's scientific results.

Scripts expect a CSV file specified by the environment variable:

```bash
PLR_DATA_CSV=/path/to/preprocessed_data.csv
```

If the variable is not set, scripts use the placeholder path:

```text
data/preprocessed_data.csv
```

Only the bundled synthetic example may be committed from the `data/` directory. Do not add real or derived participant data unless the corresponding ethics approval, data-use agreement, and institutional permission explicitly allow public release.

## Expected CSV format

See [DATA_SCHEMA.md](DATA_SCHEMA.md).

In brief, the code expects:

- `D1` to `D125`: one 125-frame pupil-diameter sequence per participant;
- `label`: binary urine-screen label, where `0` denotes healthy control and `1` denotes positive urine drug screen.

## Validation design

The primary implementation follows these steps:

- participant-level stratified 80:20 split;
- random state `42` for the main held-out split in the 1D baseline scripts;
- extraction of 44 predefined engineered PLR features;
- removal of three invariant acquisition descriptors (`baseline_reference`,
  `stimulus_onset_frame`, and `sequence_length`), leaving the reported 41
  signal-derived model inputs;
- fold-specific median imputation, zero-variance filtering and, where needed,
  standardisation inside each classifier pipeline;
- five-fold outer and five-fold inner stratified cross-validation on the
  training partition;
- GA selection based only on inner out-of-fold probabilities, using
  `0.6 * F1 + 0.4 * AUROC`;
- ensemble weights estimated from nested outer-fold accuracy;
- probability calibration and threshold selection using nested training OOF
  predictions, followed by one evaluation on the held-out test partition;
- repeated complete analyses over ten random seeds with mean and sample SD;
- 1D sequence baselines trained with 80 epochs, batch size 64, AdamW, learning rate `3e-4`, and weight decay `1e-4`.

The file `training_feature_filter.csv` records the feature filter fitted on the
complete training partition for reporting. During validation, separate copies
of the same preprocessing steps are fitted inside each fold.

## Installation

The revision was tested with Python 3.8.5. Create an isolated environment and install the listed dependencies:

```bash
pip install -r requirements.txt
```

Some optional tabular deep-learning baselines may require additional packages depending on the local environment.

## Quick executable check

The following command regenerates the simulated CSV and runs a small two-fold
smoke test. This is a software check, not a scientific analysis.

```bash
python -m scripts.smoke_test
```

## Scientific analysis on an authorised local CSV

PowerShell:

```powershell
$env:PLR_DATA_CSV="D:\private_data\preprocessed_data.csv"
python -m src.cv_whsve.run_cv_whsve_with_shap --out outputs\main_nested
```

Bash:

```bash
export PLR_DATA_CSV=/private_data/preprocessed_data.csv
python -m src.cv_whsve.run_cv_whsve_with_shap --out outputs/main_nested
```

Ten-seed repeated evaluation:

```bash
python -m src.run_repro_3x --data /private_data/preprocessed_data.csv --out outputs/repeated_nested
```

Use the default `full` candidate profile for manuscript analyses. The `smoke`
profile exists only to keep the synthetic software test short.

## Reviewer-requested auxiliary analyses

Drug-category metrics require an author-supplied CSV with `row_id` and
`drug_category`. Categories are never inferred from binary labels or aggregate
counts. Each category is evaluated against the held-out control participants:

```bash
python -m scripts.drug_category_metrics \
  --predictions outputs/main_nested/test_predictions.csv \
  --categories /private_data/drug_categories.csv \
  --out outputs/main_nested/drug_category_metrics.csv
```

If original missing-position masks are unavailable, the following controlled
perturbation compares median padding with linear interpolation. Its results
must be described as a simulated sensitivity analysis, not as reconstruction
of the original missingness pattern:

```bash
python -m scripts.shap_padding_sensitivity \
  --data /private_data/preprocessed_data.csv \
  --weights outputs/main_nested/nested_weights.csv \
  --out outputs/shap_padding_sensitivity
```

## Generated outputs

Each analysis writes fold metrics, model-selection frequencies, nested weights,
calibration points, held-out metrics and local predictions to its output
directory. Output directories and CSV/JSON result files are ignored by Git so
participant-level predictions cannot be committed accidentally.

## Release checklist

Before making this repository public, verify that:

1. no participant-level data are committed;
2. no trained model weights are committed;
3. no generated result files, logs, or plots are committed;
4. no private absolute path remains in the public scripts;
5. the code DOI / release tag in the manuscript is updated after GitHub release.

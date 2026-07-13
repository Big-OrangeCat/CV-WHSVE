# PLR Drug-Use Screening Code

This repository contains code associated with the manuscript:

**Explainable Machine-Learning Framework for Non-Invasive Drug-Use Screening Based on Pupillary Light Reflex Dynamics**

The package was extracted from the local research codebase and reorganized for GitHub release. Data files, trained weights, generated figures, logs, and result tables are intentionally excluded.

## What is included

```text
src/
  cv_whsve/
    run_cv_whsve_with_shap.py          # Main CV-WHSVE pipeline with SHAP and correlation plots
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
  run_repro_3x.py
legacy_original_scripts/
  # selected traceable original scripts, renamed to ASCII filenames
```

## Data are not included

The original participant-level PLR data and urine-screen labels are sensitive human-participant data and are not included in this repository.

Scripts expect a CSV file specified by the environment variable:

```bash
PLR_DATA_CSV=/path/to/preprocessed_data.csv
```

If the variable is not set, scripts use the placeholder path:

```text
data/preprocessed_data.csv
```

Do not upload the `data/` directory unless the corresponding ethics approval, data-use agreement, and institutional permission explicitly allow public release.

## Expected CSV format

See [DATA_SCHEMA.md](DATA_SCHEMA.md).

In brief, the code expects:

- `D1` to `D125`: one 125-frame pupil-diameter sequence per participant;
- `label`: binary urine-screen label, where `0` denotes healthy control and `1` denotes positive urine drug screen.

## Relation to the manuscript

The cleaned code follows the manuscript description where code and draft differed:

- participant-level stratified 80:20 split;
- random state `42` for the main held-out split in the 1D baseline scripts;
- engineered PLR features and zero-variance filtering before CV-WHSVE modelling;
- GA-guided heterogeneous ensemble using five selected classifiers;
- post hoc SHAP/correlation analysis as model-behaviour description;
- 1D sequence baselines trained with 80 epochs, batch size 64, AdamW, learning rate `3e-4`, and weight decay `1e-4`.

Some legacy scripts keep older exploratory code paths for traceability. For a clean public release, cite the `src/` scripts first.

## Installation

```bash
pip install -r requirements.txt
```

Some optional tabular deep-learning baselines may require additional packages depending on the local environment.

## Example usage

PowerShell:

```powershell
$env:PLR_DATA_CSV="D:\private_data\preprocessed_data.csv"
python src\cv_whsve\run_cv_whsve_with_shap.py
```

Bash:

```bash
export PLR_DATA_CSV=/private_data/preprocessed_data.csv
python src/cv_whsve/run_cv_whsve_with_shap.py
```

## Important release note

Before making this repository public, verify that:

1. no participant-level data are committed;
2. no trained model weights are committed;
3. no generated result files, logs, or plots are committed;
4. no private absolute path remains in the public scripts;
5. the code DOI / release tag in the manuscript is updated after GitHub release.

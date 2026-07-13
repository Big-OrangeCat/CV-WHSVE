# Data schema

The repository does not include data. The scripts expect a local CSV file with one row per participant.

## Required columns

| Column | Description |
|---|---|
| `D1` ... `D125` | Pupil-diameter sequence values from 125 consecutive frames. |
| `label` | Binary class label. Use `0` for healthy controls and `1` for positive urine drug screens. |

## Optional identifier column

Some legacy scripts set the first column as an index. If an identifier column is present, keep it as the first column and ensure that `D1` ... `D125` and `label` remain available.

## Data release warning

Do not upload the real participant-level dataset, intermediate CSV files, predictions, or derived result tables to a public repository unless institutional and ethical approvals explicitly allow public release.

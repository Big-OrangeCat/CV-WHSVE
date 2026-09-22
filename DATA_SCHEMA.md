# Data schema

The repository does not include participant data. The scripts expect an authorised local CSV file with one row per participant; the bundled CSV is synthetic and is described below.

## Required columns

| Column | Description |
|---|---|
| `D1` ... `D125` | Pupil-diameter sequence values from 125 consecutive frames. |
| `label` | Binary class label. Use `0` for healthy controls and `1` for positive urine drug screens. |

## Optional non-identifying row key

The current loader recognises `id`, `sample_id`, `participant_id`, or
`Unnamed: 0`. This value is copied only to local prediction tables so that an
author-supplied subgroup table can be joined later. Use a study code rather
than names or other direct identifiers.

## Synthetic example

`data/synthetic_plr.csv` follows this schema but contains only simulated
curves. It is provided to demonstrate executability and is not a public or
de-identified version of the study dataset.

## Data release warning

Do not upload the real participant-level dataset, intermediate CSV files,
predictions, or derived result tables to a public repository unless
institutional and ethical approvals explicitly allow public release. The
repository `.gitignore` permits only the named synthetic example CSV.

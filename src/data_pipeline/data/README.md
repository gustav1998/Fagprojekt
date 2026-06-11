# Data

This folder contains the datasets used in the project.

## Raw Data

Raw datasets are stored under:

```text
src/data_pipeline/data/raw/
```

The raw files include the UCI-style datasets and the additional report datasets
used for the experiments. Dataset-specific paths are configured in:

```text
src/data_pipeline/dataset_configs.py
```

### UCI-Style Datasets

These are the smaller raw datasets stored directly under
`src/data_pipeline/data/raw/`.

| Dataset config | Source |
|---|---|
| `balance_scale` | https://archive.ics.uci.edu/dataset/12/balance+scale |
| `car_evaluation` | https://archive.ics.uci.edu/dataset/19/car+evaluation |
| `krkopt` | https://archive.ics.uci.edu/dataset/23/chess+king+rook+vs+king |
| `house_votes_84` | https://archive.ics.uci.edu/dataset/105/congressional+voting+records |
| `credit_approval` | https://archive.ics.uci.edu/dataset/27/credit+approval |
| `lenses` | https://archive.ics.uci.edu/dataset/58/lenses |
| `hayesroth` | https://archive.ics.uci.edu/dataset/44/hayes+roth |
| `monk_1` | https://archive.ics.uci.edu/dataset/70/monk+s+problems |
| `monk_2` | https://archive.ics.uci.edu/dataset/70/monk+s+problems |
| `monk_3` | https://archive.ics.uci.edu/dataset/70/monk+s+problems |
| `nursery` | https://archive.ics.uci.edu/dataset/76/nursery |
| `primary_tumor` | https://archive.ics.uci.edu/dataset/83/primary+tumor |

### Report Datasets

These are stored under `src/data_pipeline/data/raw/report_datasets/`.

| Dataset config | Source |
|---|---|
| `asia_lung` | https://www.openml.org/d/43151 |
| `cleveland` | https://www.openml.org/d/40711 |
| `conf_ad` | https://www.openml.org/d/41538 |
| `coronary` | https://www.openml.org/d/43154 |
| `dmft` | https://www.openml.org/d/469 |
| `german_gss` | https://www.openml.org/d/1025 |
| `led7` | https://www.openml.org/d/40678 |
| `mofn` | https://www.openml.org/d/40680 |
| `parity5p5` | https://epistasislab.github.io/pmlb/profile/parity5+5.html |
| `ppd` | https://www.openml.org/d/40683 |
| `ptumor` | https://www.openml.org/d/1003 |
| `sensory` | https://www.openml.org/d/826 |
| `three_of_nine` | https://www.openml.org/d/40690 |
| `vehicle` | https://www.openml.org/d/835 |
| `xd6` | https://www.openml.org/d/40693 |

### Large Datasets

Large raw datasets are stored locally under:

```text
src/data_pipeline/data/raw/large_datasets/
```

These large downloaded files are ignored by Git. The expected filenames and
source pages are listed in:

```text
src/data_pipeline/data/raw/large_datasets/README.md
```

## Processed Data

Processed train, validation, and test splits are stored under:

```text
src/data_pipeline/data/processed/
```

Each dataset has two processed representations:

```text
<dataset>_baseline_<split>.csv
<dataset>_tensor_<split>.csv
```

where `<split>` is one of:

```text
train
val
test
```

The baseline representation is used by logistic regression and the MLP. The
tensor representation is used by CPD, MBA, TT, and TR.

Preprocessing is fitted on the training split only. The fitted category
mappings, numerical fill values, and tensor discretization bins are then reused
for validation and test splits.

Rows with missing target labels are dropped before splitting. Datasets can also
define a minimum target-class count when very rare classes cannot be split
reliably.

Seeded experiment splits are stored in folders such as:

```text
src/data_pipeline/data/processed/seed_42/
```

The MONK datasets use their official test files. A validation split is carved
out of the official training file. Hayes-Roth keeps using the labeled
`hayes-roth.data` file because the companion `hayes-roth.test` file does not
include labels in the same supervised-learning format.

## Regenerating Processed Splits

To regenerate processed files from the raw data, run:

```bash
uv run python -m src.data_pipeline.make_dataset --dataset car_evaluation --representation both
```

For seeded experiment folders, use the experiment runner:

```bash
uv run python -m src.training.run_experiments --dataset car_evaluation --seed 42
```

## Visualizing Processed Data

To generate CSV summaries and SVG plots for the processed datasets, run:

```bash
uv run python -m src.data_pipeline.describe_datasets --representation both --seed 42
```

The generated report is written to:

```text
src/data_pipeline/data/reports/
```

This folder is ignored by Git because it is generated output.

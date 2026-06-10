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

# Data

This folder contains the datasets used in the project.

## Raw Data

Raw datasets are stored under:

```text
data/raw/
```

The raw files include the UCI-style datasets and the additional report datasets
used for the experiments. Dataset-specific paths are configured in:

```text
src/data/dataset_configs.py
```

## Processed Data

Processed train, validation, and test splits are stored under:

```text
data/processed/
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

Seeded experiment splits are stored in folders such as:

```text
data/processed/seed_42/
```

## Regenerating Processed Splits

To regenerate processed files from the raw data, run:

```bash
uv run python -m src.data.make_dataset --dataset car_evaluation --representation both
```

For seeded experiment folders, use the experiment runner:

```bash
uv run python -m src.run_experiments --dataset car_evaluation --seed 42
```


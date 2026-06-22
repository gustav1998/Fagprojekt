# Data Pipeline

The active preprocessing pipeline is implemented in:

```text
src/data_pipeline/make_dataset2.py
src/data_pipeline/preprocessing2.py
src/data_pipeline/load_processed2.py
src/data_pipeline/datamodule2.py
```

## Raw Data

Raw files are stored under:

```text
src/data_pipeline/data/raw/
```

Dataset-specific paths, column names, target names, missing-value tokens, and
feature types are defined in:

```text
src/data_pipeline/dataset_configs.py
```

## Splitting

`make_dataset2.py` creates:

```text
tuning_train
tuning_val
fold_1_train, fold_1_test
...
fold_5_train, fold_5_test
```

The tuning split is used for grid search. The five folds are used for final
cross-validation experiments.

`lenses` is skipped in the current K-fold pipeline because rare-class filtering
leaves too few usable classes.

## Representations

Every dataset is written in two processed representations:

```text
<dataset>_baseline_*.csv
<dataset>_tensor_*.csv
```

The baseline representation is used by:

```text
lr, mlp, rf
```

The tensor representation is used by:

```text
cpd, mba, tt, tr
```

Categorical features are mapped to integer IDs. Numerical features stay
continuous in the baseline representation and are discretized into bins in the
tensor representation.

## Generating Processed Data

One dataset:

```bash
uv run python -m src.data_pipeline.make_dataset2 \
  --dataset car_evaluation \
  --representation both
```

All current K-fold datasets:

```bash
for dataset in $(uv run python -c "from src.data_pipeline.dataset_configs import DATASET_CONFIGS; names = sorted(DATASET_CONFIGS); names.remove('lenses'); print(' '.join(names))"); do
  uv run python -m src.data_pipeline.make_dataset2 --dataset "$dataset" --representation both
done
```

Generated processed files are written to:

```text
src/data_pipeline/data/processed/
```

This folder is ignored by Git.

## Loading Processed Data

`load_processed2.py` reads either the tuning split or one CV fold:

```text
mode="tuning" -> tuning_train, tuning_val
mode="fold"   -> fold_<k>_train, fold_<k>_test
```

`datamodule2.py` wraps those dataframes in PyTorch DataLoaders. For fold runs,
the fold training portion is split internally into train/validation, while the
held-out fold is used as the test set.

## Dataset Reports

Processed-data diagnostics can be generated with:

```bash
uv run python -m src.data_pipeline.describe_datasets --representation both
```

The report is written to:

```text
src/data_pipeline/data/reports/
```

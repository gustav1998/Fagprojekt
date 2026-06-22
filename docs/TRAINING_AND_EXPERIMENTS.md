# Training and Experiments

The active training pipeline uses:

```text
src/training/train2.py
src/training/tune_hyperparameters2.py
src/training/run_experiments2.py
```

## Single Fold Training

Train one model on one dataset and one CV fold:

```bash
uv run python -m src.training.train2 \
  --dataset house_votes_84 \
  --model cpd \
  --mode fold \
  --fold 1 \
  --seed 42
```

Model choices:

```text
lr, mlp, cpd, mba, tt, tr, rf
```

Tensor options:

```bash
--rank 16                 # cpd, tt, tr
--interaction-order 2     # mba
```

Training uses early stopping by default. Disable it only for controlled
comparisons:

```bash
--disable-early-stopping
```

## Grid-Search Tuning

Grid search uses the tuning split created by `make_dataset2.py`:

```bash
uv run python -m src.training.tune_hyperparameters2 \
  --dataset house_votes_84 \
  --model cpd \
  --accelerator cpu
```

Best parameters are written to:

```text
src/summary_results/results/tuning/<model>_<dataset>_best_params.json
```

For MBA, the grid only includes interaction orders that fit under the parameter
budget controlled by:

```bash
--max-mba-order
--max-mba-parameters
```

## K-Fold Experiment Runner

Run all selected models across all five folds:

```bash
uv run python -m src.training.run_experiments2 \
  --dataset house_votes_84 \
  --model lr \
  --model mba \
  --seed 42 \
  --accelerator cpu
```

Run the full current benchmark after preprocessing:

```bash
uv run python -m src.training.run_experiments2 \
  --skip-preprocessing \
  --seed 42 \
  --accelerator cpu
```

The runner automatically reads tuning JSON files when they exist.

## Result Folders

Fold runs are written as:

```text
src/summary_results/results/<model>/<dataset>_fold<fold>_seed<seed>/
```

Each folder contains:

```text
metrics.csv
run_metadata.json
test_confusion_matrix.csv    # for models that write one
```

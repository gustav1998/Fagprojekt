Fagprojekt
==========

This repository contains the code used for the project experiments on
classification methods for discrete/tabular data.

Implemented models:

- `lr`: logistic regression baseline
- `mlp`: multilayer perceptron baseline
- `rf`: random forest baseline
- `cpd`: class-specific PARAFAC/CPD tensor classifier
- `mba`: many-body approximation classifier
- `tt`: class-specific tensor-train classifier
- `tr`: class-specific tensor-ring classifier

All models are supervised classifiers. Given an input

```text
x = (x_1, ..., x_D)
```

the model returns one logit per class. Probabilities are obtained with softmax
and training uses cross-entropy loss.

Repository Layout
-----------------

```text
docs/                         Project/code documentation
hpc/                          DTU HPC job scripts
src/data_pipeline/            Raw loading, K-fold preprocessing, dataset reports
src/models/                   Model implementations
src/training/                 Training, grid search, K-fold experiment runner
src/summary_results/          Result summarization, plotting, statistical tests
src/data_pipeline/data/raw/   Raw datasets used by the project
```

Generated files are ignored by Git:

```text
src/data_pipeline/data/processed/
src/summary_results/results/
results/
```

Setup
-----

The project uses Python 3.12 and `uv`.

```bash
uv sync
uv run python test_environment.py
```

On Windows PowerShell, a manually activated environment can be started with:

```powershell
.venv\Scripts\Activate.ps1
```

Data
----

Raw datasets live in:

```text
src/data_pipeline/data/raw/
```

Dataset paths and column settings are defined in:

```text
src/data_pipeline/dataset_configs.py
```

Generate the current tuning split plus 5-fold CV splits for one dataset:

```bash
uv run python -m src.data_pipeline.make_dataset2 \
  --dataset car_evaluation \
  --representation both
```

Generate processed files for all configured datasets except `lenses`, which is
excluded from the K-fold pipeline after rare-class filtering:

```bash
for dataset in $(uv run python -c "from src.data_pipeline.dataset_configs import DATASET_CONFIGS; names = sorted(DATASET_CONFIGS); names.remove('lenses'); print(' '.join(names))"); do
  uv run python -m src.data_pipeline.make_dataset2 --dataset "$dataset" --representation both
done
```

Training
--------

Train one model on one held-out fold:

```bash
uv run python -m src.training.train2 \
  --dataset car_evaluation \
  --model cpd \
  --mode fold \
  --fold 1 \
  --seed 42
```

Run a small local smoke experiment:

```bash
uv run python -m src.training.run_experiments2 \
  --dataset house_votes_84 \
  --model lr \
  --model mba \
  --epochs 1 \
  --interaction-order 1 \
  --accelerator cpu
```

Run grid-search tuning on the tuning split:

```bash
uv run python -m src.training.tune_hyperparameters2 \
  --dataset house_votes_84 \
  --model cpd \
  --accelerator cpu
```

Run the K-fold benchmark. If processed files already exist, use
`--skip-preprocessing`:

```bash
uv run python -m src.training.run_experiments2 \
  --skip-preprocessing \
  --accelerator cpu
```

Results
-------

Training writes logs under:

```text
src/summary_results/results/<model>/<dataset>_fold<fold>_seed<seed>/
```

Analysis outputs for the report are written to:

```text
results/
```

Generate all main tables and plots:

```bash
uv run python -m src.summary_results.analyze_results all \
  --dataset house_votes_84 \
  --model cpd
```

Useful individual commands:

```bash
uv run python -m src.summary_results.analyze_results tables
uv run python -m src.summary_results.analyze_results timing
uv run python -m src.summary_results.analyze_results stats --stats-metric test_acc
uv run python -m src.summary_results.analyze_results stats --stats-metric test_loss --lower-is-better
```

Documentation
-------------

Start with:

- `docs/README.md`
- `docs/MODEL_DESCRIPTIONS.md`
- `docs/DATA_PIPELINE.md`
- `docs/TRAINING_AND_EXPERIMENTS.md`
- `docs/RESULTS_AND_SUMMARIES.md`
- `docs/AI_USAGE.md`

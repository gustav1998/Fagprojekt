# Source Code Documentation

This folder explains the active project code used for the report submission.

## Documents

- `MODEL_DESCRIPTIONS.md`: model formulas, forward passes, loss, and optimizer.
- `DATA_PIPELINE.md`: raw data loading, tuning split, 5-fold CV preprocessing,
  and DataLoaders.
- `TRAINING_AND_EXPERIMENTS.md`: single-fold training, grid-search tuning, and
  K-fold experiment runs.
- `RESULTS_AND_SUMMARIES.md`: result logs, plots, summary tables, timing tables,
  and Friedman/Nemenyi tests.
- `CONFIGURATION_AND_UTILITIES.md`: dataset configs, one-hot encoding,
  dependencies, and environment checks.
- `HPC_JOBS.md`: current notes for running model jobs on the HPC.
- `AI_USAGE.md`: how AI assistance was used and verified.

## Active Project Flow

```text
raw CSV files
    -> src.data_pipeline.make_dataset2
    -> tuning split + 5-fold processed CSV files
    -> src.training.tune_hyperparameters2
    -> best parameter JSON files
    -> src.training.run_experiments2
    -> fold logs under src/summary_results/results/
    -> src.summary_results.analyze_results
    -> report tables/plots under results/
```

The baseline models use one-hot encoded input:

```text
lr, mlp -> baseline representation
rf      -> baseline representation as stored in processed CSV files
```

The tensor models use integer-valued feature indices:

```text
cpd, mba, tt, tr -> tensor representation
```

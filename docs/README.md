# Source Code Documentation

This folder explains the project code from top to bottom. The documents use
code snippets first, then explain what the code computes or controls.

## Documents

- `docs/MODEL_DESCRIPTIONS.md` explains the classifiers and shared Lightning
  training wrapper.
- `docs/DATA_PIPELINE.md` explains raw dataset loading, preprocessing,
  splitting, processed data loading, and the Lightning data module.
- `docs/TRAINING_AND_EXPERIMENTS.md` explains the command-line training script
  the multi-run experiment script, early stopping, and optional tuning.
- `docs/RESULTS_AND_SUMMARIES.md` explains how logged metrics are turned into
  benchmark summary tables.
- `docs/CONFIGURATION_AND_UTILITIES.md` explains dataset configuration,
  one-hot encoding, package configuration, and the environment smoke test.
- `docs/AI_USAGE.md` records how AI assistance was used and verified during
  the project.

## Project Flow

```text
raw CSV files
    -> src.data.make_dataset
    -> processed train/val/test CSV files and metadata
    -> src.train
    -> Lightning training logs under results/
    -> src.summarize_results
    -> benchmark summary CSV files
```

The baseline models use one-hot encoded input:

```text
lr, mlp -> baseline representation
```

The tensor models use integer-valued feature indices:

```text
cpd, mba, tt, tr -> tensor representation
```

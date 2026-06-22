# HPC Job Scripts

This document describes the current DTU HPC scripts in `hpc/`. The scripts use
the current fold-based pipeline:

- `src.training.tune_hyperparameters2`
- `src.training.run_experiments2`
- processed data from `src/data_pipeline/data/processed/`
- experiment outputs in `src/summary_results/results/`

The scripts are meant for benchmark runs after the raw datasets have already
been placed in `src/data_pipeline/data/raw/`.

## Script Layout

The current HPC folder splits jobs by model and, for the largest datasets, by
dataset:

```text
hpc/
  rf2.sh
  lr2_*.sh
  mlp2_*.sh
  cpd2_*.sh
  mba2_*.sh
  tt2_*.sh
  tr2_*.sh
```

The `*_rest.sh` scripts run the smaller and medium datasets together. The
dataset-specific scripts handle larger datasets such as `aps_failure`,
`census_income_kdd`, `phiusiil_phishing`, `rt_iot2022`, and
`skin_segmentation`.

## What Each Job Does

Each job follows the same basic pattern:

```bash
python3 -m src.training.tune_hyperparameters2 \
    --model <model> \
    --dataset <dataset> \
    --accelerator gpu \
    --num-workers 4

python3 -m src.training.run_experiments2 \
    --model <model> \
    --dataset <dataset> \
    --seed 42 \
    --skip-preprocessing \
    --accelerator gpu \
    --num-workers 4
```

Random Forest runs on CPU and therefore does not use `--accelerator gpu`.

## Submitting Jobs

From the HPC account, submit a script with:

```bash
bsub < hpc/cpd2_rest.sh
```

Submit one script at a time while testing the setup. Once the environment and
paths are confirmed, related model scripts can be submitted in parallel.

## Important Notes

- The scripts currently fetch `origin/main` and reset the HPC checkout to it
  before running. Only submit them when the branch state on GitHub is the one
  you want the cluster to use.
- The scripts remove checkpoints and fitted Random Forest model files after
  each run to keep result folders small.
- Generated results are ignored by Git in this repository. Copy back only the
  final CSV/plot outputs that are needed for the report.
- If a job times out, resubmit the corresponding script with a larger `#BSUB -W`
  value rather than changing the Python code.

## Local Equivalent

The local command shape is the same, but uses `uv`:

```bash
uv run python -m src.training.tune_hyperparameters2 --model cpd --dataset house_votes_84 --accelerator cpu
uv run python -m src.training.run_experiments2 --model cpd --dataset house_votes_84 --seed 42 --skip-preprocessing --accelerator cpu
```

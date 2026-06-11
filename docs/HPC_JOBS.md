# HPC Job Scripts

This document describes the LSF job scripts used to run the full benchmark pipeline
on DTU's HPC cluster. All scripts live in the repository root and are submitted via
`bsub < <script>.sh` from the home directory on the HPC.

---

## Pipeline Overview

The full pipeline runs in two stages:

1. **Preprocessing** (`submit-preprocess.sh`) — generates train/val/test splits for
   all 27 datasets across all 6 seeds. Must complete before any model job starts.
2. **Model jobs** (`rf.sh`, `lr.sh`, ...) — one job per model, run in parallel after
   preprocessing. Each job tunes hyperparameters on seed 42, then trains and evaluates
   on seeds 1–5 and 42.

Hyperparameter tuning uses Optuna with 25 trials per dataset (seed 42 only). The best
parameters are saved to `src/summary_results/results/tuning/` and automatically picked
up by the training step.

---

## Submitting All Jobs

Submit the preprocessing job first, then all 7 model jobs once it finishes:

```bash
# Submit preprocessing
bsub < submit-preprocess.sh

# Once preprocessing is done (check with bjobs), submit all model jobs:
for f in rf.sh lr.sh mlp.sh cpd.sh mba.sh tt.sh tr.sh; do
    bsub < ~/$f
done
```

To submit model jobs with an automatic dependency on preprocessing finishing:

```bash
PREPJOB=<jobid>
for f in rf.sh lr.sh mlp.sh cpd.sh mba.sh tt.sh tr.sh; do
    bsub -w "done($PREPJOB)" < ~/$f
done
```

Monitor job status with `bjobs` or inspect a specific job with `bjobs -l <jobid>`.

---

## Shared Setup (All Scripts)

Every job script begins with the same environment setup block:

```bash
unset PYTHONHOME
unset PYTHONPATH
module load python3/3.11.9
export PATH=/appl9/python/3.11.9/bin:$PATH
cd ~/Fagprojekt
git pull
python3 -m pip install -e . --quiet
```

- `unset PYTHONHOME` / `unset PYTHONPATH` — clears any conflicting Python environment
  variables that would cause the wrong Python interpreter to be used.
- `module load python3/3.11.9` — loads the correct Python version from the module system.
- `export PATH=...` — ensures the module Python takes precedence over the system Python.
- `git pull` — keeps the job running the latest version of the code.
- `pip install -e .` — installs any new or changed dependencies before running.

---

## RF (`rf.sh`)

```
Queue:     hpc (CPU only)
Cores:     16
RAM:       64 GB (16 x 4 GB per slot)
Wall time: 24 h
```

Random Forest is a scikit-learn model and does not use a GPU. 16 cores are requested
because the RF trainer uses `n_jobs=-1`, which parallelises tree building across all
available cores. More cores directly reduces training time for the 300-tree forest.

No `--accelerator` flag is needed since RF bypasses the Lightning trainer entirely.

---

## LR (`lr.sh`)

```
Queue:     gpuv100
Cores:     8
RAM:       32 GB (8 x 4 GB per slot)
GPU:       1x V100 (exclusive process)
Wall time: 24 h
```

Logistic Regression is implemented as a single linear layer in PyTorch Lightning.
Although it is a simple model, running 25 tuning trials across 27 datasets with
`--accelerator gpu` keeps total runtime within 24 hours, particularly for the larger
datasets such as `census_income_kdd` and `aps_failure`.

---

## MLP (`mlp.sh`)

```
Queue:     gpuv100
Cores:     8
RAM:       32 GB (8 x 4 GB per slot)
GPU:       1x V100 (exclusive process)
Wall time: 24 h
```

Multi-layer perceptron with tuned hidden dimension (64–512) and dropout (0–0.5).
Slightly heavier than LR but still expected to finish within 24 hours on GPU.

---

## CPD (`cpd.sh`)

```
Queue:     gpuv100
Cores:     8
RAM:       64 GB (8 x 8 GB per slot)
GPU:       1x V100 (exclusive process)
Wall time: 24 h
```

CANDECOMP/PARAFAC decomposition classifier. Tensor rank is tuned over [4, 8, 16, 32, 64].
Higher RAM is requested compared to LR/MLP because tensor operations on large datasets
can require more intermediate memory. 24 hours is the optimistic estimate; 48 hours may
be needed if large datasets dominate.

---

## MBA (`mba.sh`)

```
Queue:     gpuv100
Cores:     8
RAM:       64 GB (8 x 8 GB per slot)
GPU:       1x V100 (exclusive process)
Wall time: 24 h
```

Multilinear Background Architecture classifier. Interaction order is tuned up to a
maximum of 3, subject to a hard parameter budget of 5,000,000 explicit parameters
(`--max-mba-order 3 --max-mba-parameters 5000000`). The budget cap prevents high-
dimensional datasets from producing interaction tables with hundreds of millions of
parameters. Allowed orders per dataset are printed at the start of each tuning run.

---

## TT (`tt.sh`)

```
Queue:     gpuv100
Cores:     8
RAM:       64 GB (8 x 8 GB per slot)
GPU:       1x V100 (exclusive process)
Wall time: 24 h
```

Tensor Train classifier. Tensor rank tuned over [4, 8, 16, 32, 64]. Resource
requirements match CPD as the model complexity and dataset scale are similar.

---

## TR (`tr.sh`)

```
Queue:     gpuv100
Cores:     8
RAM:       64 GB (8 x 8 GB per slot)
GPU:       1x V100 (exclusive process)
Wall time: 24 h
```

Tensor Ring classifier. Identical resource requirements to TT. The ring topology
adds one extra core connection compared to TT but does not significantly change
memory or compute demands at the ranks used here.

---

## Notes

- **If a job hits the 24h wall time limit**, resubmit with `#BSUB -W 48:00`. The GPU
  jobs that are most likely to need this are CPD, MBA, TT, and TR on the largest datasets.
- **GPU mode** — `exclusive_process` ensures no other job shares the GPU during training,
  preventing out-of-memory errors from concurrent jobs.
- **`--skip-preprocessing`** is passed to both `tune_hyperparameters` and
  `run_experiments` in all model jobs, since preprocessing is handled separately by
  `submit-preprocess.sh`.
- **Results** are committed and pushed to the repository at the end of each job, so
  partial results are preserved even if some jobs finish before others.

---

## Full Example: cpd.sh

```bash
#!/bin/bash
#BSUB -q gpuv100
#BSUB -J cpd
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o cpd_%J.out
#BSUB -e cpd_%J.err
#BSUB -u <your-dtu-email>

unset PYTHONHOME
unset PYTHONPATH
module load python3/3.11.9
export PATH=/appl9/python/3.11.9/bin:$PATH
cd ~/Fagprojekt
git pull
python3 -m pip install -e . --quiet

python3 -m src.training.tune_hyperparameters \
    --model cpd \
    --skip-preprocessing \
    --accelerator gpu \
    --num-workers 4

python3 -m src.training.run_experiments \
    --model cpd \
    --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --seed 42 \
    --skip-preprocessing \
    --accelerator gpu \
    --num-workers 4

git config user.email "<your-dtu-email>"
git config user.name "<your-name>"
git add src/summary_results/results/
git commit -m "CPD results: all datasets, 6 seeds"
git push
```
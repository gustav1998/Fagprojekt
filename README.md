Fagprojekt
==========

This project compares supervised classifiers for discrete/tabular data.

The current implemented models are:

- `lr`: logistic regression baseline
- `mlp`: multilayer perceptron baseline
- `cpd`: supervised CPD tensor classifier
- `mba`: supervised many-body approximation classifier
- `tt`: supervised tensor-train classifier
- `tr`: supervised tensor-ring classifier

Planned models from the project report are:

- `rf`: random forest baseline

Project Idea
------------

All models are trained as supervised classifiers. Given an input

    x = (x_1, ..., x_D)

each model returns one logit per class. The class probabilities are computed with
softmax:

    P(y = c | x) = softmax(logits)_c

The tensor models differ only in how the class logits are parameterized:

- CPD uses a low-rank product of feature factors.
- MBA uses interaction terms up to a chosen order.
- TT uses a tensor-train contraction of feature cores.
- TR uses a tensor-ring contraction with class-specific closing matrices.

See `docs/MODEL_DESCRIPTIONS.md` for a code-by-code explanation of the implemented
models and the corresponding mathematical formulas.

Documentation
-------------

The code documentation is split by area:

- `docs/MODEL_DESCRIPTIONS.md`: model definitions, forward passes, loss, and optimizer.
- `docs/DATA_PIPELINE.md`: raw loading, preprocessing, splitting, processed data loading, and DataLoaders.
- `docs/TRAINING_AND_EXPERIMENTS.md`: single training runs and multi-run experiments.
- `docs/RESULTS_AND_SUMMARIES.md`: metric logs, majority baselines, and benchmark summaries.
- `docs/CONFIGURATION_AND_UTILITIES.md`: dataset configs, one-hot encoding, package setup, and environment checks.

Setup
-----

The repository includes `.python-version` with Python 3.12. This avoids
bleeding-edge Python compatibility issues while still working on older Intel
Macs through the PyTorch version markers in `pyproject.toml`.

The simplest cross-platform setup is to sync the locked environment and run the
smoke test through `uv`:

    uv sync
    uv run python test_environment.py

If you prefer to activate the virtual environment manually, create it first:

    uv venv .venv

On macOS/Linux:

    source .venv/bin/activate

On Windows PowerShell:

    .venv\Scripts\Activate.ps1

On Windows Command Prompt:

    .venv\Scripts\activate.bat

Then install the project and run the environment check:

    uv pip install -e .
    python test_environment.py

Data
----

Raw datasets live in `data/raw`. Processed splits live in `data/processed`.

Create processed train/validation/test splits for one dataset:

    uv run python -m src.data.make_dataset --dataset car_evaluation --representation both

The supported dataset names are defined in `src/data/dataset_configs.py`.

Additional report datasets should be downloaded manually and saved as CSV files
under:

    data/raw/report_datasets/

Each CSV should use `class` as the target column name. After placing a file,
create processed splits in the usual way:

    uv run python -m src.data.make_dataset --dataset asia_lung --representation both

The OpenML report datasets use these sources and expected local filenames:

| Dataset | Source | Expected file |
| --- | --- | --- |
| `asia_lung` | OpenML dataset 43151: `https://www.openml.org/d/43151` | `data/raw/report_datasets/asia_lung.csv` |
| `cleveland` | OpenML dataset 40711: `https://www.openml.org/d/40711` | `data/raw/report_datasets/cleveland.csv` |
| `conf_ad` | OpenML dataset 41538: `https://www.openml.org/d/41538` | `data/raw/report_datasets/conf_ad.csv` |
| `coronary` | OpenML dataset 43154: `https://www.openml.org/d/43154` | `data/raw/report_datasets/coronary.csv` |
| `dmft` | OpenML dataset 469: `https://www.openml.org/d/469` | `data/raw/report_datasets/dmft.csv` |
| `german_gss` | OpenML dataset 1025: `https://www.openml.org/d/1025` | `data/raw/report_datasets/german_gss.csv` |
| `led7` | OpenML dataset 40678: `https://www.openml.org/d/40678` | `data/raw/report_datasets/led7.csv` |
| `mofn` | OpenML dataset 40680: `https://www.openml.org/d/40680` | `data/raw/report_datasets/mofn.csv` |
| `ppd` | OpenML dataset 40683: `https://www.openml.org/d/40683` | `data/raw/report_datasets/ppd.csv` |
| `ptumor` | OpenML dataset 1003: `https://www.openml.org/d/1003` | `data/raw/report_datasets/ptumor.csv` |
| `sensory` | OpenML dataset 826: `https://www.openml.org/d/826` | `data/raw/report_datasets/sensory.csv` |
| `three_of_nine` | OpenML dataset 40690: `https://www.openml.org/d/40690` | `data/raw/report_datasets/three_of_nine.csv` |
| `vehicle` | OpenML dataset 835: `https://www.openml.org/d/835` | `data/raw/report_datasets/vehicle.csv` |
| `xd6` | OpenML dataset 40693: `https://www.openml.org/d/40693` | `data/raw/report_datasets/xd6.csv` |

`parity5p5` is already available locally. If it needs to be recreated later,
download `parity5+5` from PMLB and save it as:

    data/raw/report_datasets/parity5p5.csv

Training
--------

Train and test one model on one dataset:

    uv run python -m src.train --dataset car_evaluation --model cpd

Useful examples:

    uv run python -m src.train --dataset car_evaluation --model lr
    uv run python -m src.train --dataset car_evaluation --model mlp
    uv run python -m src.train --dataset car_evaluation --model cpd --rank 32
    uv run python -m src.train --dataset car_evaluation --model mba --interaction-order 3
    uv run python -m src.train --dataset car_evaluation --model tt --rank 32
    uv run python -m src.train --dataset car_evaluation --model tr --rank 32

Run a seeded experiment. This writes results to
`results/<model>/<dataset>_seed<seed>/metrics.csv`:

    uv run python -m src.train --dataset car_evaluation --model cpd --seed 42

Run multiple datasets, models, and seeds:

    uv run python -m src.run_experiments --seed 1 --seed 2 --seed 3

Limit a multi-seed run to selected datasets or models by repeating options:

    uv run python -m src.run_experiments \
      --dataset car_evaluation \
      --dataset asia_lung \
      --model cpd \
      --model mba \
      --model tt \
      --model tr \
      --seed 1 \
      --seed 2 \
      --seed 3

Model Inputs
------------

The data module chooses the feature representation from the model name:

- `lr` and `mlp` use the `baseline` representation: categorical features are one-hot encoded and numerical features stay numerical.
- `cpd`, `mba`, `tt`, and `tr` use the `tensor` representation: all features are integer indices. Numerical features are discretized during preprocessing.

Default Hyperparameters
-----------------------

The training CLI uses model-specific defaults:

- `lr`: 100 epochs, learning rate `1e-3`
- `mlp`: 100 epochs, learning rate `1e-3`
- `cpd`: 60 epochs, learning rate `1e-2`, rank 16
- `mba`: 60 epochs, learning rate `1e-2`, interaction order 3
- `tt`: 60 epochs, learning rate `1e-2`, rank 16
- `tr`: 60 epochs, learning rate `1e-2`, rank 16

The CPD defaults are intentionally stronger than the old smoke-test settings.
With `1e-3`, CPD tended to collapse to the majority class.

Training uses early stopping by default. The best validation checkpoint is
tested instead of the final epoch:

    uv run python -m src.train --dataset car_evaluation --model cpd --epochs 150 --patience 15

Disable early stopping only for controlled comparisons:

    uv run python -m src.train --dataset car_evaluation --model cpd --disable-early-stopping

Optional hyperparameter tuning is separate from normal training:

    uv run python -m src.tune_hyperparameters --dataset car_evaluation --model cpd --trials 25

The tuning script optimizes validation performance and writes best parameters
under `results/tuning/`. Final benchmarks should use fixed settings.

Results
-------

Lightning writes metrics to:

    results/<model>/<dataset>/metrics.csv

Build a summary table with test accuracy, balanced accuracy, macro/weighted F1,
majority-class baseline, parameter counts, timings, and best model per dataset:

    uv run python -m src.summarize_results

When seeded result folders are present, the summary also writes:

    results/benchmark_summary_aggregate.csv

Current results should be treated as preliminary because not all planned models,
datasets, and metrics have been run yet.

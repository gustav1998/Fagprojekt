Fagprojekt
==========

This project compares supervised classifiers for discrete/tabular data.

The current implemented models are:

- `lr`: logistic regression baseline
- `mlp`: multilayer perceptron baseline
- `cpd`: supervised CPD tensor classifier
- `tt`: supervised tensor-train classifier
- `tr`: supervised tensor-ring classifier

Planned models from the project report are:

- `rf`: random forest baseline
- `mba`: many-body approximation classifier

Project Idea
------------

All models are trained as supervised classifiers. Given an input

    x = (x_1, ..., x_D)

each model returns one logit per class. The class probabilities are computed with
softmax:

    P(y = c | x) = softmax(logits)_c

The tensor models differ only in how the class logits are parameterized:

- CPD uses a low-rank product of feature factors.
- MBA should use interaction terms up to a chosen order.
- TT uses a tensor-train contraction of feature cores.
- TR uses a tensor-ring contraction with class-specific closing matrices.

See `MODEL_DESCRIPTIONS.md` for a code-by-code explanation of the implemented
models and the corresponding mathematical formulas.

Setup
-----

Create an environment and install the project:

    uv venv .venv
    source .venv/bin/activate
    uv pip install -e .
    python test_environment.py

Or sync from the lockfile:

    uv sync

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
    uv run python -m src.train --dataset car_evaluation --model tt --rank 32
    uv run python -m src.train --dataset car_evaluation --model tr --rank 32

Model Inputs
------------

The data module chooses the feature representation from the model name:

- `lr` and `mlp` use the `baseline` representation: categorical features are one-hot encoded and numerical features stay numerical.
- `cpd`, `tt`, `tr`, and `mba` use the `tensor` representation: all features are integer indices. Numerical features are discretized during preprocessing.

Default Hyperparameters
-----------------------

The training CLI uses model-specific defaults:

- `lr`: 100 epochs, learning rate `1e-3`
- `mlp`: 100 epochs, learning rate `1e-3`
- `cpd`: 60 epochs, learning rate `1e-2`, rank 16
- `tt`: 60 epochs, learning rate `1e-2`, rank 16
- `tr`: 60 epochs, learning rate `1e-2`, rank 16

The CPD defaults are intentionally stronger than the old smoke-test settings.
With `1e-3`, CPD tended to collapse to the majority class.

Results
-------

Lightning writes metrics to:

    results/<model>/<dataset>/metrics.csv

Build a summary table with test accuracy, majority-class baseline, and best
model per dataset:

    uv run python -m src.summarize_results

Current results should be treated as preliminary because not all planned models,
datasets, and metrics have been run yet.

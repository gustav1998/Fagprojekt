Fagprojekt
==========

This project compares supervised classifiers for discrete/tabular data.

The current implemented models are:

- `lr`: logistic regression baseline
- `mlp`: multilayer perceptron baseline
- `cpd`: supervised CPD tensor classifier

Planned models from the project report are:

- `rf`: random forest baseline
- `mba`: many-body approximation classifier
- `tt`: tensor-train classifier
- `tr`: tensor-ring classifier

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
- TT and TR should replace the CPD score with tensor-train or tensor-ring contractions.

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

Training
--------

Train and test one model on one dataset:

    uv run python -m src.train --dataset car_evaluation --model cpd --epochs 20

Useful examples:

    uv run python -m src.train --dataset car_evaluation --model lr
    uv run python -m src.train --dataset car_evaluation --model mlp
    uv run python -m src.train --dataset car_evaluation --model cpd --rank 8

Model Inputs
------------

The data module chooses the feature representation from the model name:

- `lr` and `mlp` use the `baseline` representation: categorical features are one-hot encoded and numerical features stay numerical.
- `cpd`, `mba`, `tt`, and `tr` use the `tensor` representation: all features are integer indices. Numerical features are discretized during preprocessing.

Results
-------

Lightning writes metrics to:

    results/<model>/<dataset>/metrics.csv

Current results should be treated as preliminary because not all planned models,
datasets, and metrics have been run yet.

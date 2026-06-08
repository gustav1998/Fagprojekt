# Training and Experiments

This document explains the command-line training path and the script used to
run many dataset/model/seed combinations.

## Single Training Run

File: `src/train.py`

### Model Defaults

```python
DEFAULT_TRAINING_CONFIGS = {
    "lr": {"epochs": 100, "learning_rate": 1e-3},
    "mlp": {"epochs": 100, "learning_rate": 1e-3},
    "cpd": {"epochs": 60, "learning_rate": 1e-2, "rank": 16},
    "mba": {"epochs": 60, "learning_rate": 1e-2, "interaction_order": 3},
    "tt": {"epochs": 60, "learning_rate": 1e-2, "rank": 16},
    "tr": {"epochs": 60, "learning_rate": 1e-2, "rank": 16},
}
```

These are the default hyperparameters used when the command line does not
override them. Tensor models use a larger default learning rate because their
parameters are initialized with small random values and are often trained on
integer feature lookups.

### Command-Line Arguments

```python
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--model", type=str, required=True, choices=["lr", "mlp", "cpd", "mba", "tt", "tr"])
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--learning-rate", type=float, default=None)
parser.add_argument("--rank", type=int, default=None)
parser.add_argument("--interaction-order", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
```

The required arguments identify the dataset and model. The optional arguments
control training length, optimization, tensor rank, MBA interaction order, and
reproducibility.

### Applying Defaults

```python
if args.epochs is None:
    args.epochs = defaults["epochs"]
if args.learning_rate is None:
    args.learning_rate = defaults["learning_rate"]
if args.rank is None:
    args.rank = defaults.get("rank", 8)
if args.interaction_order is None:
    args.interaction_order = defaults.get("interaction_order", 3)
```

Unset values are filled from `DEFAULT_TRAINING_CONFIGS`. This means a basic
command still runs with model-specific settings:

```bash
uv run python -m src.train --dataset house_votes_84 --model cpd
```

### Seeding

```python
if args.seed is not None:
    L.seed_everything(args.seed, workers=True)
```

The seed controls model initialization and data shuffling. Running multiple
seeds gives a better estimate of model stability.

### Data Module Construction

```python
datamodule = TabularDataModule(
    dataset_name=args.dataset,
    batch_size=args.batch_size,
    model_type=args.model,
    processed_dir=args.processed_dir,
    seed=args.seed,
)
datamodule.setup()
```

The data module loads the processed splits and converts them into the input
representation expected by the selected model.


## Model Selection

### Baseline Models

```python
if args.model == "lr":
    base_model = LogisticRegression(
        input_dim=datamodule.input_dim,
        num_classes=datamodule.num_classes,
    )
```

Logistic regression uses the one-hot baseline representation and needs the
final input dimension:

```text
D = sum_d I_d + number of numerical features.
```

```python
elif args.model == "mlp":
    base_model = MLPClassifier(
        input_dim=datamodule.input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=datamodule.num_classes,
        dropout=args.dropout,
    )
```

The MLP also uses the one-hot baseline representation.

### Tensor Models

```python
elif args.model == "cpd":
    base_model = CPDClassifier(
        feature_dims=datamodule.cardinalities,
        rank=args.rank,
        num_classes=datamodule.num_classes,
    )
```

CPD receives the feature cardinalities:

```text
(I_1, ..., I_D)
```

and a tensor rank `R`.

```python
elif args.model == "mba":
    base_model = MBAClassifier(
        feature_dims=datamodule.cardinalities,
        interaction_order=args.interaction_order,
        num_classes=datamodule.num_classes,
    )
```

MBA receives the maximum interaction order `K`. The model includes interaction
tables for feature subsets up to that order.

```python
elif args.model == "tt":
    base_model = TTClassifier(...)
elif args.model == "tr":
    base_model = TRClassifier(...)
```

TT and TR receive feature cardinalities and rank, then construct tensor cores
for integer-valued feature inputs.


## Shared Training Wrapper

### Lightning Module

```python
model = TabularClassifierModule(
    model=base_model,
    learning_rate=args.learning_rate,
)
```

The base model defines only:

```text
x -> logits
```

The Lightning wrapper defines the shared loss, optimizer, accuracy logging, and
balanced accuracy logging. The model-specific math is described in
`docs/MODEL_DESCRIPTIONS.md`.


## Logging

### Result Version

```python
if result_version is None:
    result_version = (
        f"{args.dataset}_seed{args.seed}"
        if args.seed is not None
        else args.dataset
    )
```

The result version becomes the folder name under:

```text
results/<model>/<result_version>/
```

For seeded experiments, this creates paths such as:

```text
results/cpd/house_votes_84_seed42/
```

### CSV Logger

```python
logger = CSVLogger(
    save_dir="results",
    name=args.model,
    version=result_version,
)
```

Lightning writes metrics to:

```text
results/<model>/<version>/metrics.csv
```

The summary script later reads these files.


## Trainer

### Fit and Test

```python
trainer = L.Trainer(
    max_epochs=args.epochs,
    accelerator=args.accelerator,
    devices=1,
    logger=logger,
    enable_checkpointing=False,
    log_every_n_steps=1,
)

trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule)
```

Training optimizes parameters on the training split and logs validation metrics
during training. Testing evaluates the final model on the held-out test split.

Checkpointing is disabled because the experiments currently compare final
training outcomes rather than saved model files.


## Multi-Run Experiments

File: `src/run_experiments.py`

### Models

```python
MODELS = ["lr", "mlp", "cpd", "mba", "tt", "tr"]
```

If the command does not specify models, every model in this list is run.

### Command Execution

```python
def run_command(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)
```

Each preprocessing or training command is executed as a subprocess. If one
command fails, the experiment script stops immediately.

### Dataset, Model, and Seed Loops

```python
for seed in seeds:
    processed_dir = processed_root / f"seed_{seed}"

    for dataset in selected_datasets:
        run_command([... "src.data.make_dataset", ...])

        for model in selected_models:
            run_command([... "src.train", ...])
```

The experiment script loops over:

```text
seeds x datasets x models
```

For each seed and dataset, it first creates processed data in:

```text
data/processed/seed_<seed>/
```

Then it trains every selected model on those processed splits.

### Passing Model-Specific Parameters

```python
if model in {"cpd", "tt", "tr"}:
    command.extend(["--rank", str(rank or config["rank"])])
if model == "mba":
    command.extend(["--interaction-order", str(interaction_order or config["interaction_order"])])
```

CPD, TT, and TR use tensor rank. MBA uses interaction order. Logistic
regression and the MLP do not use either parameter.


## Example Commands

### One Dataset, One Model

```bash
uv run python -m src.train --dataset house_votes_84 --model cpd --accelerator cpu
```

### One Dataset, All Models, Three Seeds

```bash
uv run python -m src.run_experiments --dataset house_votes_84 --seed 1 --seed 2 --seed 3 --accelerator cpu
```

### Tensor Models With Heavier Defaults

```bash
uv run python -m src.run_experiments --dataset house_votes_84 --model cpd --model tt --model tr --rank 32 --epochs 150 --learning-rate 0.005 --accelerator cpu
```

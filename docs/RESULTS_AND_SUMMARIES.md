# Results and Summaries

This document explains how experiment logs are read and converted into summary
tables.

## Result Files

File: `src/summarize_results.py`

Lightning writes metrics to:

```text
results/<model>/<version>/metrics.csv
```

The summary script searches those files with:

```python
for path in sorted(results_dir.glob("*/*/metrics.csv")):
    model, version = path.parts[-3], path.parts[-2]
```

Only known model folders are included:

```python
MODELS = ["lr", "mlp", "cpd", "mba", "tt", "tr"]
```


## Result Versions

### Seeded Version Names

```python
SEEDED_VERSION_PATTERN = re.compile(r"(?P<dataset>.+)_seed(?P<seed>-?\d+)$")
```

Seeded runs are named like:

```text
house_votes_84_seed42
```

The parser extracts:

```text
dataset = house_votes_84
seed = 42
```

If the folder name does not match the seeded pattern, the whole folder name is
treated as the dataset name and the seed is stored as `None`.


## Reading Test Metrics

### Final Test Row

```python
metrics = pd.read_csv(path)
test_rows = metrics[metrics["test_acc"].notna()]
final_test = test_rows.iloc[-1]
```

Lightning logs many rows during training. Test metrics are identified by rows
where `test_acc` is present. The final such row is used as the test result.

### Stored Metrics

```python
rows.append(
    {
        "dataset": dataset,
        "seed": seed,
        "model": model,
        "test_acc": float(final_test["test_acc"]),
        "test_loss": float(final_test["test_loss"]),
        "test_balanced_acc": (...),
    }
)
```

The summary keeps ordinary test accuracy, test loss, and balanced accuracy when
balanced accuracy is available.

Accuracy is:

```text
accuracy = correct predictions / number of examples.
```

Balanced accuracy is the mean recall over present classes:

```text
balanced_accuracy = (1 / C) sum_c recall_c
recall_c = true positives for class c / examples with class c.
```

Balanced accuracy is useful when the majority class is much larger than the
minority classes.


## Majority-Class Baseline

### Reading Test Targets

```python
test_path = seed_dir / f"{run.dataset}_tensor_test.csv"
if not test_path.exists():
    test_path = seed_dir / f"{run.dataset}_baseline_test.csv"
```

The majority baseline is computed from the processed test split. The tensor
test file is preferred because all tensor models use it, but the baseline file
is used as a fallback.

### Majority Accuracy

```python
targets = test_df["target"]
targets.value_counts(normalize=True).iloc[0]
```

The majority-class baseline is:

```text
majority_acc = max_c count(y = c) / N.
```

This is the accuracy of a classifier that always predicts the most frequent
test-set class. A model is only useful if it beats this baseline consistently.


## Building the Summary Table

### Accuracy Pivot

```python
accuracy_table = metrics.pivot(
    index=["dataset", "seed"],
    columns="model",
    values="test_acc",
).reindex(columns=MODELS)
```

The long metrics table is converted into one row per dataset and seed:

```text
dataset | seed | lr | mlp | cpd | mba | tt | tr
```

Each model column contains test accuracy.

### Balanced Accuracy Pivot

```python
balanced_accuracy_table = metrics.pivot(
    index=["dataset", "seed"],
    columns="model",
    values="test_balanced_acc",
).reindex(columns=MODELS)
balanced_accuracy_table = balanced_accuracy_table.add_suffix("_balanced_acc")
```

Balanced accuracy is added in separate columns:

```text
lr_balanced_acc
mlp_balanced_acc
cpd_balanced_acc
...
```

### Best Model and Improvement

```python
summary["best_model"] = accuracy_table.idxmax(axis=1)
summary["best_acc"] = accuracy_table.max(axis=1)
summary["best_minus_majority"] = summary["best_acc"] - summary["majority_acc"]
```

For each dataset and seed, the summary records the best model and how far it is
above the majority-class baseline:

```text
best_minus_majority = best_acc - majority_acc.
```

Each individual model also gets the same comparison:

```python
summary[f"{model}_minus_majority"] = summary[model] - summary["majority_acc"]
```


## Aggregating Across Seeds

### Mean and Standard Deviation

```python
grouped = summary.groupby("dataset", dropna=False)
means = grouped[metric_columns].mean().add_suffix("_mean")
stds = grouped[metric_columns].std(ddof=0).add_suffix("_std")
counts = grouped.size().rename("n_runs")
```

When multiple seeds are present, the aggregate table reports:

```text
mean metric over seeds
standard deviation over seeds
number of runs
```

The standard deviation is computed with `ddof=0`, so it is the population
standard deviation over the completed runs.


## Command-Line Interface

### Options

```python
@click.option("--results-dir", default=Path("results"))
@click.option("--processed-dir", default=Path("data/processed"))
@click.option("--output", default=Path("results/benchmark_summary.csv"))
@click.option("--aggregate-output", default=Path("results/benchmark_summary_aggregate.csv"))
```

The normal command is:

```bash
uv run python -m src.summarize_results
```

For seeded experiment folders, use:

```bash
uv run python -m src.summarize_results --processed-dir data/processed
```

The script writes:

```text
results/benchmark_summary.csv
results/benchmark_summary_aggregate.csv
```


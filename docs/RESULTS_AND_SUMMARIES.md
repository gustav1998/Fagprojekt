# Results and Summaries

This document explains how experiment logs are read and converted into summary
tables.

## Result Files

File: `src/summarize_results.py`

Lightning writes metrics to:

```text
src/summary_results/results/<model>/<version>/metrics.csv
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
        "test_macro_precision": (...),
        "test_macro_recall": (...),
        "test_macro_f1": (...),
        "test_weighted_f1": (...),
    }
)
```

The summary keeps ordinary test accuracy, test loss, balanced accuracy, macro
precision, macro recall, macro F1, weighted F1, parameter count, and timings
when those values are available.

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

Macro precision is the unweighted mean precision over present classes:

```text
macro_precision = (1 / C) sum_c precision_c
precision_c = true positives for class c / predictions of class c.
```

Macro recall is equal to balanced accuracy:

```text
macro_recall = balanced_accuracy.
```

Macro F1 computes F1 for each class and then averages:

```text
F1_c = 2 precision_c recall_c / (precision_c + recall_c)
macro_F1 = (1 / C) sum_c F1_c.
```

Weighted F1 averages class F1 scores using class frequencies:

```text
weighted_F1 = sum_c (support_c / N) F1_c.
```

Macro F1 is useful when all classes should matter equally. Weighted F1 is useful
when the score should reflect the actual class distribution.

Each test run also writes a confusion matrix:

```text
src/summary_results/results/<model>/<version>/test_confusion_matrix.csv
```

Rows are true classes and columns are predicted classes.


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

### Extra Metric Pivots

```python
metric_table = metrics.pivot(
    index=["dataset", "seed"],
    columns="model",
    values=metric,
).reindex(columns=MODELS)
summary = summary.join(metric_table.add_suffix(f"_{suffix}"))
```

Additional metrics are added in separate columns:

```text
lr_balanced_acc
mlp_balanced_acc
cpd_balanced_acc
lr_macro_f1
mlp_macro_f1
cpd_macro_f1
lr_weighted_f1
...
```

Run metadata is also added when available:

```text
lr_parameters
lr_fit_seconds
lr_test_seconds
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
@click.option("--results-dir", default=Path("src/summary_results/results"))
@click.option("--processed-dir", default=Path("src/data_pipeline/data/processed"))
@click.option("--output", default=Path("src/summary_results/results/benchmark_summary.csv"))
@click.option("--aggregate-output", default=Path("src/summary_results/results/benchmark_summary_aggregate.csv"))
```

The normal command is:

```bash
uv run python -m src.summary_results.summarize_results
```

For seeded experiment folders, use:

```bash
uv run python -m src.summary_results.summarize_results --processed-dir src/data_pipeline/data/processed
```

The script writes:

```text
src/summary_results/results/benchmark_summary.csv
src/summary_results/results/benchmark_summary_aggregate.csv
```

## Result Plots

File: `src/summary_results/plot_results.py`

After creating `benchmark_summary.csv`, generate the standard result plots:

```bash
uv run python -m src.summary_results.plot_results
```

The script writes SVG files to:

```text
src/summary_results/results/plots/
```

The generated plots are:

```text
macro_f1_by_dataset.svg
balanced_accuracy_by_dataset.svg
accuracy_minus_majority.svg
fit_time_by_model.svg
```

The first three plots compare models across datasets. Values are averaged across
available seeds before plotting. `accuracy_minus_majority.svg` shows whether
each model beats the majority-class baseline. The fit-time plot shows the mean
training time per model across the available runs.

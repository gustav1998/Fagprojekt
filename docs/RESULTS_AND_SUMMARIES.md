# Results and Summaries

The active analysis script is:

```text
src/summary_results/analyze_results.py
```

It reads training logs from:

```text
src/summary_results/results/
```

and writes report-ready tables and plots to:

```text
results/
```

Both folders are ignored by Git because they contain generated outputs.

## Dataset and Model Plots

Plot all models' accuracy and loss for one dataset:

```bash
uv run python -m src.summary_results.analyze_results dataset \
  --dataset house_votes_84
```

## Tuning Curves

Plot all grid-search combinations for one dataset/model:

```bash
uv run python -m src.summary_results.analyze_results tuning \
  --dataset house_votes_84 \
  --model cpd
```

The default tuning metric is `val_loss`. It can be changed with:

```bash
--tuning-metric val_macro_f1
```

## Accuracy and Loss Tables

Create dataset/model tables and overall model averages:

```bash
uv run python -m src.summary_results.analyze_results tables
```

Outputs:

```text
results/accuracy_loss_by_dataset_model.csv
results/accuracy_loss_overall_by_model.csv
```

## Timing Tables

Create timing summaries:

```bash
uv run python -m src.summary_results.analyze_results timing
```

Outputs:

```text
results/timing_by_dataset_model.csv
results/timing_overall_by_model.csv
```

## Statistical Tests

Run Friedman and Nemenyi tests across models:

```bash
uv run python -m src.summary_results.analyze_results stats \
  --stats-metric test_acc
```

For metrics where lower is better:

```bash
uv run python -m src.summary_results.analyze_results stats \
  --stats-metric test_loss \
  --lower-is-better
```

Outputs:

```text
results/friedman_<metric>.csv
results/average_ranks_<metric>.csv
results/nemenyi_p_values_<metric>.csv
results/statistical_test_matrix_<metric>.csv
```

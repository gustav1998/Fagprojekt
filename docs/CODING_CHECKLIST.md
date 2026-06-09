# Coding Project Checklist

This is a living checklist for the coding part of the project. Add new goals as
they appear, mark completed items with `[x]`, and keep short notes beside items
when a decision needs to be remembered.

## How to Use This Checklist

- Use `[ ]` for open tasks and `[x]` for completed tasks.
- Add new items under the closest existing section.
- Add unresolved questions under `Open Questions`.
- Prefer small, testable tasks over broad goals.
- Commit meaningful groups of changes on a feature branch.

## Current Status Summary

- [x] Core model implementations exist for LR, MLP, CPD, MBA, TT, and TR.
- [x] Main dataset preprocessing pipeline is implemented.
- [x] Extended metrics are logged: accuracy, balanced accuracy, macro/weighted
  F1, confusion matrices, parameter counts, and timings.
- [x] Early stopping and best validation checkpoint testing are implemented.
- [x] Optional Bayesian hyperparameter tuning is implemented separately from
  normal training.
- [ ] Final experiment protocol still needs to be fixed and run.
- [ ] Final result tables/plots still need to be generated from final runs.

## Git and Workflow

- [x] Keep generated `results/` out of git.
- [x] Keep generated seed-specific processed data out of git.
- [x] Use feature branches for new work instead of committing directly to
  `main`.
- [ ] Decide when the current local branches should be pushed.
- [ ] Keep commits grouped by topic: data, models, metrics, docs, experiments.
- [ ] Before pushing, run a clean `git status` and verify no generated outputs
  are staged.

## Data Pipeline

- [x] Load supported raw datasets from local files.
- [x] Support both baseline and tensor representations.
- [x] Split raw data before fitting preprocessing transformations.
- [x] Fit category mappings, numeric bins, and fill values on training data
  only.
- [x] Handle official MONK train/test splits.
- [x] Drop missing targets before splitting.
- [x] Remove rare target classes that cannot be split safely.
- [ ] Run a final data audit for all datasets used in the report.
- [ ] Confirm each final dataset has the expected number of rows, classes, and
  feature cardinalities.
- [ ] Document any dataset-specific exclusions or caveats in the report.

## Models

- [x] Logistic regression baseline implemented.
- [x] MLP baseline implemented.
- [x] CPD tensor classifier implemented.
- [x] MBA classifier implemented.
- [x] TT classifier implemented.
- [x] TR classifier implemented.
- [ ] Decide whether the planned random forest baseline is still required.
- [ ] If random forest is required, implement it or document why it was omitted.
- [ ] Confirm all model default hyperparameters are defensible.
- [ ] Check whether MLP architecture changes are fully reflected in docs.

## Training and Evaluation

- [x] Single-model training command works.
- [x] Multi-dataset/multi-model experiment runner works.
- [x] Early stopping is enabled by default.
- [x] Best validation checkpoint is tested instead of final epoch.
- [x] Optional hyperparameter tuning script exists.
- [ ] Decide final monitor metric: likely `val_loss` or `val_macro_f1`.
- [ ] Decide final seeds, probably `1` through `5`.
- [ ] Decide final hyperparameter strategy:
  - fixed current defaults,
  - tuned defaults from representative datasets,
  - or model-specific heavier defaults.
- [ ] Run final full benchmark with the chosen protocol.
- [ ] Summarize final results using `src.summarize_results`.
- [ ] Verify final summaries include all expected metrics.

## Results and Plots

- [x] Benchmark summary tables can be generated.
- [x] Majority-class baseline and lift over majority are computed.
- [x] Confusion matrices are exported per run.
- [ ] Create final plots for the report:
  - mean accuracy by model,
  - mean balanced accuracy by model,
  - macro F1 by model,
  - lift over majority by model,
  - model win counts,
  - dataset-by-model heatmap.
- [ ] Identify datasets where models collapse to majority class.
- [ ] Identify datasets where heavy tensor models overfit.
- [ ] Decide which plots belong in the report and which belong in appendices.

## Documentation

- [x] Model descriptions exist.
- [x] Data pipeline documentation exists.
- [x] Training and experiment documentation exists.
- [x] Results and summaries documentation exists.
- [x] AI usage documentation exists.
- [x] Coding checklist exists.
- [ ] Update docs after final experiment protocol is chosen.
- [ ] Add final command examples for the exact benchmark used in the report.
- [ ] Make sure docs and README agree on defaults, metrics, and model list.
- [ ] Keep report-facing explanations separate from temporary experiment notes.

## Code Quality

- [x] Main source files compile.
- [x] Recently changed files pass flake8.
- [ ] Decide whether to clean existing unrelated flake8 warnings.
- [ ] Add focused automated tests for:
  - preprocessing without train/test leakage,
  - metric calculations,
  - confusion matrix shape,
  - result summarization.
- [ ] Run smoke training for each model after any model or data change.
- [ ] Confirm the project works after a fresh `uv sync`.

## Final Readiness

- [ ] All final datasets are available and documented.
- [ ] Final experiment protocol is fixed before test result interpretation.
- [ ] Final benchmark has been run across all chosen seeds.
- [ ] Final metrics and plots have been generated.
- [ ] Results have been checked for obvious failures or collapsed models.
- [ ] Code documentation matches the implemented code.
- [ ] AI usage documentation is up to date.
- [ ] Repo has no unwanted generated files staged.
- [ ] Final branch is pushed and ready for review/merge.

## Open Questions

- [ ] Is the random forest baseline still required by the report/project plan?
- [ ] Should final model selection use `val_loss`, `val_acc`, or
  `val_macro_f1`?
- [ ] How many tuning trials are acceptable before the final benchmark?
- [ ] Which datasets should be used for hyperparameter tuning?
- [ ] Should final results use tuned defaults or current fixed defaults?

## New Goals

Add new tasks here first if it is not clear where they belong.

- [ ] Add new goal here.

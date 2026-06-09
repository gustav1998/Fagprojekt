# Use of AI Assistance

This document records how AI tools were used during the software part of the
project. It is intended as a living document that can be updated when new
AI-assisted work is added.

## Summary Statement

AI assistance was used as a programming and project-organization aid. It helped
inspect the repository, suggest code simplifications, identify possible data
pipeline issues, draft documentation, and generate temporary scripts for running
and comparing experiments.

The AI tool was not treated as an authority on the mathematical theory, project
scope, or final conclusions. Suggested changes were reviewed, tested, and
adapted by the project group before being accepted.

## How AI Was Used

AI assistance was used for the following types of work:

- Reviewing repository structure and suggesting simpler organization.
- Refactoring code to make naming, comments, and module boundaries more
  consistent.
- Drafting explanations of implemented models, training code, data processing,
  and result summaries.
- Helping identify data handling issues such as train/test leakage, missing
  target values, rare target classes, and official train/test splits.
- Generating temporary experiment scripts to run model comparisons.
- Generating result comparison tables and plots from logged metrics.
- Helping interpret experiment outputs, especially majority-class collapse and
  lift over majority baseline.

## How AI Was Not Used

AI assistance was not used as a replacement for:

- Choosing the mathematical model family studied in the project.
- Deciding which theoretical claims are correct.
- Making final interpretations without checking experimental evidence.
- Accepting code changes without running tests or smoke experiments.
- Writing final project conclusions without project group review.

## Verification

AI-assisted code changes were checked through a combination of:

- Reading the surrounding code before applying changes.
- Running formatting or linting checks when relevant.
- Running `compileall` or import checks after structural changes.
- Running smoke training jobs on selected datasets.
- Running larger experiment batches when model behavior needed validation.
- Comparing model accuracy against majority-class baselines.
- Inspecting generated plots and summary CSV files.

## Temporary Scripts

Some AI-assisted experiment work used temporary scripts under `/private/tmp`.
These scripts were used only to run one-off comparisons or generate plots, then
removed after use. The permanent outputs from those experiments are the metric
CSV files and plots under `results/`, which are ignored by git.

## Update Log

Use this table to keep track of future AI-assisted work.

| Date | Area | AI-assisted activity | Verification |
| --- | --- | --- | --- |
| 2026-06-09 | Repository documentation | Created this AI usage document. | Reviewed wording and linked from docs index. |
| 2026-06-08 | Dataset pipeline | Helped refactor preprocessing so splitting happens before train-fitted transformations. | Ran data invariant checks, compile checks, and smoke training jobs. |
| 2026-06-08 | Experiments | Helped run and compare default and heavier model configurations. | Compared all 162 runs against majority baselines and generated result plots. |
| 2026-06-08 | Model documentation | Helped draft model/code explanation documents. | Checked against implemented code and formulas. |

## Suggested Report Text

The following text can be adapted for the final report:

> AI tools were used as programming assistants during the project. They supported
> repository inspection, code refactoring, documentation drafting, experiment
> scripting, and result visualization. All AI-generated suggestions were reviewed
> by the project group and validated through code inspection, tests, and
> experiments before being accepted. The mathematical framing, final model
> choices, and interpretation of results remained the responsibility of the
> project group.

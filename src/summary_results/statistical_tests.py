"""
Statistical tests for hyperparameter sensitivity (RQ2) and model comparison (RQ3).

RQ2:
  - CPD, TT, TR: Friedman test across ranks {4, 8, 16, 32}, Nemenyi post-hoc if significant
  - MBA: Wilcoxon signed-rank test comparing interaction order 1 vs 2

RQ3:
  - Friedman test across all 7 models on test balanced accuracy, Nemenyi post-hoc if significant
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, wilcoxon, t

RESULTS_DIR = Path("src/summary_results/results")
AGGREGATE_CSV = RESULTS_DIR / "benchmark_summary_aggregate.csv"
ALPHA = 0.05


def load_tuning_data() -> pd.DataFrame:
    rows = []
    for model in ["cpd", "tt", "tr", "mba"]:
        for combo_dir in sorted((RESULTS_DIR / model).glob("tuning_*_combo*")):
            meta = combo_dir / "run_metadata.json"
            if not meta.exists():
                continue
            m = json.loads(meta.read_text(encoding="utf-8"))
            hyperparam = m.get("interaction_order") if model == "mba" else m.get("rank")
            rows.append({
                "model": model,
                "dataset": m["dataset"],
                "hyperparam": hyperparam,
                "learning_rate": m["learning_rate"],
                "val_loss": m["best_model_score"],
            })
    df = pd.DataFrame(rows)
    return df.groupby(["model", "dataset", "hyperparam"])["val_loss"].min().reset_index()


def _ci95(values: np.ndarray) -> tuple[float, float]:
    n = len(values)
    mean = values.mean()
    se = values.std(ddof=1) / np.sqrt(n)
    margin = t.ppf(0.975, df=n - 1) * se
    return mean - margin, mean + margin


def run_friedman_nemenyi(matrix: pd.DataFrame, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"Friedman test: {label}")
    print(f"{'='*60}")
    print(f"  Datasets (blocks): {len(matrix)}")
    print(f"  Conditions: {list(matrix.columns)}")

    print("\n  Descriptive statistics (mean ± std, 95% CI):")
    for col in matrix.columns:
        vals = matrix[col].values
        lo, hi = _ci95(vals)
        print(f"    {col}: mean={vals.mean():.4f}, std={vals.std(ddof=1):.4f}, 95% CI=[{lo:.4f}, {hi:.4f}]")

    stat, p = friedmanchisquare(*[matrix[col].values for col in matrix.columns])
    print(f"  Friedman statistic: {stat:.4f}, p-value: {p:.4f}")

    if p < ALPHA:
        print(f"  Result: SIGNIFICANT (p < {ALPHA}) — proceeding to Nemenyi post-hoc")
        nemenyi = sp.posthoc_nemenyi_friedman(matrix.values)
        nemenyi.index = matrix.columns
        nemenyi.columns = matrix.columns
        print("\n  Nemenyi p-values:")
        print(nemenyi.round(4).to_string())
        print(f"\n  Significant pairs (p < {ALPHA}):")
        found = False
        cols = list(matrix.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                p_val = nemenyi.iloc[i, j]
                if p_val < ALPHA:
                    print(f"    {cols[i]} vs {cols[j]}: p = {p_val:.4f}")
                    found = True
        if not found:
            print("    None")
    else:
        print(f"  Result: NOT significant (p = {p:.4f}) — no post-hoc test needed")


def run_wilcoxon(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"Wilcoxon signed-rank test: {label}")
    print(f"{'='*60}")
    stat, p = wilcoxon(a, b)
    direction = label_a if np.median(a - b) < 0 else label_b

    for vals, lbl in [(a, label_a), (b, label_b)]:
        lo, hi = _ci95(vals)
        print(f"  {lbl}: mean={vals.mean():.4f}, std={vals.std(ddof=1):.4f}, median={np.median(vals):.4f}, 95% CI=[{lo:.4f}, {hi:.4f}]")
    print(f"  Statistic: {stat:.4f}, p-value: {p:.4f}")
    if p < ALPHA:
        print(f"  Result: SIGNIFICANT (p < {ALPHA}) — {direction} has lower loss")
    else:
        print(f"  Result: NOT significant (p = {p:.4f})")


def rq2_rank_sensitivity(tuning_df: pd.DataFrame) -> None:
    print("\n" + "#"*60)
    print("RQ2: HYPERPARAMETER SENSITIVITY")
    print("#"*60)

    for model in ["cpd", "tt", "tr"]:
        sub = tuning_df[tuning_df["model"] == model].copy()
        matrix = sub.pivot(index="dataset", columns="hyperparam", values="val_loss").dropna()
        matrix.columns = [f"rank={int(c)}" for c in matrix.columns]
        run_friedman_nemenyi(matrix, f"{model.upper()} rank sensitivity")

    # MBA: Wilcoxon
    mba = tuning_df[tuning_df["model"] == "mba"].copy()
    mba_pivot = mba.pivot(index="dataset", columns="hyperparam", values="val_loss").dropna()
    if 1.0 in mba_pivot.columns and 2.0 in mba_pivot.columns:
        run_wilcoxon(
            mba_pivot[1.0].values,
            mba_pivot[2.0].values,
            "order=1", "order=2",
            "MBA interaction order sensitivity",
        )


def rq3_model_comparison() -> None:
    print("\n" + "#"*60)
    print("RQ3: MODEL COMPARISON")
    print("#"*60)

    df = pd.read_csv(AGGREGATE_CSV)
    models = ["lr", "mlp", "cpd", "mba", "tt", "tr", "rf"]
    cols = {m: f"{m}_balanced_acc_mean" for m in models}
    available = {m: c for m, c in cols.items() if c in df.columns}

    matrix = df[list(available.values())].copy()
    matrix.columns = [m.upper() for m in available.keys()]
    matrix = matrix.dropna()
    matrix.index = df["dataset"]

    run_friedman_nemenyi(matrix, "All models — balanced accuracy")


if __name__ == "__main__":
    tuning_df = load_tuning_data()
    rq2_rank_sensitivity(tuning_df)
    rq3_model_comparison()

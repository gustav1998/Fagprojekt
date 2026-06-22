from __future__ import annotations

from pathlib import Path

import click
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


MODELS = ["lr", "mlp", "cpd", "mba", "tt", "tr", "rf"]
MODEL_LABELS = {
    "lr": "LR",
    "mlp": "MLP",
    "cpd": "CPD",
    "mba": "MBA",
    "tt": "TT",
    "tr": "TR",
    "rf": "RF",
}


def load_tuning_sensitivity(results_dir: Path) -> pd.DataFrame:
    rows = []
    for model in ["cpd", "tt", "tr", "mba"]:
        for combo_dir in sorted((results_dir / model).glob("tuning_*_combo*")):
            meta = combo_dir / "run_metadata.json"
            if not meta.exists():
                continue
            import json
            m = json.loads(meta.read_text(encoding="utf-8"))
            hyperparam = m.get("interaction_order") if model == "mba" else m.get("rank")
            rows.append({
                "model": model,
                "dataset": m["dataset"],
                "hyperparam": hyperparam,
                "learning_rate": m["learning_rate"],
                "val_loss": m["best_model_score"],
            })
    if not rows:
        return pd.DataFrame(columns=["model", "dataset", "hyperparam", "val_loss"])
    df = pd.DataFrame(rows)
    return df.groupby(["model", "dataset", "hyperparam"])["val_loss"].min().reset_index()


def draw_sensitivity_plot(tuning_df: pd.DataFrame, output_path: Path) -> None:
    if tuning_df.empty:
        print("  Skipping sensitivity plot — no tuning data found")
        return
    colors = {"cpd": "#1d4ed8", "tt": "#d97706", "tr": "#16a34a"}
    rank_models = ["cpd", "tr", "tt"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 3.2), gridspec_kw={"width_ratios": [3, 1]})

    # left: rank sensitivity for CPD, TT, TR
    for model in rank_models:
        sub = tuning_df[tuning_df["model"] == model]
        grouped = sub.groupby("hyperparam")["val_loss"]
        means = grouped.mean()
        stds = grouped.std()
        x = means.index.astype(int)
        ax1.plot(x, means.values, marker="o", label=MODEL_LABELS[model], color=colors[model], linewidth=1.8)

    ax1.set_xlabel("Rank", fontsize=9)
    ax1.set_ylabel("Mean validation loss", fontsize=9)
    ax1.set_title("Rank sensitivity — CPD, TT, TR", fontsize=9, fontweight="bold", loc="left")
    ax1.set_xticks([4, 8, 16, 32])
    ax1.legend(fontsize=8, frameon=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(labelsize=8)
    ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    # right: order sensitivity for MBA
    mba = tuning_df[tuning_df["model"] == "mba"]
    grouped = mba.groupby("hyperparam")["val_loss"]
    means = grouped.mean()
    stds = grouped.std()
    x = means.index.astype(int)
    ax2.plot(x, means.values, marker="o", color="#9333ea", linewidth=1.8)
    ax2.set_xlabel("Interaction order", fontsize=9)
    ax2.set_title("MBA", fontsize=9, fontweight="bold", loc="left")
    ax2.set_xticks([1, 2])
    ax2.set_ylim(ax1.get_ylim())
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(labelsize=8)
    ax2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def load_aggregate(aggregate_path: Path) -> pd.DataFrame:
    df = pd.read_csv(aggregate_path)
    rename = {}
    for col in df.columns:
        if col.endswith("_mean"):
            rename[col] = col[: -len("_mean")]
    return df.rename(columns=rename)


def draw_heatmap(
    data: pd.DataFrame,
    *,
    value_suffix: str,
    title: str,
    output_path: Path,
) -> None:
    models = [
        m for m in MODELS
        if (m if value_suffix == "accuracy" else f"{m}_{value_suffix}") in data.columns
        and data[(m if value_suffix == "accuracy" else f"{m}_{value_suffix}")].notna().any()
    ]
    if not models:
        return

    columns = [
        m if value_suffix == "accuracy" else f"{m}_{value_suffix}"
        for m in models
    ]
    labels = [MODEL_LABELS[m] for m in models]
    datasets = data["dataset"].astype(str).tolist()

    matrix = data[columns].to_numpy(dtype=float)

    n_rows, n_cols = len(datasets), len(models)

    # Design at actual output size so LaTeX doesn't scale it down.
    # A4 text width with standard margins ≈ 6.3 inches.
    fig_w = 6.3
    row_h = 0.27
    fig_h = n_rows * row_h + 1.2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.get_cmap("Blues")
    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(datasets, fontsize=8)

    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # annotate cells; bold the best value in each row
    threshold = vmin + (vmax - vmin) * 0.65
    for row in range(n_rows):
        best_col = int(np.nanargmax(matrix[row]))
        for col in range(n_cols):
            val = matrix[row, col]
            if np.isnan(val):
                continue
            color = "white" if val > threshold else "black"
            weight = "bold" if col == best_col else "normal"
            ax.text(
                col, row, f"{val:.2f}",
                ha="center", va="center",
                fontsize=7.5, color=color, fontweight=weight,
            )

    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.text(0.02, 0.97, title, fontsize=11, fontweight="bold", va="top")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def write_cpd_vs_mba_table(data: pd.DataFrame, *, metric: str, output_path: Path) -> None:
    cpd_col = f"cpd_{metric}"
    mba_col = f"mba_{metric}"
    if cpd_col not in data.columns or mba_col not in data.columns:
        return

    cpd_wins = 0
    mba_wins = 0

    lines = []
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & CPD & MBA \\")
    lines.append(r"\midrule")

    for _, row in data.iterrows():
        dataset = str(row["dataset"]).replace("_", r"\_")
        cpd = row[cpd_col]
        mba = row[mba_col]
        if pd.isna(cpd) or pd.isna(mba):
            continue
        cpd_str = f"{cpd:.3f}"
        mba_str = f"{mba:.3f}"
        if cpd > mba:
            cpd_str = rf"\textbf{{{cpd_str}}}"
            cpd_wins += 1
        elif mba > cpd:
            mba_str = rf"\textbf{{{mba_str}}}"
            mba_wins += 1
        lines.append(rf"{dataset} & {cpd_str} & {mba_str} \\")

    lines.append(r"\midrule")
    lines.append(rf"\textbf{{Wins}} & \textbf{{{cpd_wins}}} & \textbf{{{mba_wins}}} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved {output_path}")


def _bold_higher(a_str, b_str, a_val, b_val, a_wins, b_wins, higher_is_better=True):
    if a_str == b_str:
        return a_str, b_str, a_wins, b_wins
    if higher_is_better:
        if a_val > b_val:
            return rf"\textbf{{{a_str}}}", b_str, a_wins + 1, b_wins
        else:
            return a_str, rf"\textbf{{{b_str}}}", a_wins, b_wins + 1
    else:
        if a_val < b_val:
            return rf"\textbf{{{a_str}}}", b_str, a_wins + 1, b_wins
        else:
            return a_str, rf"\textbf{{{b_str}}}", a_wins, b_wins + 1


def write_cpd_vs_mba_appendix_table(data: pd.DataFrame, output_path: Path) -> None:
    needed = ["cpd_balanced_acc", "mba_balanced_acc", "cpd_macro_f1", "mba_macro_f1", "cpd_loss", "mba_loss"]
    if not all(c in data.columns for c in needed):
        return

    wins = {"cpd_acc": 0, "mba_acc": 0, "cpd_f1": 0, "mba_f1": 0, "cpd_loss": 0, "mba_loss": 0}

    lines = []
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{2}{c}{Balanced accuracy} & \multicolumn{2}{c}{Macro F1} & \multicolumn{2}{c}{Cross-entropy loss} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    lines.append(r"Dataset & CPD & MBA & CPD & MBA & CPD & MBA \\")
    lines.append(r"\midrule")

    for _, row in data.iterrows():
        dataset = str(row["dataset"]).replace("_", r"\_")
        if pd.isna(row["cpd_balanced_acc"]):
            continue

        ca, ma = f"{row['cpd_balanced_acc']:.3f}", f"{row['mba_balanced_acc']:.3f}"
        cf, mf = f"{row['cpd_macro_f1']:.3f}", f"{row['mba_macro_f1']:.3f}"
        cl, ml = f"{row['cpd_loss']:.3f}", f"{row['mba_loss']:.3f}"

        ca, ma, wins["cpd_acc"], wins["mba_acc"] = _bold_higher(ca, ma, row["cpd_balanced_acc"], row["mba_balanced_acc"], wins["cpd_acc"], wins["mba_acc"])
        cf, mf, wins["cpd_f1"], wins["mba_f1"] = _bold_higher(cf, mf, row["cpd_macro_f1"], row["mba_macro_f1"], wins["cpd_f1"], wins["mba_f1"])
        cl, ml, wins["cpd_loss"], wins["mba_loss"] = _bold_higher(cl, ml, row["cpd_loss"], row["mba_loss"], wins["cpd_loss"], wins["mba_loss"], higher_is_better=False)

        lines.append(rf"{dataset} & {ca} & {ma} & {cf} & {mf} & {cl} & {ml} \\")

    lines.append(r"\midrule")
    lines.append(
        rf"\textbf{{Wins}} & \textbf{{{wins['cpd_acc']}}} & \textbf{{{wins['mba_acc']}}} "
        rf"& \textbf{{{wins['cpd_f1']}}} & \textbf{{{wins['mba_f1']}}} "
        rf"& \textbf{{{wins['cpd_loss']}}} & \textbf{{{wins['mba_loss']}}} \\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{CPD vs MBA across all 32 datasets for balanced accuracy, macro F1, and cross-entropy loss. Bold indicates the better score per metric (higher for accuracy and F1, lower for loss). Ties are left unbolded.}")
    lines.append(r"\label{tab:cpd_vs_mba_full}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved {output_path}")


def draw_model_boxplot(data: pd.DataFrame, output_path: Path) -> None:
    models = [m for m in MODELS if f"{m}_balanced_acc" in data.columns and data[f"{m}_balanced_acc"].notna().any()]
    matrix = data[[f"{m}_balanced_acc" for m in models]].dropna().copy()
    matrix.columns = [MODEL_LABELS[m] for m in models]

    medians = matrix.median().sort_values(ascending=False)
    matrix = matrix[medians.index]

    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    ax.boxplot(
        [matrix[col].values for col in matrix.columns],
        tick_labels=matrix.columns,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor="#dbeafe", linewidth=1),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1),
        flierprops=dict(marker="o", markersize=3, linestyle="none", markerfacecolor="#999"),
    )
    ax.set_ylabel("Balanced accuracy", fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def draw_cd_diagram(data: pd.DataFrame, output_path: Path) -> None:
    import scikit_posthocs as sp

    models = [m for m in MODELS if f"{m}_balanced_acc" in data.columns and data[f"{m}_balanced_acc"].notna().any()]
    matrix = data[[f"{m}_balanced_acc" for m in models]].dropna().copy()
    matrix.columns = models
    n = len(matrix)
    k = len(models)

    rank_matrix = matrix.rank(axis=1, ascending=False, method="average")
    mean_ranks = rank_matrix.mean().sort_values()
    sorted_models = mean_ranks.index.tolist()

    nemenyi = sp.posthoc_nemenyi_friedman(matrix[sorted_models].values)
    nemenyi.index = sorted_models
    nemenyi.columns = sorted_models

    # find clique spans: for each i, extend as far right as all pairs non-sig
    raw_spans = []
    seen = set()
    for i in range(k):
        j_max = i
        for j in range(i + 1, k):
            all_ns = all(
                nemenyi.loc[sorted_models[a], sorted_models[b]] >= 0.05
                for a in range(i, j + 1)
                for b in range(a + 1, j + 1)
            )
            if all_ns:
                j_max = j
        if j_max > i:
            key = (sorted_models[i], sorted_models[j_max])
            if key not in seen:
                seen.add(key)
                raw_spans.append((mean_ranks[sorted_models[i]], mean_ranks[sorted_models[j_max]]))

    # remove spans fully contained within another span
    spans = []
    for span in raw_spans:
        subsumed = any(
            other[0] <= span[0] and other[1] >= span[1] and other != span
            for other in raw_spans
        )
        if not subsumed:
            spans.append(span)

    # critical difference (q_alpha for k=7, alpha=0.05)
    q_alpha = 2.949
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-0.55, 0.75)
    ax.axis("off")

    axis_y = 0.25
    ax.plot([1, k], [axis_y, axis_y], "k-", linewidth=1.5, zorder=2)

    for idx, model in enumerate(sorted_models):
        rank = mean_ranks[model]
        label = MODEL_LABELS[model]
        ax.plot([rank, rank], [axis_y - 0.04, axis_y + 0.04], "k-", linewidth=1, zorder=3)
        if idx % 2 == 0:
            ax.text(rank, axis_y + 0.07, label, ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.text(rank, axis_y + 0.20, f"{rank:.2f}", ha="center", va="bottom", fontsize=7, color="#666")
        else:
            ax.text(rank, axis_y - 0.07, label, ha="center", va="top", fontsize=9, fontweight="bold")
            ax.text(rank, axis_y - 0.20, f"{rank:.2f}", ha="center", va="top", fontsize=7, color="#666")

    for i, (r1, r2) in enumerate(spans):
        y = axis_y - 0.38 - i * 0.10
        ax.plot([r1, r2], [y, y], "k-", linewidth=4, solid_capstyle="butt", zorder=2)

    # CD bar top right
    cd_right = k - 0.1
    cd_left = cd_right - cd
    bar_y = axis_y + 0.58
    ax.annotate("", xy=(cd_left, bar_y), xytext=(cd_right, bar_y),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax.text((cd_left + cd_right) / 2, bar_y + 0.05, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.text(1, bar_y, "← better", ha="left", va="center", fontsize=7.5, style="italic")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def draw_fit_time_bar(data: pd.DataFrame, output_path: Path) -> None:
    rows = []
    for model in MODELS:
        col = f"{model}_fit_seconds"
        if col not in data.columns:
            continue
        val = data[col].mean(skipna=True)
        if pd.notna(val):
            rows.append((MODEL_LABELS[model], float(val)))

    if not rows:
        return

    rows.sort(key=lambda x: x[1], reverse=True)
    names = [r[0] for r in rows]
    values = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(names, values, color="#d97706")
    ax.set_xlabel("Seconds")
    ax.set_title("Mean fit time by model", fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}s", va="center", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


@click.command()
@click.option(
    "--aggregate-summary",
    type=click.Path(path_type=Path),
    default=Path("src/summary_results/results/benchmark_summary_aggregate.csv"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("src/summary_results/results/plots"),
    show_default=True,
)
def main(aggregate_summary: Path, output_dir: Path) -> None:
    """Create PDF plots from the aggregated benchmark summary."""
    if not aggregate_summary.exists():
        raise click.ClickException(
            f"Could not find {aggregate_summary}. Run summarize_results first."
        )

    data = load_aggregate(aggregate_summary)
    data = data.sort_values("dataset").reset_index(drop=True)

    tuning_df = load_tuning_sensitivity(aggregate_summary.parent)
    draw_sensitivity_plot(tuning_df, output_dir / "hyperparameter_sensitivity.pdf")

    draw_heatmap(
        data,
        value_suffix="balanced_acc",
        title="Balanced accuracy by dataset and model",
        output_path=output_dir / "balanced_accuracy_by_dataset.pdf",
    )
    draw_heatmap(
        data,
        value_suffix="macro_f1",
        title="Macro F1 by dataset and model",
        output_path=output_dir / "macro_f1_by_dataset.pdf",
    )
    draw_fit_time_bar(data, output_dir / "fit_time_by_model.pdf")
    draw_model_boxplot(data, output_dir / "model_boxplot.pdf")
    draw_cd_diagram(data, output_dir / "cd_diagram.pdf")

    tables_dir = output_dir.parent / "tables"
    write_cpd_vs_mba_table(
        data,
        metric="balanced_acc",
        output_path=tables_dir / "cpd_vs_mba_balanced_acc.tex",
    )
    write_cpd_vs_mba_appendix_table(
        data,
        output_path=tables_dir / "cpd_vs_mba_appendix.tex",
    )


if __name__ == "__main__":
    main()

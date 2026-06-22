from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

import click
import pandas as pd


MODELS = ["lr", "mlp", "cpd", "mba", "tt", "tt3", "tr", "rf"]
MODEL_LABELS = {
    "lr": "LR",
    "mlp": "MLP",
    "cpd": "CPD",
    "mba": "MBA",
    "tt": "TT",
    "tt3": "TT3",
    "tr": "TR",
    "rf": "RF",
}

WHITE = "#ffffff"
INK = "#1f2933"
MUTED = "#6b7280"
GRID = "#e5e7eb"
MISSING = "#f3f4f6"
BLUE_LIGHT = "#dbeafe"
BLUE_DARK = "#1d4ed8"
GOLD = "#d97706"
NEGATIVE = "#f8d7da"
POSITIVE = "#dbeafe"


def metric_columns(summary: pd.DataFrame, suffix: str) -> list[str]:
    columns = []
    for model in MODELS:
        column = model if suffix == "accuracy" else f"{model}_{suffix}"
        if column in summary.columns:
            columns.append(column)
    return columns


def aggregate_by_dataset(summary: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = summary.select_dtypes(include="number").columns.tolist()
    grouped = summary.groupby("dataset", dropna=False)[numeric_columns].mean()
    grouped = grouped.reset_index()

    score_columns = metric_columns(grouped, "macro_f1")
    if not score_columns:
        score_columns = metric_columns(grouped, "accuracy")
    if score_columns:
        grouped["_sort_score"] = grouped[score_columns].max(axis=1, skipna=True)
        grouped = grouped.sort_values(["_sort_score", "dataset"], ascending=[False, True])
        grouped = grouped.drop(columns=["_sort_score"])
    else:
        grouped = grouped.sort_values("dataset")

    return grouped.reset_index(drop=True)


def available_models(data: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    present = []
    column_set = set(columns)
    for model in MODELS:
        direct = model
        if direct in column_set and data[direct].notna().any():
            present.append(model)
            continue
        for column in column_set:
            if column.startswith(f"{model}_") and data[column].notna().any():
                present.append(model)
                break
    return present


def interpolate_color(value: float, min_value: float, max_value: float) -> str:
    if pd.isna(value):
        return MISSING
    if max_value <= min_value:
        fraction = 0.65
    else:
        fraction = (value - min_value) / (max_value - min_value)
    fraction = max(0.0, min(1.0, fraction))

    start = tuple(int(BLUE_LIGHT[i : i + 2], 16) for i in (1, 3, 5))
    end = tuple(int(BLUE_DARK[i : i + 2], 16) for i in (1, 3, 5))
    rgb = tuple(round(start[i] + (end[i] - start[i]) * fraction) for i in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def signed_color(value: float, min_value: float, max_value: float) -> str:
    if pd.isna(value):
        return MISSING
    if value < 0:
        return NEGATIVE
    return interpolate_color(value, 0.0, max(max_value, 0.01))


def text(
    parts: list[str],
    x: float,
    y: float,
    value: str,
    *,
    size: int = 12,
    fill: str = INK,
    anchor: str = "start",
    weight: str = "400",
) -> None:
    parts.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{escape(value)}</text>'
    )


def save_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{WHITE}"/>',
        *body,
        "</svg>",
        "",
    ]
    path.write_text("\n".join(svg), encoding="utf-8")


def draw_heatmap(
    data: pd.DataFrame,
    *,
    value_suffix: str,
    title: str,
    subtitle: str,
    output_path: Path,
    signed: bool = False,
) -> None:
    models = [
        model
        for model in MODELS
        if (model if value_suffix == "accuracy" else f"{model}_{value_suffix}") in data
        and data[(model if value_suffix == "accuracy" else f"{model}_{value_suffix}")].notna().any()
    ]
    if not models:
        return

    datasets = data["dataset"].astype(str).tolist()
    left = 190
    top = 105
    cell_width = 82
    cell_height = 30
    right = 35
    bottom = 45
    width = left + len(models) * cell_width + right
    height = top + len(datasets) * cell_height + bottom

    columns = [
        model if value_suffix == "accuracy" else f"{model}_{value_suffix}"
        for model in models
    ]
    values = data[columns].to_numpy().flatten()
    values = pd.Series(values).dropna()
    min_value = min(0.0, float(values.min())) if signed and not values.empty else 0.0
    max_value = float(values.max()) if not values.empty else 1.0

    body: list[str] = []
    text(body, 28, 36, title, size=22, weight="700")
    text(body, 28, 62, subtitle, size=12, fill=MUTED)

    for idx, model in enumerate(models):
        x = left + idx * cell_width + cell_width / 2
        text(body, x, top - 16, MODEL_LABELS[model], size=12, anchor="middle", weight="700")

    for row_idx, row in data.iterrows():
        y = top + row_idx * cell_height
        text(body, left - 12, y + 20, str(row["dataset"]), size=12, anchor="end")
        for col_idx, model in enumerate(models):
            column = model if value_suffix == "accuracy" else f"{model}_{value_suffix}"
            value = row[column]
            x = left + col_idx * cell_width
            fill = signed_color(value, min_value, max_value) if signed else interpolate_color(value, min_value, max_value)
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_width - 4}" '
                f'height="{cell_height - 4}" rx="3" fill="{fill}"/>'
            )
            label = "" if pd.isna(value) else f"{value:.2f}"
            text(
                body,
                x + (cell_width - 4) / 2,
                y + 18,
                label,
                size=11,
                anchor="middle",
                fill=WHITE if not pd.isna(value) and value > max_value * 0.65 else INK,
                weight="700" if not pd.isna(value) else "400",
            )

    save_svg(output_path, width, height, body)


def draw_fit_time_bar(data: pd.DataFrame, output_path: Path) -> None:
    rows = []
    for model in MODELS:
        column = f"{model}_fit_seconds"
        if column not in data.columns:
            continue
        value = data[column].mean(skipna=True)
        if pd.notna(value):
            rows.append((MODEL_LABELS[model], float(value)))

    if not rows:
        return

    rows.sort(key=lambda item: item[1], reverse=True)
    left = 120
    top = 105
    bar_height = 28
    gap = 12
    chart_width = 520
    width = 720
    height = top + len(rows) * (bar_height + gap) + 55
    max_value = max(value for _, value in rows)

    body: list[str] = []
    text(body, 28, 36, "Mean fit time by model", size=22, weight="700")
    text(body, 28, 62, "Seconds averaged across available dataset/seed runs", size=12, fill=MUTED)
    body.append(
        f'<line x1="{left}" y1="{top - 12}" x2="{left + chart_width}" '
        f'y2="{top - 12}" stroke="{GRID}" stroke-width="1"/>'
    )

    for idx, (label, value) in enumerate(rows):
        y = top + idx * (bar_height + gap)
        width_value = 0 if max_value == 0 else chart_width * value / max_value
        text(body, left - 14, y + 19, label, size=12, anchor="end", weight="700")
        body.append(
            f'<rect x="{left}" y="{y}" width="{width_value:.1f}" '
            f'height="{bar_height}" rx="3" fill="{GOLD}"/>'
        )
        text(body, left + width_value + 8, y + 19, f"{value:.1f}s", size=12, fill=INK)

    save_svg(output_path, width, height, body)


@click.command()
@click.option(
    "--summary",
    type=click.Path(path_type=Path),
    default=Path("src/summary_results/results/benchmark_summary.csv"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("src/summary_results/results/plots"),
    show_default=True,
)
def main(summary: Path, output_dir: Path) -> None:
    """Create SVG plots from benchmark summary CSVs."""
    if not summary.exists():
        raise click.ClickException(
            f"Could not find {summary}. Run src.summary_results.summarize_results first."
        )

    raw_summary = pd.read_csv(summary)
    if raw_summary.empty:
        raise click.ClickException(f"{summary} has no rows to plot.")

    data = aggregate_by_dataset(raw_summary)
    output_dir.mkdir(parents=True, exist_ok=True)

    draw_heatmap(
        data,
        value_suffix="macro_f1",
        title="Macro F1 by dataset and model",
        subtitle="Mean test macro F1 across available seeds; higher is better",
        output_path=output_dir / "macro_f1_by_dataset.svg",
    )
    draw_heatmap(
        data,
        value_suffix="balanced_acc",
        title="Balanced accuracy by dataset and model",
        subtitle="Mean test balanced accuracy across available seeds; higher is better",
        output_path=output_dir / "balanced_accuracy_by_dataset.svg",
    )
    draw_heatmap(
        data,
        value_suffix="minus_majority",
        title="Accuracy above majority baseline",
        subtitle="Mean test accuracy minus majority-class accuracy; values above 0 beat the baseline",
        output_path=output_dir / "accuracy_minus_majority.svg",
        signed=True,
    )
    draw_fit_time_bar(data, output_dir / "fit_time_by_model.svg")

    print(f"Saved result plots to {output_dir}")


if __name__ == "__main__":
    main()

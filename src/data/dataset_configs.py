from __future__ import annotations

from typing import Any


DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "mushroom": {
        "file_name": "agaricus-lepiota.data",
        "sep": ",",
        "header": None,
        "target_column": "class",
        "column_names": [
            "class",
            "cap-shape",
            "cap-surface",
            "cap-color",
            "bruises",
            "odor",
            "gill-attachment",
            "gill-spacing",
            "gill-size",
            "gill-color",
            "stalk-shape",
            "stalk-root",
            "stalk-surface-above-ring",
            "stalk-surface-below-ring",
            "stalk-color-above-ring",
            "stalk-color-below-ring",
            "veil-type",
            "veil-color",
            "ring-number",
            "ring-type",
            "spore-print-color",
            "population",
            "habitat",
        ],
        "categorical_columns": "all_except_target",
        "numerical_columns": [],
        "missing_tokens": ["?"],
        "task": "classification",
    },
    "dataset": {
        "filename": "dataset.data",
        "sep": ",",
        "header": None,
        "column_names": [...],
        "target": "target",
        "categorical": [...],
        "numerical": [...],
        "missing_tokens": ["?"],
    },
}
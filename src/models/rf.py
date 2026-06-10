from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


DEFAULT_RF_CONFIG = {
    "n_estimators": 300,
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_leaf": 1,
}


def parse_max_features(value: str) -> str | float:
    # accepts "sqrt", "log2", or a float string like "0.5"
    try:
        return float(value)
    except ValueError:
        return value


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "acc": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }

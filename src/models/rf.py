from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.data.load_processed import load_processed_dataset


# Grid searched over train split only
PARAM_GRID = {
    "max_depth": [None, 5, 10, 15, 20, 30], # None means nodes are expanded until all leaves are pure or contain < min_samples_split samples
    "max_features": ["sqrt", "log2", 0.5],
} # max number of features considered for splitting a node; "sqrt" means sqrt(n_features), "log2" means log2(n_features), 0.5 means 50% of features

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Random Forest with CV-tuned hyperparameters."
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the forest and CV splitter.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--result-version",
        type=str,
        default=None,
        help="Result folder name under results/rf/.",
    )
    return parser.parse_args()

# main function to train a Random Forest with CV-tuned hyperparameters, evaluate on test set, and save results and metadata to a specified directory
def main() -> None:
    args = parse_args()

    result_version = args.result_version
    if result_version is None:
        result_version = (
            f"{args.dataset}_seed{args.seed}"
            if args.seed is not None
            else args.dataset
        ) # if result_version is not provided, use dataset name and seed to create a unique result folder name

    data = load_processed_dataset(
        dataset_name=args.dataset,
        representation="baseline",
        processed_dir=args.processed_dir,
    ) # load the processed dataset for the specified dataset name and representation (baseline) from the processed directory

    train_df = data["train_df"] # get the train dataframe from the loaded data
    test_df = data["test_df"] # get the test dataframe from the loaded data
    metadata = data["metadata"] # get the metadata from the loaded data

    X_train = train_df.drop(columns=["target"]).to_numpy() # gets the training features by dropping the target column and then converting to numpy array
    y_train = train_df["target"].to_numpy() # get the training labels by selecting the target column and then converting

    X_test = test_df.drop(columns=["target"]).to_numpy() # does the same as for training features
    y_test = test_df["target"].to_numpy() # does the same as for test labels

    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.seed,
    )
    # n_splits = number of folds in the cross-validation
    # shuffle = whether to shuffle the data before splitting into batches
    # random_state = controls the randomness of the shuffling and splitting based of the given seed


    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=1,
    ) 
    # n_estimators = number of trees in the forest
    # random_state = controls the randomness of the estimator
    # n_jobs = number of jobs to run in parallel; 1 means no parallelism, -1 means using all processors (processes such as fit, predict, decision_path and apply)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=PARAM_GRID,
        cv=cv,
        scoring="balanced_accuracy",
        refit=False,
        n_jobs=-1,
    )
    # estimator = the base model to fit on different subsets of the dataset
    # param_grid = the dictionary with parameters names (string) as keys and lists of parameter settings to try as values
    # cv = as definied above, the cross-validation splitting strategy
    # scoring = the strategy to evaluate the performance of the cross-validated model on the test
    # refit = whether to refit an estimator using the best found parameters on the whole dataset; False means do not refit, we will manually fit later with the best hyperparameters
    # n_jobs = same as before

    grid_search.fit(X_train, y_train) # run the grid search on the training data

    best_params = grid_search.best_params_ # gets the best hyperparameters found by the grid search based on the specified scoring metric (balanced accuracy)

    # Refit on the full train split with the best hyperparameters
    # Val and test splits are untouched during CV
    # models in this project (train for fitting, val/CV for selection, test once)
    final_rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=-1,
        **best_params,
    )

    # times the trainnng
    fit_start = time.perf_counter()
    final_rf.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    # evaluate the best model on the test set
    test_start = time.perf_counter()
    y_pred = final_rf.predict(X_test)
    test_seconds = time.perf_counter() - test_start

    # computes various test metrics
    test_metrics = {
        "epoch": 0,
        "step": 0,
        "test_acc": accuracy_score(y_test, y_pred),
        "test_loss": None,
        "test_balanced_acc": balanced_accuracy_score(y_test, y_pred),
        "test_macro_precision": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "test_macro_recall": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "test_macro_f1": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "test_weighted_f1": f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
    }

    # sets the result directory 
    result_dir = Path("results") / "rf" / result_version
    result_dir.mkdir(parents=True, exist_ok=True)

    # saves the test metrics to a CSV file in the result directory
    pd.DataFrame([test_metrics]).to_csv(result_dir / "metrics.csv", index=False)

    # saves the best hyperparameters and other relevant metadata about the run as JSON file in the result directory
    run_metadata = {
        "dataset": args.dataset,
        "model": "rf",
        "seed": args.seed,
        "n_estimators": args.n_estimators,
        "best_max_depth": best_params["max_depth"],
        "best_max_features": best_params["max_features"],
        "cv_folds": args.cv_folds,
        "cv_scoring": "balanced_accuracy",
        "num_classes": len(metadata["target_mapping"]),
        "input_dim": X_train.shape[1],
        "trainable_parameters": None,
        "fit_seconds": fit_seconds,
        "test_seconds": test_seconds,
    }
    (result_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    # prints the location of the saved results and the run metadata to the console
    print(f"Saved results to {result_dir}")
    print(json.dumps(run_metadata, indent=2))


if __name__ == "__main__":
    main()

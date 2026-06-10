# Data Pipeline

This document explains how raw datasets become model-ready tensors.

## 1. Raw Dataset Loading

File: `src/data_pipeline/preprocessing.py`

File for getting configs (explained in `docs\CONFIGURATION_AND_UTILITIES.md`):

File: `src/data_pipeline/dataset_configs.py`

### Loading a CSV

```python
def load_raw_dataset(config: dict[str, Any], raw_dir: Path) -> pd.DataFrame:
    file_path = raw_dir / config["file_name"]

    df = pd.read_csv(
        file_path,
        sep=config.get("sep", ","),
        engine=config.get("engine"),
        header=config.get("header", "infer"),
        names=config.get("column_names"),
        na_values=config.get("missing_tokens", ["?"]),
        keep_default_na=True,
        skipinitialspace=True,
    )
```

The dataset configuration specifies where the raw file is and how Pandas should
read it. The important fields are:

```text
file_name       -> path relative to src/data_pipeline/data/raw
sep             -> column separator
header          -> whether the file has a header row
column_names    -> names to assign if the file has no header
missing_tokens  -> values treated as missing
```

The output is a Pandas DataFrame containing the raw examples.

### Dropping Unused Columns

```python
drop_columns = config.get("drop_columns", [])
if drop_columns:
    df = df.drop(columns=drop_columns)
```

Some datasets contain identifiers that should not be used as predictive
features. These columns are removed before preprocessing.


## Feature Column Resolution

### Selecting Categorical and Numerical Columns

```python
def infer_feature_columns(
    df: pd.DataFrame,
    target_column: str,
    categorical_columns: list[str] | str | None,
    numerical_columns: list[str] | None,
) -> tuple[list[str], list[str]]:
```

This function decides which columns are categorical features and which columns
are numerical features. If the configuration says:

```python
categorical_columns = "all_except_target"
```

then every non-target column is treated as categorical:

```text
categorical = columns(df) \ {target_column}
```

The function also checks that a column is not listed as both categorical and
numerical, and that all configured columns actually exist in the DataFrame.


## Missing Values

### Filling Categorical and Numerical Missing Values

```python
for col in [target_column] + categorical_columns:
    df[col] = df[col].fillna(missing_token)

for col in numerical_columns:
    median = pd.to_numeric(df[col], errors="coerce").median(skipna=True)
    if pd.isna(median):
        median = 0.0
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median)
```

Categorical missing values are turned into a real category called
`__MISSING__`. Numerical missing values are replaced by the median of the
column. If a numerical column has no valid numbers, the fallback value is `0.0`.


## Label Encoding

### Encoding One Column

```python
def encode_column(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    categories = sorted(series.astype(str).unique())
    mapping = {value: idx for idx, value in enumerate(categories)}
    encoded = series.astype(str).map(mapping)
    return encoded, mapping
```

Each category is mapped to an integer:

```text
category -> {0, 1, ..., K - 1}
```

The categories are sorted first, so the mapping is deterministic for a fixed
dataset. The same function is used for target labels and categorical feature
values.

### Encoding Categorical Features

```python
for col in categorical_columns:
    encoded_col, mapping = encode_column(df[col])
    encoded_df[col] = encoded_col.astype(int)
    mappings[col] = mapping
    cardinalities.append(len(mapping))
```

For a categorical feature `x_d`, the cardinality is:

```text
I_d = number of unique encoded values in feature d.
```

The tensor models use these cardinalities to size their factor matrices,
interaction tables, TT cores, or TR cores.


## Numerical Discretization

### Binning Numerical Columns

```python
if strategy == "quantile":
    binned, edges = pd.qcut(
        series,
        q=min(n_bins, series.nunique()),
        labels=False,
        retbins=True,
        duplicates="drop",
    )
elif strategy == "uniform":
    binned, edges = pd.cut(
        series,
        bins=min(n_bins, series.nunique()),
        labels=False,
        retbins=True,
        duplicates="drop",
    )
```

Tensor models require discrete feature values. Numerical columns are therefore
converted into integer bin indices:

```text
x_d in {0, 1, ..., I_d - 1}
```

Quantile binning tries to put a similar number of examples in each bin. Uniform
binning splits the numerical range into equally wide intervals.


## Processed Representations

### Main Processing Function

```python
def process_dataset(
    df: pd.DataFrame,
    config: dict[str, Any],
    representation: str = "baseline",
    n_bins: int | None = None,
    binning_strategy: str = "quantile",
) -> tuple[pd.DataFrame, dict[str, Any]]:
```

This function builds one processed representation from the raw DataFrame.

For the baseline representation:

```python
X_num = df[numerical_columns].copy() if numerical_columns else pd.DataFrame(index=df.index)
processed_df = pd.concat([X_cat_encoded, X_num], axis=1)
cardinalities = cat_cardinalities
```

Categorical features are integer encoded, while numerical features stay
numerical. The later data module turns the categorical part into one-hot
features for logistic regression and the MLP.

For the tensor representation:

```python
X_num_disc, numerical_bin_metadata, numerical_cardinalities = discretize_numerical_dataframe(...)
processed_df = pd.concat([X_cat_encoded, X_num_disc], axis=1)
cardinalities = cat_cardinalities + numerical_cardinalities
```

All features become integer indices. This is the representation used by CPD,
MBA, TT, and TR.

### Metadata

```python
metadata = {
    "target_column": target_column,
    "feature_names": categorical_columns + numerical_columns,
    "categorical_columns": categorical_columns,
    "numerical_columns": numerical_columns,
    "representation": representation,
    "cardinalities": cardinalities,
    "target_mapping": target_mapping,
    "feature_mappings": feature_mappings,
    "numerical_bin_metadata": numerical_bin_metadata,
}
```

The metadata preserves the information needed later by the data module and the
tensor models. In particular:

```text
target_mapping -> number and meaning of classes
cardinalities  -> feature dimensions I_1, ..., I_D
representation -> baseline or tensor
```


## Train, Validation, and Test Splits

### Split Function

```python
train_df, temp_df = split_with_optional_stratify(
    df,
    test_fraction=(1.0 - train_size),
)
relative_test_size = test_size / (val_size + test_size)
val_df, test_df = split_with_optional_stratify(
    temp_df,
    test_fraction=relative_test_size,
)
```

The default split is:

```text
train = 70%
validation = 15%
test = 15%
```

When possible, the split is stratified. That means each split tries to preserve
the class distribution:

```text
p_train(y = c) approx p_full(y = c)
p_val(y = c)   approx p_full(y = c)
p_test(y = c)  approx p_full(y = c)
```

If a dataset is too small or a class has too few examples, the code falls back
to an ordinary random split.


## Saving Processed Data

### CSV Splits

```python
paths = {
    "train": output_dir / f"{dataset_name}_{representation}_train.csv",
    "val": output_dir / f"{dataset_name}_{representation}_val.csv",
    "test": output_dir / f"{dataset_name}_{representation}_test.csv",
}
```

Each processed representation gets its own train, validation, and test CSV
files.

### Metadata JSON

```python
metadata_path = output_dir / f"{dataset_name}_{representation}_metadata.json"
json.dump(metadata, f, indent=2)
```

The metadata JSON is saved next to the processed CSV files.


## 2. Dataset Creation Command

File: `src/data_pipeline/make_dataset.py`

### CLI Options

```python
@click.option("--dataset", "dataset_name", required=True, ...)
@click.option("--representation", default="both", ...)
@click.option("--raw-dir", default=Path("src/data_pipeline/data/raw"), ...)
@click.option("--output-dir", default=Path("src/data_pipeline/data/processed"), ...)
@click.option("--seed", default=42, ...)
@click.option("--n-bins", default=None, ...)
@click.option("--binning-strategy", default="quantile", ...)
```

This command loads one raw dataset, creates either the baseline representation,
the tensor representation, or both, then writes processed splits and metadata.

The normal command shape is:

```bash
uv run python -m src.data_pipeline.make_dataset --dataset house_votes_84 --representation both
```


## 3. Loading Processed Data

File: `src/data_pipeline/load_processed.py`

### Reading Splits and Metadata

```python
train_path = processed_dir / f"{dataset_name}_{representation}_train.csv"
val_path = processed_dir / f"{dataset_name}_{representation}_val.csv"
test_path = processed_dir / f"{dataset_name}_{representation}_test.csv"
metadata_path = processed_dir / f"{dataset_name}_{representation}_metadata.json"
```

The loader reconstructs the paths from the dataset name and representation. It
returns:

```text
train_df
val_df
test_df
metadata
```

### Converting DataFrames to Tensors

```python
X = df.drop(columns=[target_column]).to_numpy(dtype="float32")
y = df[target_column].to_numpy(dtype="int64")

X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
y_tensor = torch.tensor(y, dtype=torch.long, device=device)
```

PyTorch `CrossEntropyLoss` expects targets as integer class indices, so labels
are stored as `torch.long`.


## 4. Lightning Data Module

File: `src/data_pipeline/datamodule.py`

### Representation Selection

```python
ONE_HOT_MODELS = {"lr", "mlp"}
TENSOR_MODELS = {"cpd", "mba", "tt", "tr"}
```

The data module chooses the processed representation from the model type:

```text
lr, mlp       -> baseline
cpd, mba, tt, tr -> tensor
```

### Setup

```python
data = load_processed_dataset(
    dataset_name=self.dataset_name,
    representation=representation,
    processed_dir=self.processed_dir,
)
```

The setup step loads processed splits and metadata.

For baseline models:

```python
self.X_train = one_hot_encode_features(
    X_train_raw,
    categorical_cardinalities=categorical_cardinalities,
    num_numerical_features=num_numerical_features,
)
```

The categorical part is expanded into one-hot vectors. If feature `d` has
cardinality `I_d`, then:

```text
x_d -> e_{x_d} in R^{I_d}.
```

For tensor models:

```python
self.X_train = X_train_raw.long()
```

The features stay as integer indices because tensor models perform row or core
lookups.

### DataLoaders

```python
return DataLoader(
    dataset,
    batch_size=self.batch_size,
    shuffle=True,
    num_workers=self.num_workers,
    generator=generator,
)
```

The training loader shuffles examples. If a seed is supplied, a seeded
generator is used so the shuffle order is reproducible.

Validation and test loaders do not shuffle:

```python
shuffle=False
```

This keeps evaluation deterministic.


# Configuration and Utilities

This document explains the project configuration files and small utility
modules.

## Dataset Configurations

File: `src/data/dataset_configs.py`

### Configuration Dictionary

```python
DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "balance_scale": {
        "file_name": "balance+scale/balance-scale.data",
        "sep": ",",
        "header": None,
        "target_column": "class",
        "column_names": [...],
        "categorical_columns": [],
        "numerical_columns": [...],
        "missing_tokens": ["?"],
        "task": "classification",
    },
}
```

Each dataset entry describes how to load and process one raw dataset.

The important fields are:

```text
file_name           -> raw file path under data/raw
sep                 -> separator passed to pandas.read_csv
header              -> whether the file contains a header
target_column       -> column to predict
column_names        -> names assigned to raw columns
drop_columns        -> columns removed before modeling
categorical_columns -> categorical feature columns
numerical_columns   -> numerical feature columns
missing_tokens      -> strings treated as missing values
task                -> classification
```

### All-Except-Target Shortcut

```python
"categorical_columns": "all_except_target"
```

This marks every non-target column as categorical. It is useful for datasets
where all features are discrete.

Mathematically, if the raw columns are:

```text
{x_1, ..., x_D, y}
```

then:

```text
categorical_features = {x_1, ..., x_D}.
```

### Report Datasets

```python
for _dataset_name in [
    "asia_lung",
    "cleveland",
    "conf_ad",
    ...
]:
    DATASET_CONFIGS[_dataset_name] = {
        "file_name": f"report_datasets/{_dataset_name}.csv",
        "sep": ",",
        "header": "infer",
        "target_column": "class",
        "categorical_columns": "all_except_target",
        "numerical_columns": [],
        "missing_tokens": ["?"],
        "task": "classification",
    }
```

The report datasets share the same CSV structure, so they are added with one
loop instead of repeating identical configuration blocks.


## One-Hot Encoding

File: `src/utils/encoding.py`

### Function Signature

```python
def one_hot_encode_features(
    X: torch.Tensor,
    categorical_cardinalities: list[int],
    num_numerical_features: int = 0,
) -> torch.Tensor:
```

This utility converts integer-encoded categorical features into one-hot
vectors and appends any numerical features unchanged.

### Shape Check

```python
num_categorical = len(categorical_cardinalities)
expected_total = num_categorical + num_numerical_features

if X.shape[1] != expected_total:
    raise ValueError(...)
```

The input must be ordered as:

```text
[categorical features..., numerical features...]
```

If the number of columns does not match the metadata, the function stops with a
clear error.

### Encoding Each Categorical Feature

```python
for feature_idx, cardinality in enumerate(categorical_cardinalities):
    feature_values = X[:, feature_idx].long()
    one_hot = F.one_hot(feature_values, num_classes=cardinality).float()
    parts.append(one_hot)
```

For feature value `x_d` with cardinality `I_d`, one-hot encoding produces:

```text
e_{x_d} in R^{I_d}
```

where:

```text
e_{x_d, j} = 1 if j = x_d
e_{x_d, j} = 0 otherwise.
```

### Appending Numerical Features

```python
if num_numerical_features > 0:
    numerical_part = X[:, num_categorical:].float()
    parts.append(numerical_part)
```

Numerical features are not one-hot encoded. They are appended as floating point
values.

The final shape is:

```text
B x (sum_d I_d + N)
```

where:

```text
B = batch size
I_d = cardinality of categorical feature d
N = number of numerical features
```


## Package Configuration

File: `pyproject.toml`

### Build System

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"
```

The project is packaged with setuptools.

### Project Metadata

```toml
[project]
name = "src"
version = "0.1.0"
description = "Discrete tabular classification experiments"
requires-python = ">=3.10,<3.13"
```

This defines the package name, version, project description, and supported
Python versions.

### Dependencies

```toml
dependencies = [
  "click",
  "flake8",
  "python-dotenv>=0.5.1",
  "numpy>=1.26.4,<2",
  "pandas>=2.2,<3",
  "scikit-learn>=1.3.2",
  "torch>=2.5.1",
  "lightning>=2.3.3",
]
```

The main dependencies are:

```text
pandas, numpy       -> data handling
scikit-learn        -> train/validation/test splitting
torch               -> tensor computation and neural network modules
lightning           -> training loop framework
click               -> command-line interfaces
```

### Package Discovery

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]
```

This tells setuptools to package the `src` Python package.


## Cross-Platform Setup

The recommended setup command is:

```bash
uv sync
uv run python test_environment.py
```

This works on macOS, Linux, and Windows because `uv run` executes commands
inside the project environment without requiring shell-specific activation.

Manual activation is optional. If you create the environment with:

```bash
uv venv .venv
```

activate it with the command for your shell:

```bash
# macOS/Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```cmd
:: Windows Command Prompt
.venv\Scripts\activate.bat
```

After activation, install the project and run the smoke test:

```bash
uv pip install -e .
python test_environment.py
```


## Environment Smoke Test

File: `test_environment.py`

### Python Version Check

```python
REQUIRED_PYTHON = "python3"

system_major = sys.version_info.major
...
if system_major != required_major:
    raise TypeError(...)
else:
    print(">>> Development environment passes all tests!")
```

This script checks that the active interpreter is Python 3. It is a minimal
environment smoke test, not a full model or data test.

Run it with:

```bash
uv run python test_environment.py
```

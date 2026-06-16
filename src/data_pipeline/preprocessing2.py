# %% Imports
import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split, StratifiedKFold

# %% Loads data

def load_raw_dataset(config, raw_dir):
    file_name = config.get("file_name")

    if file_name is None: # handles the case where there is predetermined splits (so seperate train/test files)
        file_name = config.get("train_file_name")

        if file_name is None: # checks if it still doesn't work
            raise ValueError("Dataset config must define file_name or train_file_name")
        
    file_path = raw_dir / file_name

    # creates a dictionary of settings that tells how to exactly parse the raw data (seperator, names to use etc.)
    read_csv_options = {
        "sep": config.get("sep", ","),
        "engine": config.get("engine"),
        "header": config.get("header", "infer"),
        "names": config.get("column_names"),
        "na_values": config.get("missing_tokens", ["?"]),
        "keep_default_na": True,
        "skipinitialspace": True,
    }

    # handles less common settings than the ones above (skiprows: fx skips the rows of comments in the start of aps_failure etc.)
    optional_keys = ["skiprows", "compression", "comment", "encoding", "low_memory"]
    for key in optional_keys:
        if key in config:
            read_csv_options[key] = config[key]

    df = pd.read_csv(file_path, **read_csv_options) # unpacks the dictionary of settings given above

    drop_columns = config.get("drop_columns", []) # checks whether any columns are explicitly listed to be dropped
    if drop_columns: # and then drops them
        df = df.drop(columns=drop_columns, errors="ignore")

    return df

def load_raw_dataset_splits(config, raw_dir):
    has_separate_files = "train_file_name" in config and "test_file_name" in config

    if not has_separate_files: # handles the case where there isn't predetermined splits from data origin
        train_df = load_raw_dataset(config=config, raw_dir=raw_dir)
        return train_df, None
    
    else: # otherwise (i.e. the origin has predetermined test and train splits)
        train_config = config.copy()
        train_config["file_name"] = config["train_file_name"]

        test_config = config.copy()
        test_config["file_name"] = config["test_file_name"]

        train_df = load_raw_dataset(config=train_config, raw_dir=raw_dir)
        test_df = load_raw_dataset(config=test_config, raw_dir=raw_dir)

        return train_df, test_df
    
# %% Pools data (combines preditermined train/test splits where there are some)

def pool_dataset_splits(raw_train_source, raw_test_source):

    if raw_test_source is None: # either a dataframe or None
        return raw_train_source
    
    pooled_df = pd.concat([raw_train_source, raw_test_source], ignore_index=True) 
    # pd.concat takes a list of dataframes and combines them into a single one
    # -> afterwards, ignore_index=True resets the row numbering, so each row gets a unique index (0, 1, etc.) instead of keeping the original indices from each dataframe
    
    return pooled_df
# %% Clean target columns (drops rows where the respective target "label" is missing i.e. "?" and rows where the target label is rare occurence (perhaps only one patient has tumor 1 out of 15 tumor classes))

def clean_targets(df, target_column):
    df = df.dropna(subset=[target_column]).copy() # drops any row where that specific column is missing

    return df.reset_index(drop=True) # since some rows where removed, then we need to fix indexing because of gaps (i.e. reset_index renumbers it while drop=true means we don't keep the old index)

def remove_rare_classes(df, target_column, min_target_count = 10):
    counts = df[target_column].value_counts() # counts how many rows belong to a unique class
    keep = counts[counts >= min_target_count].index # filters and keeps the rows/classes that meet the minimum 

    df = df[df[target_column].isin(keep)].copy() # checks for which rows where the above holds, df keeps those where it holds True

    return df.reset_index(drop=True) # same as earlier
# %% Splitting (into tunning and actual K-fold, from there splitting it into the folds)

# random_state makes the splits reproducible
def split_tuning_and_folds(df, target_column, tuning_size=0.2, n_folds=5, random_state=42):
    # splits the pooled data into the tunning portion (20%) and eval. portion (80%)
    tuning_df, evaluation_df = train_test_split(
        df,
        test_size=1 - tuning_size, # i.e. 80%
        stratify=df[target_column],
        random_state=random_state,
    )

    #  splits the tuning portion further into tuning_train and tuning val (train: trains each hyperparameter combo, val: scores each combo and helps picking the best one)
    tuning_train_df, tuning_val_df = train_test_split(
        tuning_df,
        test_size=0.25, # 75% / 25% split
        stratify=tuning_df[target_column],
        random_state=random_state,
    )

    folds = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state) # shuffles in case there is some inherent ordering (might as well)

    for train_index, test_index in skf.split(evaluation_df, evaluation_df[target_column]):
        fold_train_df = evaluation_df.iloc[train_index].reset_index(drop=True) # .iloc uses the positions to actually pull out the corresponding rows from evaluation_df
        fold_test_df = evaluation_df.iloc[test_index].reset_index(drop=True)

        folds.append((fold_train_df, fold_test_df)) # creates 5 tuples corresponding to each fold

    return tuning_train_df.reset_index(drop=True), tuning_val_df.reset_index(drop=True), folds # returns the folds and the tuning-val/-train df w. resets indices

# %% Inferering (whether feature columns are categorical or numerical while ignoring target column)

def infer_feature_columns(df, target_column, categorical_columns, numerical_columns):
    if categorical_columns == "all_except_target": # meaning every column except the target because it is not an input 
        categorical = []
        for col in df.columns:
            if col != target_column:
                categorical.append(col)
    else:
        categorical = categorical_columns or []

    if numerical_columns == "all_except_target": # the same as above but handles the numerical case
        numerical = []
        for col in df.columns:
            if col != target_column:
                numerical.append(col)
    else:
        numerical = numerical_columns or []

    # safety measure to check for config mistake
    overlap = set(categorical) & set(numerical)
    if overlap:
        raise ValueError(f"Columns cannot be both categorical and numerical: {sorted(overlap)}")
    
    # same motivation as above but checks for if every column is mentioned
    unknown = (set(categorical) | set(numerical) | {target_column}) - set(df.columns)
    if unknown:
        raise ValueError(f"Unknown columns found in config: {sorted(unknown)}")

    return categorical, numerical

# %% Missing values

def fill_missing_values(df, target_column, categorical_columns, numerical_columns, missing_token="__MISSING__"):
    df = df.copy()

    # replaces missing values with a placeholder string ""__MISSING__""
    for col in [target_column] + categorical_columns:
        df[col] = df[col].fillna(missing_token)

    for col in numerical_columns:
        median = pd.to_numeric(df[col], errors="coerce").median(skipna=True)
        if pd.isna(median):
            median = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median)

    return df.reset_index(drop=True)

# %% Mapping (turns category into a single integer index, this does it's work on Pandas data)

def build_mapping(series, extra_values=None):
    values = set(series.astype(str).unique()) # converts every value to a string (also numbers)
    values.update(extra_values or []) 

    mapping = {}
    for idx, value in enumerate(sorted(values)): # sorts all values alphabetically then numerically
        mapping[value] = idx # turns it into a single integer index (fx. 0 and 1)

    return mapping

def apply_mapping(series, mapping, unknown_token=None):
    values = series.astype(str) # same a before

    if unknown_token is not None:
        values = values.where(values.isin(mapping), unknown_token) # keeps the value if it's a known key in mapping, otherwise it replaces it with unknown_token

    encoded = values.map(mapping) # replaces each string with its integer from the mapping dictionary from earlier

    if encoded.isna().any():
        unknown_values = sorted(values[encoded.isna()].unique())
        raise ValueError(f"Values not found in fitted mapping: {unknown_values}")

    return encoded.astype(int) # converts the column from floats to clean intergers

# %% Numerical bins (turns continuous numbers into bins/categories for CPD/MBA etc.)

def fit_numerical_bins(df, numerical_columns, n_bins, strategy):
    bin_metadata = {}
    cardinalities = []

    for col in numerical_columns:
        series = pd.to_numeric(df[col], errors="coerce")
        unique_count = int(series.nunique()) # count how many distinct values each numerical column has

        if unique_count <= 1: # if so then it just creates 1 bin
            edges = [float(series.min()), float(series.max())]
            cardinality = 1
        elif strategy == "quantile": # divides the data so each bin has roughly the same number of rows
            _, edges_array = pd.qcut(series, q=min(n_bins, unique_count), labels=False, retbins=True, duplicates="drop")
            edges = []
            for x in edges_array:
                edges.append(float(x))
            cardinality = max(1, len(edges) - 1)
        elif strategy == "uniform": # divides the value range into equal-width intervals instead
            _, edges_array = pd.cut(series, bins=min(n_bins, unique_count), labels=False, retbins=True, duplicates="drop")
            edges = [] # list of boundary values between bins
            for x in edges_array:
                edges.append(float(x))
            cardinality = max(1, len(edges) - 1) # number of bins is always one less than the number of edges
        else:
            raise ValueError(f"Unsupported binning strategy: {strategy}")

        # all of the above bin_metadata per column
        bin_metadata[col] = {
            "strategy": strategy,
            "edges": edges,
            "n_bins": cardinality,
        }
        cardinalities.append(cardinality) # creates a list of cardinalities that we will need for later

    return bin_metadata, cardinalities

# apply the above bins
def apply_numerical_bins(df, numerical_columns, numerical_bin_metadata):
    discretized_columns = {}

    # for each numerical column look up the bin info that was fitted earlier (in metadata)
    for col in numerical_columns:
        metadata = numerical_bin_metadata[col]
        cardinality = int(metadata["n_bins"])

        if cardinality <= 1:
            discretized_columns[col] = 0
            continue

        edges = []
        for x in metadata["edges"]:
            edges.append(float(x))

        cut_edges = [-np.inf]
        for edge in edges[1:-1]:
            cut_edges.append(edge)
        cut_edges.append(np.inf)

        binned = pd.cut(
            pd.to_numeric(df[col], errors="coerce"),
            bins = cut_edges,
            labels = False,
            include_lowest = True,
        )
        discretized_columns[col] = binned.astype(int)

    return pd.DataFrame(discretized_columns, index=df.index) # builds a new dataframe from all the binned columns at once, using the same row index as the original df (next function)

# %% Final setup (fitting it)

def fit_preprocessor(df, config, representation="baseline", n_bins=None, binning_strategy="quantile", missing_token="__MISSING__", unknown_token="__UNKNOWN__"):
    # eveyrthing below until the next comment just resolves which columns are cat. vs. num. and fills in missing values (using what we defined earlier)
    if representation not in {"baseline", "tensor"}:
        raise ValueError(f"Unsupported representation: {representation}")

    target_column = config["target_column"]

    categorical_columns, numerical_columns = infer_feature_columns(
        df=df,
        target_column=target_column,
        categorical_columns=config.get("categorical_columns"),
        numerical_columns=config.get("numerical_columns"),
    )

    df = fill_missing_values(
        df,
        target_column=target_column,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        missing_token=missing_token,
    )

    # eveyrthing below builds the target and feature mappings
    target_mapping = build_mapping(df[target_column])

    feature_mappings = {} # the string-to-integer mapping for the target column
    cat_cardinalities = [] # tracks the cardinalities

    for col in categorical_columns:
        mapping = build_mapping(df[col], extra_values=[missing_token, unknown_token])
        feature_mappings[col] = mapping
        cat_cardinalities.append(len(mapping))
        
    # recording numerical fill values
    numerical_fill_values = {}

    for col in numerical_columns:
        median = pd.to_numeric(df[col], errors="coerce").median(skipna=True)
        if pd.isna(median):
            median = 0.0
        numerical_fill_values[col] = float(median)
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median)

    # numerical binning (tensor representation only)
    n_bins = n_bins or config.get("default_n_bins", 10)
    numerical_bin_metadata = {}
    numerical_cardinalities = []

    if representation == "tensor" and numerical_columns:
        numerical_bin_metadata, numerical_cardinalities = fit_numerical_bins(
            df=df,
            numerical_columns=numerical_columns,
            n_bins=n_bins,
            strategy=binning_strategy,
        )

    cardinalities = cat_cardinalities + numerical_cardinalities # combines the cardinalities

    # bundles everything that is needed later to transform future dataframe (i.e. metadata)
    return {
        "target_column": target_column,
        "feature_names": categorical_columns + numerical_columns,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "representation": representation,
        "cardinalities": cardinalities,
        "categorical_cardinalities": cat_cardinalities,
        "numerical_cardinalities": numerical_cardinalities,
        "target_mapping": target_mapping,
        "feature_mappings": feature_mappings,
        "numerical_fill_values": numerical_fill_values,
        "numerical_bin_metadata": numerical_bin_metadata,
        "missing_token": missing_token,
        "unknown_token": unknown_token,
        "n_features": len(categorical_columns) + len(numerical_columns),
    }

# %% Transformation

def transform_dataset(df, metadata):
    target_column = metadata["target_column"]
    categorical_columns = metadata["categorical_columns"]
    numerical_columns = metadata["numerical_columns"]
    missing_token = metadata.get("missing_token", "__MISSING__")
    unknown_token = metadata.get("unknown_token", "__UNKNOWN__")

    df = fill_missing_values(
        df,
        target_column=target_column,
        categorical_columns=categorical_columns,
        numerical_columns=[], # fills missing target and categorical values wih the placeholder token from earlier
        missing_token=missing_token,
    )

    # applies categorical mappings:
    processed_df = pd.DataFrame(index=df.index)
    for col in categorical_columns:
        processed_df[col] = apply_mapping(
            df[col],
            metadata["feature_mappings"][col],
            unknown_token=unknown_token,
        )

    # handling the numerical columns:
    for col in numerical_columns:
        fill_value = metadata.get("numerical_fill_values", {}).get(col, 0.0)
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill_value)

    if metadata["representation"] == "baseline": # numerical columns are just appended as-is (plain numbers)
        if numerical_columns:
            processed_df = pd.concat([processed_df, df[numerical_columns].copy()], axis=1)
    else: # (i.e. the tensor representation) instead of raw numbers, call apply_numerical_bins to turn them into bin indices first
        if numerical_columns:
            processed_df = pd.concat(
                [
                    processed_df,
                    apply_numerical_bins(
                        df=df,
                        numerical_columns=numerical_columns,
                        numerical_bin_metadata=metadata["numerical_bin_metadata"],
                    ),
                ],
                axis=1,
            )
    
    processed_df["target"] = apply_mapping(df[target_column], metadata["target_mapping"]) # adds the target column and applies the mapping

    return processed_df.reset_index(drop=True)
# %% Saves the tuning split + 5 folds as CSV files

def save_splits(tuning_train_df, tuning_val_df, folds, output_dir, dataset_name, representation):
    output_dir.mkdir(parents=True, exist_ok=True) # creates the output folder if it doesn't already exist, creates any missing parent folder 
    paths = {} # will collect every file path created (so make_dataset knows where it was saved)

    # builds file names:
    paths["tuning_train"] = output_dir / f"{dataset_name}_{representation}_tuning_train.csv"
    paths["tuning_val"] = output_dir / f"{dataset_name}_{representation}_tuning_val.csv"
    tuning_train_df.to_csv(paths["tuning_train"], index=False)
    tuning_val_df.to_csv(paths["tuning_val"], index=False)

    for fold_number, (fold_train_df, fold_test_df) in enumerate(folds, start=1): # for each fold, build two files names corresponding to the train and test csv.
        train_path = output_dir / f"{dataset_name}_{representation}_fold_{fold_number}_train.csv"
        test_path = output_dir / f"{dataset_name}_{representation}_fold_{fold_number}_test.csv"

        fold_train_df.to_csv(train_path, index=False)
        fold_test_df.to_csv(test_path, index=False)

        paths[f"fold_{fold_number}_train"] = train_path
        paths[f"fold_{fold_number}_test"] = test_path

    return paths

# %% Saves metadata

def save_metadata(metadata, output_dir, dataset_name, representation):
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / f"{dataset_name}_{representation}_metadata.json" # makes file name

    with open(metadata_path, "w", encoding="utf-8") as f: # opens the file for writing
        json.dump(metadata, f, indent=2) # converts the metadata dictionary into JSON text and writes it directly to the open file

    return metadata_path

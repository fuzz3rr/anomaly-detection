import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold


def load_data(file_paths, dataset_config):
    """
    Universal data loader that works with any dataset configuration.

    Args:
        file_paths: List of paths to CSV files
        dataset_config: Dataset configuration from config.json
    """
    dfs = []
    label_column = dataset_config.get("label_column", "Label")
    column_mapping = dataset_config.get("column_mapping", {})
    has_header = dataset_config.get("has_header", True)

    for fp in file_paths:
        if has_header:
            df = pd.read_csv(fp, on_bad_lines="skip", low_memory=False)
            df.columns = df.columns.str.strip()
        else:
            # Read CSV without headers and generate column names
            df = pd.read_csv(fp, on_bad_lines="skip", low_memory=False, header=None)
            # Generate column names: col_0, col_1, ..., col_N-2, <label_column>
            num_cols = len(df.columns)
            col_names = [f"col_{i}" for i in range(num_cols - 1)] + [label_column]
            df.columns = col_names

        # Apply column mapping if specified
        if column_mapping:
            df = df.rename(columns=column_mapping)

        dfs.append(df)

        # Show label info
        if label_column in df.columns:
            unique_labels = df[label_column].unique()
            label_preview = unique_labels[0] if len(unique_labels) == 1 else f"{len(unique_labels)} classes"
            print(f"  - {fp.split('/')[-1]}: {len(df)} records, label: {label_preview}")
        else:
            print(f"  - {fp.split('/')[-1]}: {len(df)} records")

    data = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal records loaded: {len(data)}")
    print(f"Classes in '{label_column}': {data[label_column].unique()}")
    return data


def preprocess_data(data, dataset_config, preprocessing_config):
    """
    Universal preprocessor that works with any dataset configuration.

    Args:
        data: DataFrame to preprocess
        dataset_config: Dataset configuration from config.json
        preprocessing_config: Preprocessing parameters from config.json
    """
    label_column = dataset_config.get("label_column", "Label")
    drop_columns = dataset_config.get("drop_columns", [])
    variance_threshold = preprocessing_config.get("variance_threshold", 0.0)
    correlation_threshold = preprocessing_config.get("correlation_threshold", 0.95)

    # Drop specified columns
    data = data.drop(drop_columns, axis=1, errors="ignore")

    # Convert categorical columns to numeric using one-hot encoding
    object_cols = [col for col in data.columns if data[col].dtype == 'object' and col != label_column]
    if object_cols:
        print(f"\nApplying one-hot encoding to categorical features: {', '.join(object_cols)}")
        data = pd.get_dummies(data, columns=object_cols, dummy_na=False)
    else:
        print("\nNo categorical features to encode.")

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    y = data[label_column]
    X = data.drop(label_column, axis=1)

    print("\n  >> Feature selection...")

    X = drop_low_variance_features(X, threshold=variance_threshold)
    X = drop_high_correlation_features(X, threshold=correlation_threshold)

    data = pd.concat([X, y], axis=1)

    print(f"\nFeatures remaining after preprocessing: {X.shape[1]}")

    return data


def prepare_binary_labels(y, positive_class="BENIGN"):
    """
    Prepare binary labels based on the specified positive class.

    Args:
        y: Series with labels
        positive_class: The class to be labeled as 0 (negative/benign)
    """
    return y.apply(lambda x: 0 if x == positive_class else 1)


def prepare_multiclass_labels(y):
    """Encode multiclass labels."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le


def get_binary_target_names(dataset_config):
    """Get target names for binary classification from config."""
    binary_labels = dataset_config.get("binary_labels")
    if binary_labels:
        return binary_labels
    return ["NEGATIVE", "POSITIVE"]


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def drop_low_variance_features(data, threshold=0.0):
    selector = VarianceThreshold(threshold=threshold)

    selector.fit(data)
    kept_columns = data.columns[selector.get_support(indices=True)]
    dropped_columns = list(set(data.columns) - set(kept_columns))

    if dropped_columns:
        print(f"Removed low-variance features: {len(dropped_columns)} columns")

    return data[kept_columns]


def drop_high_correlation_features(data, threshold=0.9):
    corr = data.corr().abs()

    upper = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if to_drop:
        print(f"Removed highly correlated features (> {threshold}): {to_drop}")

    return data.drop(columns=to_drop, errors="ignore")
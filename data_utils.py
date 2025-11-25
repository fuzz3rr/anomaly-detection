import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold

def load_data(file_paths):
    dfs = []
    for fp in file_paths:
        df = pd.read_csv(fp, on_bad_lines="skip", low_memory=False)
        df.columns = df.columns.str.strip()
        dfs.append(df)
        print(
            f"  - {fp.split('/')[-1]}: {len(df)} records, label: {df['Label'].unique()[0]}"
        )

    data = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal records loaded: {len(data)}")
    print(f"Attack classes: {data['Label'].unique()}")
    return data


def preprocess_data(data):
    # TODO: check mathematical solutions to exclude columns; e.g. correlation between Label and Destination IP / tests with models - add column, delete column and test
    data = data.drop(
        [
            "Unnamed: 0",
            "Flow ID",
            "Source IP",
            "Destination IP",
            "Timestamp",
            "Source Port",
            "Destination Port",
        ],
        axis=1,
        errors="ignore",
    )

    # Convert categorical columns to numeric using one-hot encoding
    object_cols = [col for col in data.columns if data[col].dtype == 'object' and col != 'Label']
    if object_cols:
        print(f"\nApplying one-hot encoding to categorical features: {', '.join(object_cols)}")
        data = pd.get_dummies(data, columns=object_cols, dummy_na=False)
    else:
        print("\nNo categorical features to encode.")

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    y = data["Label"]
    X = data.drop("Label", axis=1)

    print("\n  >> Feature selection...")

    X = drop_low_variance_features(X, threshold=0.0)

    X = drop_high_correlation_features(X, threshold=0.95)

    data = pd.concat([X, y], axis=1)

    print(f"\nFeatures remaining after preprocessing: {X.shape[1]}")

    return data


def prepare_binary_labels(y):
    return y.apply(lambda x: 0 if x == "BENIGN" else 1)


def prepare_multiclass_labels(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le


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
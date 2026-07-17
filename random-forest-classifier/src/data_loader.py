import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

RAW_DATA_PATH = "../data/churn_data.csv"

TARGET_CANDIDATES = ["churn", "exited", "is_churn", "target", "label"]

ID_LIKE_COLS = ["customerid", "customer_id", "id", "rownumber", "surname"]


def _find_target_column(df):
    lowered = {c.lower(): c for c in df.columns}
    for cand in TARGET_CANDIDATES:
        if cand in lowered:
            return lowered[cand]
    print("[data_loader] couldn't auto-match a target column name, "
          "falling back to the last column:", df.columns[-1])
    return df.columns[-1]


def load_raw_data(path=RAW_DATA_PATH):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_and_split(df, target_col=None, test_size=0.2, random_state=42):
    df = df.copy()

    drop_cols = [c for c in df.columns if c.lower() in ID_LIKE_COLS]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if target_col is None:
        target_col = _find_target_column(df)

    for col in df.columns:
        if df[col].dtype == object and col != target_col:
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().mean() > 0.9:
                df[col] = coerced

    before = len(df)
    df = df.dropna()
    after = len(df)
    if before != after:
        print(f"[data_loader] dropped {before - after} rows with missing values")

    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    if y_raw.dtype == object or y_raw.dtype.name == "category" or y_raw.dtype == bool:
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y_raw)
        print("[data_loader] target classes:", list(target_encoder.classes_))
    else:
        y = y_raw.values
        target_encoder = None

    cat_cols = X.select_dtypes(include=["object", "category", "str"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "target_encoder": target_encoder,
        "scaler": scaler,
    }


def get_processed_data(path=RAW_DATA_PATH, target_col=None):
    df = load_raw_data(path)
    print(f"[data_loader] loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return clean_and_split(df, target_col=target_col)


if __name__ == "__main__":
    data = get_processed_data()
    print("train shape:", data["X_train"].shape)
    print("test shape:", data["X_test"].shape)

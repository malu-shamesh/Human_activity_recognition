"""Feature/label separation and normalization for the HAR dataset."""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

NON_FEATURE_COLUMNS = ["subject", "Activity"]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in NON_FEATURE_COLUMNS]


def fit_label_encoder(df: pd.DataFrame) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(df["Activity"])
    return encoder


def prepare_features_labels(
    df: pd.DataFrame, scaler: MinMaxScaler, label_encoder: LabelEncoder, fit_scaler: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (scaled feature matrix, encoded labels) for the given split.

    Pass fit_scaler=True only for the training split; the scaler and label encoder
    should otherwise be fit once on train data and reused (via transform) for other splits.
    """
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    X_scaled = scaler.fit_transform(X) if fit_scaler else scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)

    y = pd.Series(label_encoder.transform(df["Activity"]), index=df.index, name="Activity")
    return X_scaled, y

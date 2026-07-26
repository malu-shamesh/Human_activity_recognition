import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.preprocessing import fit_label_encoder, get_feature_columns, prepare_features_labels


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature1": [0.0, 5.0, 10.0],
            "feature2": [1.0, 1.0, 1.0],
            "subject": [1, 1, 2],
            "Activity": ["WALKING", "SITTING", "WALKING"],
        }
    )


def test_get_feature_columns_excludes_subject_and_activity():
    df = _make_df()
    assert get_feature_columns(df) == ["feature1", "feature2"]


def test_prepare_features_labels_scales_and_encodes():
    df = _make_df()
    scaler = MinMaxScaler()
    encoder = fit_label_encoder(df)

    X, y = prepare_features_labels(df, scaler, encoder, fit_scaler=True)

    assert list(X["feature1"]) == [0.0, 0.5, 1.0]
    assert set(y.unique()) == {0, 1}

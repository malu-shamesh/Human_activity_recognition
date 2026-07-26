"""Download and load the UCI Human Activity Recognition with Smartphones dataset."""

from pathlib import Path

import kagglehub
import pandas as pd

DATASET_SLUG = "uciml/human-activity-recognition-with-smartphones"


def download_dataset() -> Path:
    """Download the dataset via kagglehub and return the local directory path."""
    dataset_dir = kagglehub.dataset_download(DATASET_SLUG)
    return Path(dataset_dir)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.replace(r"[()]", "", regex=True)
        .str.replace("-", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df = df.rename(columns={"subject/Participant": "subject"})
    return df


def load_activity_data(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the HAR dataset as a single DataFrame with 'subject' and 'Activity' columns.

    If data_dir is not provided, downloads the dataset via kagglehub first. Handles either
    a single combined CSV or Kaggle's typical train.csv/test.csv split.
    """
    if data_dir is None:
        data_dir = download_dataset()

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = [pd.read_csv(csv_path) for csv_path in csv_files]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    df = _clean_columns(df)
    df = df.dropna(subset=["Activity", "subject"])
    return df

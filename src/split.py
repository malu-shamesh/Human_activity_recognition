"""Subject-aware train/test splitting.

Random row-level splits leak information: a single subject's readings are highly
correlated, so if the same person's rows appear in both train and test, accuracy
overstates how well the model generalizes to a genuinely new person. Splitting by
subject keeps every person's data entirely in one split or the other.
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_by_subject(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (train_df, test_df) such that no subject appears in both."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df["subject"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

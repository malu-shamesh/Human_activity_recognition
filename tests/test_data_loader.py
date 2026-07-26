import pandas as pd

from src.data_loader import _clean_columns


def test_clean_columns_strips_special_characters_and_renames_subject():
    df = pd.DataFrame(
        {
            "tBodyAcc-mean()-X": [0.1],
            "angle(X,gravityMean)": [0.2],
            "subject/Participant": [1],
            "Activity": ["STANDING"],
        }
    )

    cleaned = _clean_columns(df)

    assert list(cleaned.columns) == ["tBodyAccmeanX", "angleXgravityMean", "subject", "Activity"]

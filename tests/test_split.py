import pandas as pd

from src.split import split_by_subject


def _make_df(n_subjects: int, rows_per_subject: int = 10) -> pd.DataFrame:
    rows = []
    for subject in range(n_subjects):
        for _ in range(rows_per_subject):
            rows.append({"subject": subject, "Activity": "WALKING", "feature1": 0.5})
    return pd.DataFrame(rows)


def test_split_by_subject_keeps_subjects_disjoint():
    df = _make_df(n_subjects=20)
    train_df, test_df = split_by_subject(df, test_size=0.2, seed=0)

    train_subjects = set(train_df["subject"])
    test_subjects = set(test_df["subject"])

    assert not (train_subjects & test_subjects)
    assert train_subjects | test_subjects == set(df["subject"])


def test_split_by_subject_keeps_all_rows_for_a_subject_together():
    df = _make_df(n_subjects=20, rows_per_subject=10)
    train_df, test_df = split_by_subject(df, test_size=0.2, seed=0)

    for split_df in (train_df, test_df):
        counts = split_df["subject"].value_counts()
        assert (counts == 10).all()

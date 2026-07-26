"""Train an XGBoost classifier for human activity recognition."""

import pandas as pd
from xgboost import XGBClassifier


def train_model(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model

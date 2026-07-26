"""Evaluate a trained classifier and save metrics/confusion matrix reports."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def evaluate_and_report(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label_encoder: LabelEncoder,
    output_dir: Path,
) -> dict:
    """Predict on X_test, save a classification report and confusion matrix, return metrics."""
    predictions = model.predict(X_test)
    class_names = label_encoder.classes_

    report = classification_report(
        y_test, predictions, target_names=class_names, output_dict=True
    )
    matrix = confusion_matrix(y_test, predictions)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        output_dir / "confusion_matrix.csv"
    )

    return report

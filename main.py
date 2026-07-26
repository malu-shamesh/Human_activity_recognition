"""CLI entrypoint for the human activity recognition pipeline.

Usage:
    python main.py --output-dir outputs
    python main.py --data-dir data/har --test-size 0.3  # local data, custom split
"""

import argparse
import joblib
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data_loader import load_activity_data
from src.evaluate import evaluate_and_report
from src.model import train_model
from src.preprocessing import fit_label_encoder, prepare_features_labels
from src.split import split_by_subject
from src.visualize import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_tsne,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Human activity recognition pipeline")
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Directory containing the HAR CSV(s). If omitted, downloads via kagglehub.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"),
        help="Directory to write plots and evaluation reports to.",
    )
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/xgboost_har.joblib"),
        help="Path to save the trained model to.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of subjects to hold out for testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading activity data...")
    df = load_activity_data(args.data_dir)
    print(f"Loaded {len(df)} rows across {df['subject'].nunique()} subjects.")

    print("Generating exploratory plots...")
    plot_class_distribution(df, args.output_dir)

    print(f"Splitting by subject (test_size={args.test_size})...")
    train_df, test_df = split_by_subject(df, test_size=args.test_size)
    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

    scaler = MinMaxScaler()
    label_encoder = fit_label_encoder(train_df)

    X_train, y_train = prepare_features_labels(train_df, scaler, label_encoder, fit_scaler=True)
    X_test, y_test = prepare_features_labels(test_df, scaler, label_encoder, fit_scaler=False)

    plot_tsne(X_train, train_df["Activity"], args.output_dir)

    print("Training XGBoost classifier...")
    model = train_model(X_train, y_train)
    joblib.dump({"model": model, "scaler": scaler, "label_encoder": label_encoder}, args.model_path)
    print(f"Saved model to {args.model_path}")

    print("Evaluating on held-out subjects...")
    report = evaluate_and_report(model, X_test, y_test, label_encoder, args.output_dir)
    print(f"Test accuracy: {report['accuracy']:.3f}")

    matrix_df = pd.read_csv(args.output_dir / "confusion_matrix.csv", index_col=0)
    plot_confusion_matrix(matrix_df, args.output_dir)
    plot_feature_importance(model, list(X_train.columns), args.output_dir)
    print(f"Saved evaluation report and plots to {args.output_dir}")


if __name__ == "__main__":
    main()

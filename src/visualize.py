"""Plotting helpers that save figures to an output directory instead of displaying inline."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from xgboost import XGBClassifier


def _save(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df["Activity"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
    ax.set_title("Activity Class Distribution")
    _save(fig, output_dir, "class_distribution.png")


def plot_tsne(X: pd.DataFrame, activity_labels: pd.Series, output_dir: Path, sample_size: int = 2000) -> None:
    """Project features to 2D with t-SNE, sampling rows for speed on large datasets."""
    if len(X) > sample_size:
        sample_idx = X.sample(n=sample_size, random_state=0).index
        X = X.loc[sample_idx]
        activity_labels = activity_labels.loc[sample_idx]

    embedding = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=activity_labels.values, palette="colorblind", ax=ax)
    ax.set_title("t-SNE Projection of Activity Features")
    _save(fig, output_dir, "tsne.png")


def plot_confusion_matrix(matrix_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    _save(fig, output_dir, "confusion_matrix.png")


def plot_feature_importance(model: XGBClassifier, feature_names: list[str], output_dir: Path, top_n: int = 20) -> None:
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=importance.values, y=importance.index, ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    _save(fig, output_dir, "feature_importance.png")

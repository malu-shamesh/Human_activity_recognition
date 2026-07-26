# Human Activity Recognition

Classifies human activity (walking, sitting, standing, laying, walking upstairs/downstairs) from smartphone accelerometer/gyroscope features. Originally a Kaggle notebook comparing two arbitrary Logistic Regression train/test splits; rewritten as a runnable Python pipeline with a stronger model and a subject-aware evaluation split.

## Pipeline

1. **Data loading** ([src/data_loader.py](src/data_loader.py)) - downloads the [UCI HAR](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones) dataset via `kagglehub`, or reads local CSVs, and cleans up the raw column names (which contain parentheses, hyphens, and commas).
2. **Subject-aware split** ([src/split.py](src/split.py)) - splits into train/test by subject, so no person's readings appear in both splits.
3. **Preprocessing** ([src/preprocessing.py](src/preprocessing.py)) - min-max scales the 561 feature columns (fit on train only) and label-encodes the six activity classes.
4. **Model training** ([src/model.py](src/model.py)) - trains an `XGBClassifier` on the scaled features.
5. **Evaluation** ([src/evaluate.py](src/evaluate.py)) - saves a classification report (accuracy/precision/recall/F1 per class) and confusion matrix.
6. **Visualization** ([src/visualize.py](src/visualize.py)) - saves the activity class distribution, a t-SNE projection of the features, the confusion matrix, and XGBoost feature importances as PNGs.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Kaggle API credentials are required for automatic dataset download - see [kagglehub's authentication docs](https://github.com/Kaggle/kagglehub#authenticate). Alternatively, download the dataset manually and pass its directory with `--data-dir`.

## Usage

```bash
# Full pipeline: download data, split by subject, train, evaluate, save plots
python main.py --output-dir outputs

# Local data, larger held-out subject fraction
python main.py --data-dir data/har --test-size 0.3
```

Outputs land in `outputs/` (plots, `classification_report.csv`, `confusion_matrix.csv`), and the trained model (plus its scaler and label encoder) is saved to `--model-path` via `joblib`.

## Tests

```bash
pytest
```

## Why these changes

**Model**: the original notebook used Logistic Regression and compared two oddly-chosen splits ("Case 1": 80% train, "Case 2": 10% train) mainly to observe that the smaller training set scored higher - an artifact of the split, not a real result. XGBoost is the modern standard for tabular classification with hand-engineered numeric features like this dataset's 561 sensor-derived columns, and consistently outperforms plain logistic regression on this kind of data.

**Split methodology**: the original notebook used a plain random row-level `train_test_split`. Since each of the 30 subjects contributes many highly correlated rows (consecutive sensor readings from the same walk, sit, etc.), a random split lets the same subject appear in both train and test - the model can partly "recognize" a subject's signal rather than learning to generalize to unseen people, inflating accuracy. This pipeline instead splits by subject (`GroupShuffleSplit` on the `subject` column), so held-out accuracy reflects performance on genuinely new people, which is the setting this dataset is actually meant to evaluate.

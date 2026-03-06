"""
Model training module — trains a classification model using a rolling window
strategy to ensure the model reflects recent market conditions.
"""

import joblib
import pandas as pd
from pathlib import Path
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline


FEATURE_COLS = [
    "return_1d", "return_3d", "return_7d", "return_14d",
    "price_to_ma_7d", "price_to_ma_14d", "price_to_ma_30d",
    "volatility_7d", "volatility_14d",
    "volume_ratio",
]
TARGET_COL = "direction"


def get_rolling_window_data(df: pd.DataFrame, window_days: int = 730) -> pd.DataFrame:
    """Return only the most recent `window_days` rows for training."""
    cutoff = df.index.max() - timedelta(days=window_days)
    return df[df.index >= cutoff]


def build_model_pipeline(algorithm: str = "random_forest") -> Pipeline:
    """Build a sklearn Pipeline with scaling + classifier."""
    if algorithm == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
    elif algorithm == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])


def train(df: pd.DataFrame, algorithm: str = "random_forest",
          window_days: int = 730, test_size: float = 0.2) -> tuple:
    """
    Train the model on a rolling window of data.
    Returns (trained_pipeline, metrics_dict, X_test, y_test).
    """
    data = get_rolling_window_data(df, window_days)
    X = data[FEATURE_COLS]
    y = data[TARGET_COL]

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    pipeline = build_model_pipeline(algorithm)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0
        ),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    return pipeline, metrics, X_test, y_test


def save_model(pipeline: Pipeline, output_path: str) -> None:
    """Persist trained pipeline to disk."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"[train] Model saved to {output_path}")


def load_model(path: str) -> Pipeline:
    """Load persisted pipeline from disk."""
    return joblib.load(path)
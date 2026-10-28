"""Train and evaluate a logistic-regression model from a labeled CSV dataset."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def build_model() -> Pipeline:
    """Build a scaled logistic-regression pipeline for numeric features."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_and_evaluate(
    data: pd.DataFrame,
    target_column: str,
    fraud_label: Any,
    test_size: float = 0.25,
) -> tuple[Pipeline, dict[str, float | list[list[int]]], pd.DataFrame, list[str]]:
    """Train on a stratified split and evaluate fraud predictions on held-out data."""
    if target_column not in data.columns:
        raise ValueError(f"Target column {target_column!r} was not found.")
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("The test-set size must be between 0.1 and 0.5.")

    feature_columns = [
        column
        for column in data.select_dtypes(include=["number"]).columns
        if column != target_column
    ]
    if not feature_columns:
        raise ValueError(
            "No numeric feature columns were found. "
            "The dataset needs at least one numeric input column."
        )

    required_data = data.loc[:, [*feature_columns, target_column]]
    if required_data.isna().any().any():
        missing = required_data.isna().sum()
        columns_with_missing = [
            f"{column} ({int(count)})"
            for column, count in missing.items()
            if count > 0
        ]
        raise ValueError(
            "The selected target or numeric feature columns contain missing "
            "values. Clean these columns and upload the CSV again: "
            + ", ".join(columns_with_missing[:8])
        )

    if not np.isfinite(data.loc[:, feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("Numeric feature columns must contain only finite values.")

    original_labels = data[target_column]
    unique_labels = pd.unique(original_labels)
    if len(unique_labels) != 2:
        raise ValueError(
            f"Target column {target_column!r} must have exactly two classes; "
            f"found {len(unique_labels)}."
        )
    if not any(fraud_label == label for label in unique_labels):
        raise ValueError("The selected fraud label is not present in the target.")

    labels = (original_labels == fraud_label).astype(int)
    class_counts = labels.value_counts()
    if int(class_counts.min()) < 2:
        raise ValueError(
            "The dataset needs at least two rows from each class for a "
            "stratified train/test split."
        )

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            data.loc[:, feature_columns],
            labels,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=labels,
        )
    except ValueError as error:
        raise ValueError(
            "Could not create a stratified train/test split. Add more labeled "
            "rows or choose a larger test-set size."
        ) from error

    model = build_model()
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics: dict[str, float | list[list[int]]] = {
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
    }
    test_results = pd.DataFrame(
        {
            "actual_fraud": y_test.to_numpy(),
            "predicted_fraud": predictions,
            "fraud_probability": probabilities,
        },
        index=y_test.index,
    )
    return model, metrics, test_results, feature_columns


def predict_transactions(
    model: Pipeline,
    data: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Predict fraud labels and probabilities for a CSV of unlabeled transactions."""
    if data.empty:
        raise ValueError("The prediction CSV contains no transactions.")

    missing_columns = [column for column in feature_columns if column not in data]
    if missing_columns:
        raise ValueError(
            "The prediction CSV is missing model features: "
            + ", ".join(missing_columns[:8])
        )

    try:
        features = data.loc[:, feature_columns].astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Every model feature in the prediction CSV must contain numbers."
        ) from error

    feature_values = features.to_numpy()
    if features.isna().any().any() or not np.isfinite(feature_values).all():
        raise ValueError(
            "Model features in the prediction CSV must not contain missing or "
            "infinite values."
        )

    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    results = data.copy()
    results["predicted_fraud"] = predictions
    results["fraud_probability"] = probabilities
    return results

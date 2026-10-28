"""Train, evaluate, and save a model from creditcard.csv in this folder."""

from pathlib import Path

import joblib
import pandas as pd

from fraud_detection import train_and_evaluate


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "creditcard.csv"
MODEL_PATH = PROJECT_DIR / "credit_card_fraud_model.joblib"


def main() -> None:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. Put creditcard.csv in this folder."
        )

    data = pd.read_csv(DATASET_PATH)
    if "Class" not in data.columns:
        raise ValueError(
            "The Kaggle dataset must have a 'Class' column "
            "(0 = legitimate, 1 = fraud)."
        )

    model, metrics, test_results, feature_columns = train_and_evaluate(
        data,
        target_column="Class",
        fraud_label=1,
        test_size=0.25,
    )

    labels = (data["Class"] == 1).astype(int)
    model.fit(data.loc[:, feature_columns], labels)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "target_column": "Class",
            "fraud_label": 1,
        },
        MODEL_PATH,
    )

    print(f"Dataset: {DATASET_PATH.name}")
    print(f"Transactions: {len(data):,}")
    print(f"Numeric features: {len(feature_columns)}")
    print(f"Fraud transactions: {int(labels.sum()):,} ({labels.mean():.4%})")
    print(f"Training split: {len(data) - len(test_results):,}")
    print(f"Held-out test split: {len(test_results):,}")
    print("\nHeld-out test metrics (fraud is the positive class):")
    for metric_name in (
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "average_precision",
    ):
        print(f"  {metric_name.replace('_', ' ').title()}: {metrics[metric_name]:.4f}")
    print("  Confusion matrix [[TN, FP], [FN, TP]]:")
    print(f"    {metrics['confusion_matrix']}")
    print(f"\nSaved full-data trained model: {MODEL_PATH}")
    print("Note: this project model is for learning and not real financial decisions.")


if __name__ == "__main__":
    main()

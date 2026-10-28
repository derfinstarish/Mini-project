from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import app as website
from fraud_detection import predict_transactions, train_and_evaluate


def make_dataset() -> pd.DataFrame:
    random = np.random.default_rng(42)
    fraud = np.zeros(400, dtype=int)
    fraud[::20] = 1
    return pd.DataFrame(
        {
            "Amount": random.normal(size=400) + fraud * 5,
            "Time": random.normal(size=400),
            "merchant": ["shop"] * 400,
            "Class": fraud,
        }
    )


class FraudDetectionTests(unittest.TestCase):
    def test_training_returns_metrics_predictions_and_numeric_features(self) -> None:
        model, metrics, predictions, feature_columns = train_and_evaluate(
            make_dataset(), "Class", 1
        )

        self.assertEqual(feature_columns, ["Amount", "Time"])
        self.assertEqual(len(predictions), 100)
        self.assertEqual(
            sum(map(sum, metrics["confusion_matrix"])), len(predictions)
        )
        for metric in (
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "average_precision",
        ):
            self.assertTrue(0.0 <= metrics[metric] <= 1.0)
        self.assertEqual(
            model.predict(make_dataset()[feature_columns].head(1)).shape,
            (1,),
        )

    def test_training_rejects_missing_feature_values(self) -> None:
        data = make_dataset()
        data.loc[0, "Amount"] = np.nan

        with self.assertRaisesRegex(ValueError, "missing values"):
            train_and_evaluate(data, "Class", 1)

    def test_training_rejects_non_binary_target(self) -> None:
        data = make_dataset()
        data.loc[0, "Class"] = 2

        with self.assertRaisesRegex(ValueError, "exactly two classes"):
            train_and_evaluate(data, "Class", 1)

    def test_training_rejects_invalid_fraud_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "fraud label"):
            train_and_evaluate(make_dataset(), "Class", 5)

    def test_prediction_accepts_unlabeled_transactions(self) -> None:
        data = make_dataset()
        model, _, _, feature_columns = train_and_evaluate(data, "Class", 1)
        transactions = data.loc[:4, ["Amount", "Time"]].copy()

        predictions = predict_transactions(model, transactions, feature_columns)

        self.assertEqual(len(predictions), len(transactions))
        self.assertIn("predicted_fraud", predictions.columns)
        self.assertIn("fraud_probability", predictions.columns)
        self.assertTrue(predictions["fraud_probability"].between(0, 1).all())

    def test_prediction_rejects_missing_model_features(self) -> None:
        data = make_dataset()
        model, _, _, feature_columns = train_and_evaluate(data, "Class", 1)

        with self.assertRaisesRegex(ValueError, "missing model features"):
            predict_transactions(model, data[["Amount"]], feature_columns)


class WebsiteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        output_dir = Path(self.temp_dir.name)
        self.path_patches = (
            patch.object(website, "MODEL_PATH", output_dir / "model.joblib"),
            patch.object(
                website,
                "TEST_PREDICTIONS_PATH",
                output_dir / "test_predictions.csv",
            ),
            patch.object(
                website,
                "PREDICTIONS_PATH",
                output_dir / "predictions.csv",
            ),
            patch.object(website, "DATASET_PATH", output_dir / "creditcard.csv"),
        )
        for path_patch in self.path_patches:
            path_patch.start()
        website.app.config.update(TESTING=True)
        self.client = website.app.test_client()

    def tearDown(self) -> None:
        for path_patch in reversed(self.path_patches):
            path_patch.stop()
        self.temp_dir.cleanup()

    def test_home_serves_html(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Credit Card Fraud Detection", response.data)
        self.assertIn(b"Predict transactions from a CSV", response.data)
        self.assertTrue(response.content_type.startswith("text/html"))

    def test_home_trains_from_local_csv_and_shows_automatic_prediction(self) -> None:
        make_dataset().to_csv(website.DATASET_PATH, index=False)

        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Automatic sample prediction", response.data)
        self.assertTrue(website.MODEL_PATH.is_file())
        self.assertTrue(website.TEST_PREDICTIONS_PATH.is_file())

    def test_training_and_prediction_work_through_html_forms(self) -> None:
        dataset = make_dataset()
        uploaded_csv = dataset.to_csv(index=False).encode("utf-8")
        train_response = self.client.post(
            "/train",
            data={
                "dataset": (BytesIO(uploaded_csv), "transactions.csv"),
                "target_column": "Class",
                "fraud_label": "1",
                "test_percentage": "25",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(train_response.status_code, 200)
        self.assertIn(b"Model training complete", train_response.data)
        self.assertTrue(website.MODEL_PATH.is_file())
        self.assertTrue(website.TEST_PREDICTIONS_PATH.is_file())

        model_download = self.client.get("/download/model")
        self.assertEqual(model_download.status_code, 200)
        self.assertIn(
            "credit_card_fraud_model.joblib",
            model_download.headers["Content-Disposition"],
        )
        model_download.close()

        unlabeled_csv = dataset[["Amount", "Time"]].head(5).to_csv(index=False)
        prediction_response = self.client.post(
            "/predict",
            data={"transactions": (BytesIO(unlabeled_csv.encode()), "new.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(prediction_response.status_code, 200)
        self.assertIn(b"Transaction predictions ready", prediction_response.data)
        self.assertTrue(website.PREDICTIONS_PATH.is_file())

        prediction_download = self.client.get("/download/predictions")
        self.assertEqual(prediction_download.status_code, 200)
        self.assertIn(b"fraud_probability", prediction_download.data)
        prediction_download.close()

    def test_training_reports_invalid_fraud_label(self) -> None:
        uploaded_csv = make_dataset().to_csv(index=False).encode("utf-8")
        response = self.client.post(
            "/train",
            data={
                "dataset": (BytesIO(uploaded_csv), "transactions.csv"),
                "target_column": "Class",
                "fraud_label": "5",
                "test_percentage": "25",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Fraud label", response.data)


if __name__ == "__main__":
    unittest.main()

"""HTML website for training and using the credit-card fraud model."""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename

from fraud_detection import predict_transactions, train_and_evaluate


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "creditcard.csv"
MODEL_PATH = PROJECT_DIR / "credit_card_fraud_model.joblib"
TEST_PREDICTIONS_PATH = PROJECT_DIR / "fraud_test_predictions.csv"
PREDICTIONS_PATH = PROJECT_DIR / "fraud_predictions.csv"
MAX_UPLOAD_SIZE = 256 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE


def render_home(
    error: str | None = None,
    result: dict | None = None,
    status: int = 200,
):
    return (
        render_template(
            "index.html",
            error=error,
            result=result,
            dataset_available=DATASET_PATH.is_file(),
            model_available=MODEL_PATH.is_file(),
        ),
        status,
    )


def automatic_prediction() -> dict | None:
    """Train from the local dataset if needed and predict its first transaction."""
    if not MODEL_PATH.is_file():
        if not DATASET_PATH.is_file():
            return None

        data = pd.read_csv(DATASET_PATH)
        model, _, test_results, feature_columns = train_and_evaluate(
            data,
            target_column="Class",
            fraud_label=1,
            test_size=0.25,
        )
        labels = (data["Class"] == 1).astype(int)
        model.fit(data.loc[:, feature_columns], labels)
        saved_model = {
            "model": model,
            "feature_columns": feature_columns,
            "target_column": "Class",
            "fraud_label": 1,
            "sample_transaction": data.loc[:, feature_columns].iloc[[0]],
        }
        joblib.dump(saved_model, MODEL_PATH)

        test_predictions = pd.concat(
            [data.loc[test_results.index, feature_columns], test_results],
            axis=1,
        )
        test_predictions.to_csv(TEST_PREDICTIONS_PATH, index=True)
    else:
        saved_model = joblib.load(MODEL_PATH)

    feature_columns = saved_model["feature_columns"]
    sample = saved_model.get("sample_transaction")
    if sample is None:
        if not DATASET_PATH.is_file():
            return None
        sample_data = pd.read_csv(DATASET_PATH, nrows=1)
        if any(column not in sample_data.columns for column in feature_columns):
            return None
        sample = sample_data.loc[:, feature_columns]

    prediction = predict_transactions(
        saved_model["model"],
        sample,
        feature_columns,
    ).iloc[0]
    return {
        "automatic_prediction": (
            "FRAUDULENT TRANSACTION"
            if int(prediction["predicted_fraud"]) == 1
            else "LEGITIMATE TRANSACTION"
        ),
        "automatic_probability": float(prediction["fraud_probability"]),
        "sample_amount": (
            float(sample["Amount"].iloc[0]) if "Amount" in sample.columns else None
        ),
    }


@app.get("/")
def index():
    try:
        return render_home(result=automatic_prediction())
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        return render_home(error=str(error), status=400)
    except (OSError, KeyError, EOFError, ModuleNotFoundError) as error:
        app.logger.exception("Could not prepare the automatic sample prediction.")
        return render_home(
            error=f"Could not prepare the automatic prediction: {error}",
            status=500,
        )


@app.post("/train")
def train():
    uploaded_file = request.files.get("dataset")
    try:
        if uploaded_file is not None and uploaded_file.filename:
            if Path(uploaded_file.filename).suffix.lower() != ".csv":
                raise ValueError("Upload a dataset in CSV format.")
            data = pd.read_csv(uploaded_file.stream)
            dataset_name = secure_filename(uploaded_file.filename) or "uploaded CSV"
        elif DATASET_PATH.is_file():
            data = pd.read_csv(DATASET_PATH)
            dataset_name = DATASET_PATH.name
        else:
            raise ValueError(
                "Upload your Kaggle CSV, or put creditcard.csv in the project folder."
            )

        if data.empty:
            raise ValueError("The selected CSV contains no transactions.")

        target_column = request.form.get("target_column", "Class").strip()
        fraud_label_text = request.form.get("fraud_label", "1").strip()
        if target_column not in data.columns:
            raise ValueError(
                f"Target column {target_column!r} is not in the uploaded CSV."
            )

        matching_labels = [
            label
            for label in pd.unique(data[target_column])
            if str(label) == fraud_label_text
        ]
        if len(matching_labels) != 1:
            labels = ", ".join(
                str(value) for value in pd.unique(data[target_column])[:10]
            )
            raise ValueError(
                f"Fraud label {fraud_label_text!r} was not found in "
                f"{target_column!r}. Available values include: {labels}."
            )

        test_percentage = int(request.form.get("test_percentage", "25"))
        model, metrics, test_results, feature_columns = train_and_evaluate(
            data,
            target_column=target_column,
            fraud_label=matching_labels[0],
            test_size=test_percentage / 100,
        )

        labels = (data[target_column] == matching_labels[0]).astype(int)
        model.fit(data.loc[:, feature_columns], labels)
        joblib.dump(
            {
                "model": model,
                "feature_columns": feature_columns,
                "target_column": target_column,
                "fraud_label": matching_labels[0],
                "sample_transaction": data.loc[:, feature_columns].iloc[[0]],
            },
            MODEL_PATH,
        )

        test_predictions = pd.concat(
            [
                data.loc[test_results.index, feature_columns],
                test_results,
            ],
            axis=1,
        )
        test_predictions.to_csv(TEST_PREDICTIONS_PATH, index=True)
        result = {
            "dataset_name": dataset_name,
            "row_count": len(data),
            "fraud_count": int(labels.sum()),
            "feature_count": len(feature_columns),
            "test_count": len(test_results),
            "test_percentage": test_percentage,
            "metrics": metrics,
            "feature_preview": feature_columns[:8],
        }
        return render_home(result=result)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        return render_home(error=str(error), status=400)
    except OSError as error:
        app.logger.exception("Could not save the trained fraud model or results.")
        return render_home(
            error=f"Could not save the training results: {error}",
            status=500,
        )

@app.post("/predict")
def predict():
    uploaded_file = request.files.get("transactions")
    if uploaded_file is None or not uploaded_file.filename:
        return render_home(error="Choose a CSV of transactions to predict.", status=400)
    if Path(uploaded_file.filename).suffix.lower() != ".csv":
        return render_home(error="Upload transactions in CSV format.", status=400)
    if not MODEL_PATH.is_file():
        return render_home(
            error="Train a model first, then upload transactions to predict.",
            status=400,
        )

    try:
        transactions = pd.read_csv(uploaded_file.stream)
        saved_model = joblib.load(MODEL_PATH)
        predictions = predict_transactions(
            saved_model["model"],
            transactions,
            saved_model["feature_columns"],
        )
        predictions.to_csv(PREDICTIONS_PATH, index=False)
        result = {
            "prediction_count": len(predictions),
            "predictions": predictions.head(20).to_html(
                index=False,
                classes="data-table",
                border=0,
                float_format=lambda value: f"{value:.4f}",
            ),
        }
        return render_home(result=result)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        return render_home(
            error=f"Could not predict these transactions: {error}",
            status=400,
        )
    except (OSError, KeyError, EOFError, ModuleNotFoundError) as error:
        app.logger.exception("Could not load the trained fraud model.")
        return render_home(error=f"Could not load the trained model: {error}", status=500)

@app.post("/predict-manual")
def predict_manual():
    if not MODEL_PATH.is_file():
        return render_home(
            error="Train the model first, then enter transaction values.",
            status=400,
        )

    try:
        saved = joblib.load(MODEL_PATH)
        model = saved["model"]
        feature_columns = saved["feature_columns"]

        values = {}
        for column in feature_columns:
            raw_value = request.form.get(column, "").strip()
            if not raw_value:
                raise ValueError(f"Enter a value for {column}.")
            values[column] = float(raw_value)

        transaction = pd.DataFrame(
            [[values[col] for col in feature_columns]],
            columns=feature_columns,
        )

        prediction = int(model.predict(transaction)[0])
        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(transaction)[0]
            probability = float(probabilities[1])

        result = {
            "manual_prediction": (
                "FRAUDULENT TRANSACTION"
                if prediction == 1
                else "LEGITIMATE TRANSACTION"
            ),
            "fraud_probability": probability,
        }
        return render_home(result=result)

    except (ValueError, TypeError) as error:
        return render_home(error=f"Check your input values: {error}", status=400)
    except (OSError, KeyError, EOFError, ModuleNotFoundError) as error:
        app.logger.exception("Could not load the trained model.")
        return render_home(error=f"Could not load the model: {error}", status=500)

@app.get("/download/model")
def download_model():
    if not MODEL_PATH.is_file():
        return render_home(
            error="There is no trained model to download.",
            status=404,
        )
    return send_file(
        MODEL_PATH,
        as_attachment=True,
        download_name="credit_card_fraud_model.joblib",
    )


@app.get("/download/test-predictions")
def download_test_predictions():
    if not TEST_PREDICTIONS_PATH.is_file():
        return render_home(
            error="Train a model before downloading test predictions.",
            status=404,
        )
    return send_file(
        TEST_PREDICTIONS_PATH,
        as_attachment=True,
        download_name="fraud_test_predictions.csv",
    )



@app.get("/download/predictions")
def download_predictions():
    if not PREDICTIONS_PATH.is_file():
        return render_home(
            error="Upload transactions for prediction before downloading.",
            status=404,
        )
    return send_file(
        PREDICTIONS_PATH,
        as_attachment=True,
        download_name="fraud_predictions.csv",
    )


@app.errorhandler(413)
def upload_too_large(_error):
    return render_home(
        error="The CSV exceeds the 256 MB upload limit.",
        status=413,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

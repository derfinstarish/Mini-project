# Credit Card Fraud Detection

A local HTML website served by Python for training and evaluating a
logistic-regression model using a labeled Kaggle credit-card transactions CSV.

## 1. Install dependencies

Open PowerShell in this folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 2. Start the website

```powershell
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. The app
serves a regular HTML website. When `creditcard.csv` is present, the first visit
automatically trains a model if needed and displays a sample transaction
prediction. No code edits or manual feature entry are required to see the demo.

## Publish a public website on Render

The GitHub repository stores the source code; Render runs the Flask website and
provides its public URL.

1. Sign in to [Render](https://dashboard.render.com/) and choose **New**
   then **Blueprint**.
2. Connect `derfinstarish/Mini-project` and deploy the Blueprint from the
   `main` branch. Render reads `render.yaml` to install dependencies and start
   the website.
3. When the deployment finishes, open the `onrender.com` URL shown in the
   Render service dashboard. Anyone with the link can open the website.

The included Blueprint uses Render's free web-service plan. Free services can
spin down while idle, so the first visit after inactivity may take a little
longer. The service uses temporary storage; a generated model may need to be
trained again after a restart. Subsequent pushes to `main` can be deployed
automatically by Render.

## Train a model

The website uses the local `creditcard.csv` automatically if it is in this
folder and no training file is uploaded. To use a different labeled dataset,
upload its CSV, enter the target column name and the value representing fraud,
select a test-set percentage, and select **Train and evaluate**.

The commonly used Kaggle credit-card dataset has a `Class` target column, where
`0` means legitimate and `1` means fraud. The model trains on numeric feature
columns, scales their values, and balances the classes. Evaluation metrics
(precision, recall, F1 score, ROC-AUC, average precision, and a confusion
matrix) are calculated on the held-out test split. The downloaded final model
is then trained on all rows.

You can also train directly from the local `creditcard.csv` without opening the
website:

```powershell
python train_model.py
```

## Predict new transactions

After training, upload an unlabeled transactions CSV with the same numeric
feature columns as the training dataset. The website shows fraud predictions
and probabilities and lets you download the results. The home page also
automatically predicts the first row of the model's training dataset as a demo.

The saved model is `credit_card_fraud_model.joblib`; the website also lets you
download it and the held-out test predictions. CSV uploads are limited to
256 MB.

This model is for a learning project, not real financial decisions. Evaluation
results depend on the dataset and do not guarantee performance on future
transactions.

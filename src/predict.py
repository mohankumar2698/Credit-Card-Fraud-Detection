import joblib
import pandas as pd

from config import MODEL_PATH, SCALER_PATH

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def predict_transaction(data):
    """
    Predict whether a transaction is Fraud or Legitimate.

    Parameters
    ----------
    data : dict
        Dictionary with feature names matching the training data.
    """

    df = pd.DataFrame([data])

    if "Amount" in df.columns:
        df["Amount"] = scaler.transform(df[["Amount"]])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    label = "Fraud" if prediction == 1 else "Legitimate"

    return label, probability

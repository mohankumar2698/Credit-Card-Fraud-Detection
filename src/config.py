import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model.pkl")

SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

ASSETS_PATH = os.path.join(BASE_DIR, "assets")

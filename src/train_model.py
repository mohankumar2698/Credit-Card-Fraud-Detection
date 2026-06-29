import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from preprocess import (
    load_data,
    preprocess_data,
    split_data,
    balance_data,
)

from config import DATA_PATH, MODEL_PATH, SCALER_PATH

# Load data
print("Loading dataset...")
df = load_data(DATA_PATH)

# Preprocess
print("Preprocessing...")
X, y, scaler = preprocess_data(df)

# Split
X_train, X_test, y_train, y_test = split_data(X, y)

# Balance
print("Applying SMOTE...")
X_train, y_train = balance_data(X_train, y_train)

# Train model
print("Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print("\n===== MODEL PERFORMANCE =====")

print("Accuracy :", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall   :", recall_score(y_test, pred))
print("F1 Score :", f1_score(y_test, pred))
print("ROC AUC  :", roc_auc_score(y_test, prob))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, pred))

# Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\nModel saved successfully!")

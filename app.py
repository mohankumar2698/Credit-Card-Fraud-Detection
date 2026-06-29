import os
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    DATA_PATH,
    MODEL_PATH,
    SCALER_PATH,
)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


model, scaler = load_model()

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


df = load_dataset()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Dataset",
        "Visualizations",
        "Single Prediction",
        "Batch Prediction",
        "About"
    ]
)

# ===========================================================
# HOME
# ===========================================================
if page == "Home":

    st.title("💳 Credit Card Fraud Detection")

    st.markdown("""
This application predicts whether a credit card transaction is:

- ✅ Legitimate
- 🚨 Fraudulent

The model was trained using the Kaggle Credit Card Fraud Detection dataset.
""")

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Fraud Cases", int(df["Class"].sum()))

# ===========================================================
# DATASET
# ===========================================================
elif page == "Dataset":

    st.title("Dataset")

    st.dataframe(df.head())

    st.write("Shape:", df.shape)

    st.write("Class Distribution")

    st.write(df["Class"].value_counts())

# ===========================================================
# VISUALIZATIONS
# ===========================================================
elif page == "Visualizations":

    st.title("Visualizations")

    fig = plt.figure(figsize=(6,4))

    df["Class"].value_counts().plot(kind="bar")

    plt.title("Fraud vs Legitimate")

    st.pyplot(fig)

    st.subheader("Correlation Matrix")

    corr = df.corr()

    st.dataframe(corr)

# ===========================================================
# SINGLE PREDICTION
# ===========================================================
elif page == "Single Prediction":

    st.title("Single Transaction Prediction")

    st.info(
        "Enter values for all transaction features."
    )

    feature_columns = list(df.drop("Class", axis=1).columns)

    values = {}

    cols = st.columns(2)

    for i, feature in enumerate(feature_columns):

        with cols[i % 2]:
            values[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.6f"
            )

    if st.button("Predict"):

        sample = pd.DataFrame([values])

        if "Amount" in sample.columns:
            sample["Amount"] = scaler.transform(
                sample[["Amount"]]
            )

        pred = model.predict(sample)[0]

        prob = model.predict_proba(sample)[0][1]

        if pred == 1:
            st.error(
                f"Fraud Detected\n\nProbability: {prob:.4f}"
            )
        else:
            st.success(
                f"Legitimate Transaction\n\nProbability: {1-prob:.4f}"
            )

# ===========================================================
# BATCH PREDICTION
# ===========================================================
elif page == "Batch Prediction":

    st.title("CSV Batch Prediction")

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded is not None:

        batch = pd.read_csv(uploaded)

        if "Amount" in batch.columns:
            batch["Amount"] = scaler.transform(
                batch[["Amount"]]
            )

        prediction = model.predict(batch)

        probability = model.predict_proba(batch)[:,1]

        batch["Prediction"] = prediction

        batch["Fraud Probability"] = probability

        st.dataframe(batch.head())

        csv = batch.to_csv(index=False)

        st.download_button(
            "Download Predictions",
            csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

# ===========================================================
# ABOUT
# ===========================================================
elif page == "About":

    st.title("About")

    st.markdown("""
### Credit Card Fraud Detection

Technologies Used

- Python
- Pandas
- Scikit-Learn
- Random Forest
- SMOTE
- Streamlit

Author

**Mohan Kumar**

Portfolio Project
""")

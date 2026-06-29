import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(path):
    """Load the credit card dataset."""
    return pd.read_csv(path)


def preprocess_data(df):
    """
    Prepare features and target.
    Scale the Amount column while keeping PCA features unchanged.
    """
    df = df.copy()

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()

    if "Amount" in X.columns:
        X["Amount"] = scaler.fit_transform(X[["Amount"]])

    return X, y, scaler


def split_data(X, y):
    """Split the dataset into train and test sets."""
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def balance_data(X_train, y_train):
    """Apply SMOTE to balance the training data."""
    smote = SMOTE(random_state=42)

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    return X_resampled, y_resampled

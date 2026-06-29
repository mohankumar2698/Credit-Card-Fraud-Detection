import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
)

from config import ASSETS_PATH


def save_confusion_matrix(y_true, y_pred):
    os.makedirs(ASSETS_PATH, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            ASSETS_PATH,
            "confusion_matrix.png",
        )
    )

    plt.close()


def save_roc_curve(y_true, probabilities):
    os.makedirs(ASSETS_PATH, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, probabilities)

    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.4f}",
    )

    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            ASSETS_PATH,
            "roc_curve.png",
        )
    )

    plt.close()

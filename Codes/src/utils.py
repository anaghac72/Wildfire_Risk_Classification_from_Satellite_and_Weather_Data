"""
utils.py — Utility Functions
Project: Wildfire Risk Classification from Satellite and Weather Data
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ─────────────────────────────────────────────────────────────
# Dataset Columns
# ─────────────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "Temperature",
    "RH",
    "Ws",
    "FFMC",
    "DC",
    "ISI",
    "BUI",
    "FWI",
]

TARGET_COLUMN = "Classes"

# ─────────────────────────────────────────────────────────────
# Model File Names
# ─────────────────────────────────────────────────────────────

MODEL_FILES = {
    "SVM": "svm_model.pkl",
    "Random Forest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl",
}

# ─────────────────────────────────────────────────────────────
# Compute Evaluation Metrics
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics.
    """

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1 Score": f1_score(y_true, y_pred, average="weighted"),
    }

    return metrics


# ─────────────────────────────────────────────────────────────
# Assign Risk Levels
# ─────────────────────────────────────────────────────────────

def assign_risk_levels(probabilities):
    """
    Convert probability values into risk labels.
    """

    risk_levels = []

    for prob in probabilities:

        if prob >= 0.80:
            risk_levels.append("High")

        elif prob >= 0.50:
            risk_levels.append("Medium")

        else:
            risk_levels.append("Low")

    return risk_levels


# ─────────────────────────────────────────────────────────────
# Confusion Matrix
# ─────────────────────────────────────────────────────────────

def get_confusion_matrix(y_true, y_pred):
    """
    Generate confusion matrix.
    """

    cm = confusion_matrix(y_true, y_pred)

    return pd.DataFrame(
        cm,
        index=["Actual No Fire", "Actual Fire"],
        columns=["Predicted No Fire", "Predicted Fire"],
    )


# ─────────────────────────────────────────────────────────────
# Classification Report
# ─────────────────────────────────────────────────────────────

def get_classification_report(y_true, y_pred):
    """
    Return classification report as dictionary.
    """

    return classification_report(
        y_true,
        y_pred,
        output_dict=True,
    )


# ─────────────────────────────────────────────────────────────
# Load Dataset Helper
# ─────────────────────────────────────────────────────────────

def load_dataset(csv_path):
    """
    Load wildfire dataset.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    return df


# ─────────────────────────────────────────────────────────────
# Preprocess Input
# ─────────────────────────────────────────────────────────────

def preprocess_input(input_data):
    """
    Convert user input dictionary into dataframe.
    """

    df = pd.DataFrame([input_data])

    return df[FEATURE_COLUMNS]


# ─────────────────────────────────────────────────────────────
# Risk Color Mapping
# ─────────────────────────────────────────────────────────────

def get_risk_color(risk_level):
    """
    Return color for risk label.
    """

    colors = {
        "Low": "#22C55E",
        "Medium": "#F59E0B",
        "High": "#EF4444",
    }

    return colors.get(risk_level, "#6B7280")

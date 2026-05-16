"""
utils.py — Utility Functions
Project: Wildfire Risk Classification from Satellite and Weather Data
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ─────────────────────────────────────────────────────────────
# Feature and Target Columns
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
# Compute Metrics
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """
    Compute classification evaluation metrics.
    """

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),

        "Precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),

        "F1 Score": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }

    return metrics

# ─────────────────────────────────────────────────────────────
# Assign Risk Levels
# ─────────────────────────────────────────────────────────────

def assign_risk_levels(probability):
    """
    Convert prediction probability into wildfire risk level.
    """

    if probability >= 0.80:
        return "High"

    elif probability >= 0.50:
        return "Medium"

    return "Low"

# ─────────────────────────────────────────────────────────────
# Confusion Matrix
# ─────────────────────────────────────────────────────────────

def get_confusion_matrix(y_true, y_pred):
    """
    Generate confusion matrix.
    """

    return confusion_matrix(y_true, y_pred)

# ─────────────────────────────────────────────────────────────
# Prepare Input Data
# ─────────────────────────────────────────────────────────────

def prepare_input_data(input_values):
    """
    Convert input dictionary into DataFrame.
    """

    df = pd.DataFrame([input_values])

    return df[FEATURE_COLUMNS]

# ─────────────────────────────────────────────────────────────
# Prediction Helper
# ─────────────────────────────────────────────────────────────

def predict_fire_risk(model, input_df):
    """
    Predict wildfire risk and probabilities.
    """

    prediction = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
    else:
        probabilities = np.array([1 - prediction, prediction])

    return prediction, probabilities

# ─────────────────────────────────────────────────────────────
# Risk Label Helper
# ─────────────────────────────────────────────────────────────

def get_risk_label(probability):
    """
    Return readable wildfire risk label.
    """

    if probability >= 0.80:
        return "🔴 High Risk"

    elif probability >= 0.50:
        return "🟠 Medium Risk"

    return "🟢 Low Risk"

# ─────────────────────────────────────────────────────────────
# Feature Display Names
# ─────────────────────────────────────────────────────────────

FEATURE_DISPLAY_NAMES = {
    "Temperature": "Temperature",
    "RH": "Relative Humidity",
    "Ws": "Wind Speed",
    "FFMC": "Fine Fuel Moisture Code",
    "DC": "Drought Code",
    "ISI": "Initial Spread Index",
    "BUI": "Build Up Index",
    "FWI": "Fire Weather Index",
}

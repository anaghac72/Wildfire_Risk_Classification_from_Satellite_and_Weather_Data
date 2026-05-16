import numpy as np
import joblib

# Load model and scaler
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")


def get_expected_features():
    """
    Returns number of features model was trained on
    """
    try:
        return scaler.n_features_in_
    except:
        return model.n_features_in_


def preprocess_input(data):
    """
    Converts input list → model-ready scaled numpy array
    """
    arr = np.array(data, dtype=float).reshape(1, -1)

    expected_features = get_expected_features()

    # 🚨 SAFE CHECK (prevents Streamlit crash)
    if arr.shape[1] != expected_features:
        raise ValueError(
            f"❌ Feature mismatch!\n"
            f"Model expects: {expected_features} features\n"
            f"But received: {arr.shape[1]} features\n\n"
            f"👉 Fix: Update Streamlit input features to match training data."
        )

    return scaler.transform(arr)


def predict_risk(data):
    """
    Returns wildfire risk prediction label
    """
    try:
        processed = preprocess_input(data)
        prediction = model.predict(processed)[0]

        # Map predictions safely
        if prediction == 0:
            return "Low Risk 🔵"
        elif prediction == 1:
            return "Medium Risk 🟡"
        else:
            return "High Risk 🔴"

    except Exception as e:
        return f"Error: {str(e)}"

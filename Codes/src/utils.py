import numpy as np
import joblib

# Load model and scaler
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

def preprocess_input(data):
    """
    Convert input into model-ready format
    """
    arr = np.array(data).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    return arr_scaled


def predict_risk(data):
    processed = preprocess_input(data)
    prediction = model.predict(processed)[0]

    if prediction == 0:
        return "Low Risk 🔵"
    elif prediction == 1:
        return "Medium Risk 🟡"
    else:
        return "High Risk 🔴"

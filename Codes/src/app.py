import streamlit as st
import numpy as np
from utils import predict_risk, preprocess_input, model

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Wildfire Risk Classification",
    layout="centered"
)

st.title("🔥 Wildfire Risk Classification System")
st.markdown("ML-based prediction using satellite + weather features")

# =========================
# INPUT SECTION (10 FEATURES)
# =========================
st.sidebar.header("🌦️ Input Parameters")

temperature = st.sidebar.slider("Temperature (°C)", 0, 50, 25)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 40)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0, 100, 20)
rainfall = st.sidebar.slider("Rainfall (mm)", 0, 200, 10)

ndvi = st.sidebar.slider("NDVI", 0.0, 1.0, 0.5)
elevation = st.sidebar.slider("Elevation (m)", 0, 3000, 500)
soil_moisture = st.sidebar.slider("Soil Moisture", 0.0, 1.0, 0.3)
surface_temp = st.sidebar.slider("Surface Temperature (°C)", 0, 60, 30)
vegetation_index = st.sidebar.slider("Vegetation Index", 0.0, 1.0, 0.6)
fire_risk_index = st.sidebar.slider("Fire Risk Index", 0.0, 1.0, 0.2)

# Input vector
input_data = [
    temperature,
    humidity,
    wind_speed,
    rainfall,
    ndvi,
    elevation,
    soil_moisture,
    surface_temp,
    vegetation_index,
    fire_risk_index
]

# =========================
# PREDICTION SECTION
# =========================
st.markdown("---")

if st.button("🔥 Predict Wildfire Risk"):

    processed = preprocess_input(input_data)

    # Predict class
    prediction = model.predict(processed)[0]

    # Predict probabilities
    probabilities = model.predict_proba(processed)[0]
    classes = model.classes_

    # =========================
    # RESULT
    # =========================
    st.subheader("📌 Final Prediction")

    if prediction == 0 or "Low" in str(prediction):
        st.success(f"Low Risk 🔵 ({prediction})")
    elif prediction == 1 or "Medium" in str(prediction):
        st.warning(f"Medium Risk 🟡 ({prediction})")
    else:
        st.error(f"High Risk 🔴 ({prediction})")

    # =========================
    # PROBABILITY DISPLAY
    # =========================
    st.subheader("📊 Prediction Probabilities")

    for cls, prob in zip(classes, probabilities):
        st.write(f"**{cls}** → {round(prob * 100, 2)}%")

    # Risk confidence meter
    confidence = np.max(probabilities)
    st.progress(float(confidence))
    st.write(f"Confidence Score: {round(confidence * 100, 2)}%")

# =========================
# INPUT SUMMARY
# =========================
st.markdown("---")
st.subheader("📊 Input Summary")

st.json({
    "Temperature": temperature,
    "Humidity": humidity,
    "Wind Speed": wind_speed,
    "Rainfall": rainfall,
    "NDVI": ndvi,
    "Elevation": elevation,
    "Soil Moisture": soil_moisture,
    "Surface Temp": surface_temp,
    "Vegetation Index": vegetation_index,
    "Fire Risk Index": fire_risk_index
})

# =========================
# DEBUG MODE
# =========================
st.markdown("---")

if st.checkbox("🛠 Debug Mode"):
    st.write("Raw Input:", input_data)
    st.write("Scaled Input:", preprocess_input(input_data))
    st.write("Model Classes:", model.classes_)

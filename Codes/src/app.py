import streamlit as st
import numpy as np
from utils import predict_risk

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Wildfire Risk Classification",
    layout="centered"
)

st.title("🔥 Wildfire Risk Classification System")
st.markdown("Predict wildfire risk using satellite + weather features")

# =========================
# SIDEBAR INPUTS (10 FEATURES)
# =========================
st.sidebar.header("🌦️ Input Parameters")

temperature = st.sidebar.slider("Temperature (°C)", 0, 50, 25)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 40)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0, 100, 20)
rainfall = st.sidebar.slider("Rainfall (mm)", 0, 200, 10)

ndvi = st.sidebar.slider("NDVI (Vegetation Index)", 0.0, 1.0, 0.5)
elevation = st.sidebar.slider("Elevation (m)", 0, 3000, 500)
soil_moisture = st.sidebar.slider("Soil Moisture", 0.0, 1.0, 0.3)
surface_temp = st.sidebar.slider("Surface Temperature (°C)", 0, 60, 30)

vegetation_index = st.sidebar.slider("Vegetation Index", 0.0, 1.0, 0.6)
fire_risk_index = st.sidebar.slider("Fire Risk Index", 0.0, 1.0, 0.2)

# =========================
# INPUT VECTOR (MUST BE 10 FEATURES)
# =========================
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
# PREDICTION
# =========================
st.markdown("---")

if st.button("🔥 Predict Wildfire Risk"):
    result = predict_risk(input_data)

    st.subheader("Prediction Result:")

    if "Low" in result:
        st.success(result)
    elif "Medium" in result:
        st.warning(result)
    elif "High" in result:
        st.error(result)
    else:
        st.error(result)

# =========================
# INPUT DISPLAY
# =========================
st.markdown("---")
st.subheader("📊 Input Summary")

st.write({
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
# INFO SECTION
# =========================
st.markdown("---")
st.info(
    "💡 Model expects 10 features exactly. "
    "All inputs must match training dataset used in notebook."
)

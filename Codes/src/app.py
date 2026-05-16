import streamlit as st
import numpy as np
from utils import predict_risk

# Page config
st.set_page_config(page_title="Wildfire Risk App", layout="centered")

# Title
st.title("🔥 Wildfire Risk Classification System")
st.markdown("Predict wildfire risk using satellite + weather parameters")

st.sidebar.header("Input Parameters")

# Example features (you can modify based on your dataset)
temperature = st.sidebar.slider("Temperature (°C)", 0, 50, 25)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 40)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0, 100, 20)
rainfall = st.sidebar.slider("Rainfall (mm)", 0, 200, 10)
vegetation_index = st.sidebar.slider("Vegetation Index (NDVI)", 0.0, 1.0, 0.5)

# Input vector
input_data = [
    temperature,
    humidity,
    wind_speed,
    rainfall,
    vegetation_index
]

# Prediction button
if st.button("Predict Wildfire Risk"):
    result = predict_risk(input_data)

    st.subheader("Prediction Result:")
    st.success(result)

# Optional visualization section
st.markdown("---")
st.subheader("📊 Input Overview")

st.write({
    "Temperature": temperature,
    "Humidity": humidity,
    "Wind Speed": wind_speed,
    "Rainfall": rainfall,
    "Vegetation Index": vegetation_index
})

# Simple gauge-style interpretation
st.markdown("### Risk Interpretation")
st.info("Higher wind + temperature and lower humidity increases wildfire risk.")

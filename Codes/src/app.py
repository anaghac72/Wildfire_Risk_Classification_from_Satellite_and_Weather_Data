"""
app.py — Streamlit Web Application
Project: Wildfire Risk Classification from Satellite and Weather Data
"""

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from utils import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    MODEL_FILES,
    compute_metrics,
    assign_risk_levels,
    get_confusion_matrix,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Wildfire Risk Classifier",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #F59E0B, #EF4444);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}

.sub-header {
    text-align: center;
    color: #6B7280;
    margin-bottom: 25px;
}

.metric-card {
    background: linear-gradient(135deg, #1F2937, #374151);
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #F59E0B;
}

.metric-label {
    font-size: 1rem;
    color: #D1D5DB;
}

.risk-low {
    color: #22C55E;
    font-size: 1.8rem;
    font-weight: bold;
}

.risk-medium {
    color: #F59E0B;
    font-size: 1.8rem;
    font-weight: bold;
}

.risk-high {
    color: #EF4444;
    font-size: 1.8rem;
    font-weight: bold;
}

.stButton > button {
    background: linear-gradient(90deg, #F59E0B, #EF4444);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ─────────────────────────────────────────────────────────────
# Load Models
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    models = {}

    for model_name, model_file in MODEL_FILES.items():
        model_path = os.path.join(MODELS_DIR, model_file)

        if os.path.exists(model_path):
            try:
                models[model_name] = joblib.load(model_path)
            except Exception as e:
                st.error(f"Error loading {model_name}: {e}")

    return models

models = load_models()

# ─────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────

st.sidebar.title("🔥 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📊 Predict",
        "📈 Visualizations",
        "📋 About",
    ]
)

# ─────────────────────────────────────────────────────────────
# Home Page
# ─────────────────────────────────────────────────────────────

if page == "🏠 Home":

    st.markdown(
        '<h1 class="main-header">🔥 Wildfire Risk Classification</h1>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="sub-header">Predict wildfire risk using ML models and weather data</p>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">3</div>
            <div class="metric-label">ML Models</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">8</div>
            <div class="metric-label">Features</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">243</div>
            <div class="metric-label">Dataset Samples</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("## 📌 Project Overview")

    st.write("""
    This application predicts wildfire risk using machine learning models.

    ### Models Used
    - SVM
    - Random Forest
    - XGBoost

    ### Features Used
    - Temperature
    - Humidity
    - Wind Speed
    - FFMC
    - DC
    - ISI
    - BUI
    - FWI
    """)

# ─────────────────────────────────────────────────────────────
# Prediction Page
# ─────────────────────────────────────────────────────────────

elif page == "📊 Predict":

    st.title("📊 Wildfire Risk Prediction")

    st.markdown("Enter environmental parameters below:")

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.slider("Temperature", 0, 50, 30)
        rh = st.slider("Humidity (RH)", 0, 100, 45)
        ws = st.slider("Wind Speed (Ws)", 0, 50, 15)
        ffmc = st.slider("FFMC", 0.0, 100.0, 85.0)

    with col2:
        dc = st.slider("DC", 0.0, 1000.0, 400.0)
        isi = st.slider("ISI", 0.0, 50.0, 10.0)
        bui = st.slider("BUI", 0.0, 200.0, 60.0)
        fwi = st.slider("FWI", 0.0, 50.0, 15.0)

    selected_model = st.selectbox(
        "Select Model",
        list(models.keys())
    )

    if st.button("Predict Wildfire Risk"):

        if selected_model not in models:
            st.error("Selected model not available.")
        else:

            input_data = pd.DataFrame([[
                temperature,
                rh,
                ws,
                ffmc,
                dc,
                isi,
                bui,
                fwi
            ]], columns=FEATURE_COLUMNS)

            model = models[selected_model]

            prediction = model.predict(input_data)[0]

            probability = 0.0

            if hasattr(model, "predict_proba"):
                probability = np.max(model.predict_proba(input_data))

            risk = assign_risk_levels(probability)

            st.markdown("---")

            st.subheader("Prediction Result")

            if risk == "Low":
                st.markdown(
                    '<p class="risk-low">🟢 LOW RISK</p>',
                    unsafe_allow_html=True
                )

            elif risk == "Medium":
                st.markdown(
                    '<p class="risk-medium">🟠 MEDIUM RISK</p>',
                    unsafe_allow_html=True
                )

            else:
                st.markdown(
                    '<p class="risk-high">🔴 HIGH RISK</p>',
                    unsafe_allow_html=True
                )

            st.write(f"### Confidence: {probability * 100:.2f}%")

# ─────────────────────────────────────────────────────────────
# Visualizations Page
# ─────────────────────────────────────────────────────────────

elif page == "📈 Visualizations":

    st.title("📈 Data Visualizations")

    st.subheader("Feature Importance Example")

    features = FEATURE_COLUMNS
    importance = np.random.rand(len(features))

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x=importance,
        y=features,
        ax=ax
    )

    ax.set_title("Feature Importance")

    st.pyplot(fig)

# ─────────────────────────────────────────────────────────────
# About Page
# ─────────────────────────────────────────────────────────────

elif page == "📋 About":

    st.title("📋 About Project")

    st.write("""
    ## Wildfire Risk Classification System

    This project uses Machine Learning algorithms to classify wildfire risk
    based on satellite and weather data.

    ### Technologies Used
    - Python
    - Streamlit
    - Scikit-learn
    - XGBoost
    - Matplotlib
    - Seaborn

    ### Dataset
    Algerian Forest Fire Dataset

    ### Models
    - Support Vector Machine
    - Random Forest
    - XGBoost
    """)

    st.success("Application running successfully 🚀")

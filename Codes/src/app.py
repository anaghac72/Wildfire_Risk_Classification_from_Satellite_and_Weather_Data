"""
app.py — Streamlit Web Application
Project: Wildfire Risk Classification from Satellite and Weather Data
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ─────────────────────────────────────────────────────────────
# Project Path
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import (
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
    text-align: center;
    background: linear-gradient(90deg, #F59E0B, #EF4444);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub-header {
    text-align: center;
    color: #9CA3AF;
    font-size: 1.1rem;
    margin-bottom: 20px;
}

.metric-card {
    background: linear-gradient(135deg, #1F2937, #374151);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin: 10px 0;
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #F59E0B;
}

.metric-label {
    color: #D1D5DB;
    font-size: 0.9rem;
}

.risk-low {
    color: #22C55E;
    font-size: 2rem;
    font-weight: bold;
}

.risk-medium {
    color: #F59E0B;
    font-size: 2rem;
    font-weight: bold;
}

.risk-high {
    color: #EF4444;
    font-size: 2rem;
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
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ─────────────────────────────────────────────────────────────
# Load Models
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():

    models = {}

    for name, file_name in MODEL_FILES.items():

        model_path = os.path.join(MODELS_DIR, file_name)

        if os.path.exists(model_path):

            try:
                models[name] = joblib.load(model_path)

            except Exception as e:
                st.error(f"Error loading {name}: {e}")

    return models

# ─────────────────────────────────────────────────────────────
# Risk Label Function
# ─────────────────────────────────────────────────────────────
def get_risk_label(prediction, probabilities=None):

    if prediction == 1:

        confidence = (
            probabilities[1] * 100
            if probabilities is not None
            else 0
        )

        if confidence >= 80:
            return "High", confidence

        return "Medium", confidence

    confidence = (
        probabilities[0] * 100
        if probabilities is not None
        else 100
    )

    return "Low", confidence

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔥 Navigation")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Predict",
        "📈 Visualizations",
        "📋 About",
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Wildfire Risk Classifier v1.0</small>",
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════
# HOME PAGE
# ═════════════════════════════════════════════════════════════
if page == "🏠 Home":

    st.markdown(
        '<h1 class="main-header">🔥 Wildfire Risk Classification</h1>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="sub-header">Machine Learning based wildfire risk prediction using weather and satellite data</p>',
        unsafe_allow_html=True,
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

    st.subheader("📌 Project Overview")

    st.markdown("""
    This application predicts wildfire risk using machine learning models.

    ### Models Used
    - Support Vector Machine (SVM)
    - Random Forest
    - XGBoost

    ### Features
    - Temperature
    - Relative Humidity
    - Wind Speed
    - FFMC
    - DC
    - ISI
    - BUI
    - FWI
    """)

# ═════════════════════════════════════════════════════════════
# PREDICT PAGE
# ═════════════════════════════════════════════════════════════
elif page == "📊 Predict":

    st.markdown(
        '<h1 class="main-header">📊 Wildfire Risk Prediction</h1>',
        unsafe_allow_html=True,
    )

    models = load_models()

    st.write("Loaded Models:", list(models.keys()))

    if len(models) == 0:
        st.error("No trained models found in models/ directory.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:

        temperature = st.slider(
            "Temperature",
            min_value=0,
            max_value=50,
            value=30,
        )

        rh = st.slider(
            "Relative Humidity (RH)",
            min_value=0,
            max_value=100,
            value=40,
        )

        ws = st.slider(
            "Wind Speed (Ws)",
            min_value=0,
            max_value=50,
            value=15,
        )

        ffmc = st.slider(
            "FFMC",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
        )

    with col2:

        dc = st.slider(
            "DC",
            min_value=0.0,
            max_value=1000.0,
            value=200.0,
        )

        isi = st.slider(
            "ISI",
            min_value=0.0,
            max_value=50.0,
            value=10.0,
        )

        bui = st.slider(
            "BUI",
            min_value=0.0,
            max_value=500.0,
            value=100.0,
        )

        fwi = st.slider(
            "FWI",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
        )

    selected_model = st.selectbox(
        "Select ML Model",
        list(models.keys())
    )

    input_data = np.array([[
        temperature,
        rh,
        ws,
        ffmc,
        dc,
        isi,
        bui,
        fwi,
    ]])

    if st.button("🔥 Predict Risk"):

        try:

            model = models[selected_model]

            prediction = model.predict(input_data)[0]

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
            else:
                probabilities = [1 - prediction, prediction]

            risk, confidence = get_risk_label(
                prediction,
                probabilities
            )

            st.markdown("---")

            if risk == "Low":
                st.markdown(
                    f'<p class="risk-low">🟢 {risk} Risk</p>',
                    unsafe_allow_html=True,
                )

            elif risk == "Medium":
                st.markdown(
                    f'<p class="risk-medium">🟠 {risk} Risk</p>',
                    unsafe_allow_html=True,
                )

            else:
                st.markdown(
                    f'<p class="risk-high">🔴 {risk} Risk</p>',
                    unsafe_allow_html=True,
                )

            st.metric("Confidence", f"{confidence:.2f}%")

            st.subheader("📄 Input Summary")

            input_df = pd.DataFrame(
                input_data,
                columns=FEATURE_COLUMNS
            )

            st.dataframe(input_df)

        except Exception as e:

            st.error(f"Prediction Error: {e}")

# ═════════════════════════════════════════════════════════════
# VISUALIZATION PAGE
# ═════════════════════════════════════════════════════════════
elif page == "📈 Visualizations":

    st.markdown(
        '<h1 class="main-header">📈 Data Visualizations</h1>',
        unsafe_allow_html=True,
    )

    dataset_path = os.path.join(DATA_DIR, "wildfire.csv")

    if os.path.exists(dataset_path):

        try:

            df = pd.read_csv(dataset_path)

            st.subheader("Dataset Preview")

            st.dataframe(df.head())

            st.subheader("Feature Correlation Heatmap")

            fig, ax = plt.subplots(figsize=(10, 6))

            sns.heatmap(
                df.corr(numeric_only=True),
                annot=True,
                cmap="coolwarm",
                ax=ax,
            )

            st.pyplot(fig)

            st.subheader("Temperature Distribution")

            fig2, ax2 = plt.subplots(figsize=(8, 5))

            sns.histplot(
                df["Temperature"],
                kde=True,
                ax=ax2,
            )

            st.pyplot(fig2)

        except Exception as e:

            st.error(f"Visualization Error: {e}")

    else:

        st.warning("Dataset file not found.")

# ═════════════════════════════════════════════════════════════
# ABOUT PAGE
# ═════════════════════════════════════════════════════════════
elif page == "📋 About":

    st.markdown(
        '<h1 class="main-header">📋 About Project</h1>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    ## Wildfire Risk Classification System

    This project predicts wildfire risk using machine learning algorithms
    trained on weather and environmental conditions.

    ### Technologies Used
    - Python
    - Streamlit
    - Scikit-learn
    - XGBoost
    - Pandas
    - NumPy
    - Matplotlib
    - Seaborn

    ### Dataset
    Algerian Forest Fire Dataset

    ### Objective
    To help identify wildfire-prone conditions early using AI models.
    """)

    st.success("Project developed using Machine Learning and Streamlit 🚀")

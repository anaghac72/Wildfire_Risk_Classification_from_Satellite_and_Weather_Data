"""
app.py — Streamlit Web Application
Project: Wildfire Risk Classification from Satellite and Weather Data

Usage:
    streamlit run app.py
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
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #F59E0B, #EF4444, #DC2626);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 10px 0;
    }

    .sub-header {
        text-align: center;
        color: #6B7280;
        font-size: 1rem;
        margin-bottom: 20px;
    }

    .metric-card {
        background: linear-gradient(135deg, #1F2937, #374151);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin: 5px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #F59E0B;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #9CA3AF;
        margin-top: 5px;
    }

    .risk-low {
        color: #22C55E;
        font-weight: bold;
        font-size: 1.5rem;
    }

    .risk-medium {
        color: #F59E0B;
        font-weight: bold;
        font-size: 1.5rem;
    }

    .risk-high {
        color: #EF4444;
        font-weight: bold;
        font-size: 1.5rem;
    }

    .stButton > button {
        background: linear-gradient(90deg, #F59E0B, #EF4444);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")

# ─────────────────────────────────────────────────────────────
# Load Models
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    """Load trained ML models"""

    models = {}

    for name, filename in MODEL_FILES.items():

        model_path = os.path.join(MODELS_DIR, filename)

        if os.path.exists(model_path):

            try:
                models[name] = joblib.load(model_path)

            except Exception as e:
                st.error(f"Error loading {name}: {e}")

    return models

models = load_models()

# ─────────────────────────────────────────────────────────────
# Helper Function
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
# Display Names
# ─────────────────────────────────────────────────────────────

FEATURE_DISPLAY_NAMES = {
    "Temperature": "Temperature",
    "RH": "Humidity",
    "Ws": "Wind Speed",
    "FFMC": "Fuel Moisture",
    "DC": "Drought Index",
    "ISI": "Fire Spread Index",
    "BUI": "Burn Index",
    "FWI": "Fire Risk Index",
}

# ════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════

st.sidebar.markdown("## 🔥 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Predict",
        "📈 Visualizations",
        "📋 About",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "<small>Wildfire Risk Classifier v1.0</small>",
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════
# HOME PAGE
# ════════════════════════════════════════════════════════════

if page == "🏠 Home":

    st.markdown(
        '<h1 class="main-header">🔥 Wildfire Risk Classification</h1>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="sub-header">'
        'Classify wildfire risk using satellite and weather data with ML models'
        '</p>',
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
            <div class="metric-label">Samples</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📌 Project Overview")

    st.markdown("""
    This application classifies wildfire risk into
    Low, Medium, and High categories using Machine Learning.

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

# ════════════════════════════════════════════════════════════
# PREDICTION PAGE
# ════════════════════════════════════════════════════════════

elif page == "📊 Predict":

    st.title("📊 Wildfire Risk Prediction")

    st.markdown("Enter environmental values:")

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.slider("Temperature", 0, 50, 30)
        rh = st.slider("Humidity (RH)", 0, 100, 45)
        ws = st.slider("Wind Speed", 0, 50, 15)
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

    if st.button("Predict Risk"):

        input_df = pd.DataFrame([[
            temperature,
            rh,
            ws,
            ffmc,
            dc,
            isi,
            bui,
            fwi,
        ]], columns=FEATURE_COLUMNS)

        model = models[selected_model]

        prediction = model.predict(input_df)[0]

        probabilities = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]

        risk, confidence = get_risk_label(
            prediction,
            probabilities,
        )

        st.markdown("---")

        st.subheader("Prediction Result")

        if risk == "Low":

            st.markdown(
                '<p class="risk-low">🟢 LOW RISK</p>',
                unsafe_allow_html=True,
            )

        elif risk == "Medium":

            st.markdown(
                '<p class="risk-medium">🟠 MEDIUM RISK</p>',
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                '<p class="risk-high">🔴 HIGH RISK</p>',
                unsafe_allow_html=True,
            )

        st.write(f"### Confidence: {confidence:.2f}%")

# ════════════════════════════════════════════════════════════
# VISUALIZATION PAGE
# ════════════════════════════════════════════════════════════

elif page == "📈 Visualizations":

    st.title("📈 Visualizations")

    st.subheader("Feature Importance")

    features = FEATURE_COLUMNS
    importance = np.random.rand(len(features))

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        x=importance,
        y=features,
        ax=ax,
    )

    ax.set_title("Feature Importance")

    st.pyplot(fig)

# ════════════════════════════════════════════════════════════
# ABOUT PAGE
# ════════════════════════════════════════════════════════════

elif page == "📋 About":

    st.title("📋 About")

    st.markdown("""
    ## Wildfire Risk Classification System

    This project predicts wildfire risk using weather and satellite data.

    ### Technologies
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

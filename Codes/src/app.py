"""
app.py — Streamlit Web Application
Project: Wildfire Risk Classification from Satellite and Weather Data

A professional Streamlit app for interactive wildfire risk prediction
with real-time visualization and model comparison.

Usage:
    streamlit run src/app.py
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

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils import (
    FEATURE_COLUMNS, TARGET_COLUMN, MODEL_FILES,
    compute_metrics, assign_risk_levels, get_confusion_matrix,
)

warnings.filterwarnings("ignore")

# ── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wildfire Risk Classifier",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #F59E0B, #EF4444, #DC2626);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 10px 0;
    }
    .sub-header {
        text-align: center; color: #6B7280; font-size: 1rem; margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1F2937, #374151);
        padding: 20px; border-radius: 12px; text-align: center;
        color: white; margin: 5px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 2rem; font-weight: 700; color: #F59E0B;
    }
    .metric-label {
        font-size: 0.85rem; color: #9CA3AF; margin-top: 5px;
    }
    .risk-low    { color: #22C55E; font-weight: bold; font-size: 1.5rem; }
    .risk-medium { color: #F59E0B; font-weight: bold; font-size: 1.5rem; }
    .risk-high   { color: #EF4444; font-weight: bold; font-size: 1.5rem; }
    .stButton > button {
        background: linear-gradient(90deg, #F59E0B, #EF4444);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 2rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")


@st.cache_resource
def load_models():
    """Load all available models from disk."""
    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


def get_risk_label(prediction, probabilities=None):
    """Map binary prediction to risk string with optional confidence."""
    if prediction == 1:
        conf = probabilities[1] * 100 if probabilities is not None else 0
        if conf >= 80:
            return "High", conf
        return "Medium", conf
    return "Low", (probabilities[0] * 100 if probabilities is not None else 100)


# ── Display-friendly feature names (UI only) ──────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## 🔥 Navigation")
page = st.sidebar.radio(
    "Go to", ["🏠 Home", "📊 Predict", "📈 Visualizations", "📋 About"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Wildfire Risk Classifier v1.0</small>", unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE — HOME
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🔥 Wildfire Risk Classification</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Classify wildfire risk using satellite '
        'and weather data with ML models</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">3</div>
            <div class="metric-label">ML Models</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">8</div>
            <div class="metric-label">Features</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">243</div>
            <div class="metric-label">Samples</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📌 Project Overview")
    st.markdown("""
    This application classifies **wildfire risk** into **Low**, **Medium**,
    and **High** categories using the **Algerian Forest Fire Dataset**.

    **Models Used:**
    | Model | Description |
    |-------|-------------|
    | SVM | Support Vector Machine with RBF kernel |
    | Random Forest | Ensemble of 200 decision trees |
    | XGBoost | Gradient boosted trees (200 estimators) |

    **Key Features:** Temperature, Humidity, Wind Speed,
    Fuel Moisture, Drought Index, Fire Spread Index, Burn Index, Fire Risk Index
    """)

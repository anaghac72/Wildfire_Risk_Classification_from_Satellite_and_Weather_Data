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
    .risk-high   { color: #EF4444; font-weight: bold; font-size: 1.5rem; }
    .prediction-box {
        background: linear-gradient(135deg, #1F2937, #374151);
        padding: 30px; border-radius: 16px; text-align: center;
        color: white; margin: 15px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    .prediction-label {
        font-size: 2.5rem; font-weight: 800; margin-bottom: 8px;
    }
    .prediction-label.high { color: #EF4444; }
    .prediction-label.low  { color: #22C55E; }
    .confidence-text {
        font-size: 1.1rem; color: #9CA3AF; margin-top: 5px;
    }
    .confidence-value {
        font-size: 1.5rem; font-weight: 700; color: #F59E0B;
    }
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

# ── Raw-data statistics for z-score scaling ────────────────────────────────
# These are the mean/std of the original Algerian Forest Fire dataset features
# BEFORE standardization. Used to transform raw manual inputs into the same
# z-score space the models were trained on.
RAW_FEATURE_STATS = {
    "Temperature": {"mean": 32.1523, "std": 3.6280},
    "RH":          {"mean": 62.0412, "std": 14.8282},
    "Ws":          {"mean": 15.4938, "std": 2.8114},
    "FFMC":        {"mean": 77.8424, "std": 14.3496},
    "DC":          {"mean": 49.4309, "std": 47.6656},
    "ISI":         {"mean": 4.7424,  "std": 4.1542},
    "BUI":         {"mean": 16.6905, "std": 14.2284},
    "FWI":         {"mean": 7.0354,  "std": 7.4406},
}


def standardize_input(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw feature values to z-scores using training-set statistics."""
    scaled = raw_df.copy()
    for col in FEATURE_COLUMNS:
        stats = RAW_FEATURE_STATS[col]
        scaled[col] = (scaled[col] - stats["mean"]) / stats["std"]
    return scaled


@st.cache_resource
def load_models():
    """Load all available models from disk."""
    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


def get_prediction_label(prediction, probabilities=None):
    """Map binary prediction to risk label with confidence.
    Class 0 = High Risk (fire), Class 1 = Low Risk (no fire).
    This matches the Algerian dataset encoding: 'fire' → 0, 'not fire' → 1.
    """
    if prediction == 0:
        conf = probabilities[0] * 100 if probabilities is not None else 100
        return "High Risk 🔥", conf
    conf = probabilities[1] * 100 if probabilities is not None else 100
    return "Low Risk ✅", conf


# ── Display-friendly feature names (UI only) ──────────────────────────────
FEATURE_DISPLAY_NAMES = {
    "Temperature": "Temperature (°C)",
    "RH": "Humidity (%)",
    "Ws": "Wind Speed (km/h)",
    "FFMC": "Fuel Moisture (FFMC)",
    "DC": "Drought Index (DC)",
    "ISI": "Fire Spread Index (ISI)",
    "BUI": "Burn Index (BUI)",
    "FWI": "Fire Risk Index (FWI)",
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
    This application classifies **wildfire risk** into **High Risk** and
    **Low Risk** categories using the **Algerian Forest Fire Dataset**.

    **Models Used:**
    | Model | Description |
    |-------|-------------|
    | SVM | Support Vector Machine with RBF kernel |
    | Random Forest | Ensemble of 200 decision trees |
    | XGBoost | Gradient boosted trees (200 estimators) |

    **Key Features:** Temperature, Humidity, Wind Speed,
    Fuel Moisture, Drought Index, Fire Spread Index, Burn Index, Fire Risk Index
    """)

# ══════════════════════════════════════════════════════════════════════════
#  PAGE — PREDICT
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊 Predict":
    st.markdown('<h1 class="main-header">📊 Predict Wildfire Risk</h1>',
                unsafe_allow_html=True)

    models = load_models()
    if not models:
        st.error("⚠️ No trained models found! Run `python main.py` first.")
        st.stop()

    tab1, tab2 = st.tabs(["📁 Upload CSV", "✍️ Manual Input"])

    # ── Tab 1: CSV Upload ──────────────────────────────────────────────
    with tab1:
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is not None:
            try:
                input_df = pd.read_csv(uploaded)
                input_df.columns = input_df.columns.str.strip()
                st.success(f"✅ Loaded {input_df.shape[0]} rows")
                st.dataframe(input_df.head(), use_container_width=True)

                missing = set(FEATURE_COLUMNS) - set(input_df.columns)
                if missing:
                    st.error(f"Missing columns: {missing}")
                    st.stop()

                model_name = st.selectbox("Select Model", list(models.keys()))
                if st.button("🔥 Run Predictions", use_container_width=True):
                    model = models[model_name]
                    X_raw = input_df[FEATURE_COLUMNS]
                    X_input = standardize_input(X_raw)
                    preds = model.predict(X_input)

                    result_df = input_df.copy()
                    result_df["Prediction"] = preds
                    result_df["Prediction_Label"] = [
                        "High Risk 🔥" if p == 0 else "Low Risk ✅" for p in preds
                    ]

                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_input)
                        result_df["Confidence (%)"] = [
                            f"{max(p) * 100:.1f}" for p in probs
                        ]

                    st.markdown("### 📋 Results")
                    st.dataframe(result_df, use_container_width=True)

                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        "⬇️ Download Results CSV", csv,
                        "predictions.csv", "text/csv",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Tab 2: Manual Input ────────────────────────────────────────────
    with tab2:
        st.markdown("Enter feature values in **raw units** (the app auto-scales for the model):")
        cols = st.columns(3)
        inputs = {}
        # Sensible defaults: typical non-fire conditions
        defaults = {
            "Temperature": 30.0, "RH": 60.0, "Ws": 14.0, "FFMC": 75.0,
            "DC": 40.0, "ISI": 3.0, "BUI": 12.0, "FWI": 4.0,
        }
        for i, feat in enumerate(FEATURE_COLUMNS):
            display = FEATURE_DISPLAY_NAMES.get(feat, feat)
            with cols[i % 3]:
                inputs[feat] = st.number_input(
                    display, value=defaults.get(feat, 0.0),
                    format="%.2f", key=f"manual_{feat}",
                )

        model_name = st.selectbox("Model", list(models.keys()), key="manual")
        if st.button("🔥 Predict", key="manual_btn", use_container_width=True):
            model = models[model_name]
            raw_row = pd.DataFrame([inputs])
            scaled_row = standardize_input(raw_row)
            pred = model.predict(scaled_row)[0]

            st.markdown("---")
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(scaled_row)[0]
                label, conf = get_prediction_label(pred, prob)
                css_class = "high" if pred == 0 else "low"

                # Single clean prediction box
                st.markdown(
                    f'<div class="prediction-box">'
                    f'<div class="prediction-label {css_class}">{label}</div>'
                    f'<div class="confidence-text">Confidence</div>'
                    f'<div class="confidence-value">{conf:.1f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Probability bar chart  (index 0 = High Risk, index 1 = Low Risk)
                fig, ax = plt.subplots(figsize=(6, 1.5))
                ax.barh(["High Risk", "Low Risk"], [prob[0], prob[1]],
                        color=["#EF4444", "#22C55E"], height=0.5)
                ax.set_xlim(0, 1)
                ax.set_title("Class Probabilities", fontsize=10)
                for i, v in enumerate([prob[0], prob[1]]):
                    ax.text(v + 0.02, i, f"{v*100:.1f}%", va="center", fontsize=9)
                st.pyplot(fig)
                plt.close(fig)
            else:
                label = "High Risk 🔥" if pred == 0 else "Low Risk ✅"
                st.markdown(
                    f'<div class="prediction-box">'
                    f'<div class="prediction-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════════════════
#  PAGE — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════
elif page == "📈 Visualizations":
    st.markdown('<h1 class="main-header">📈 Visualizations</h1>',
                unsafe_allow_html=True)

    plot_files = []
    if os.path.exists(PLOTS_DIR):
        plot_files = sorted([
            f for f in os.listdir(PLOTS_DIR) if f.endswith(".png")
        ])

    if not plot_files:
        st.warning("No plots found. Run `python main.py` first to generate them.")
    else:
        for pf in plot_files:
            title = pf.replace(".png", "").replace("_", " ").title()
            st.markdown(f"### {title}")
            st.image(os.path.join(PLOTS_DIR, pf), use_container_width=True)
            st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════
#  PAGE — ABOUT
# ══════════════════════════════════════════════════════════════════════════
elif page == "📋 About":
    st.markdown('<h1 class="main-header">📋 About This Project</h1>',
                unsafe_allow_html=True)
    st.markdown("""
    ### 🔥 Wildfire Risk Classification from Satellite and Weather Data

    **Dataset:** Algerian Forest Fire Dataset
    ([UCI Repository](https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset))

    **Objective:** Classify wildfire risk into **High Risk** and **Low Risk**
    categories using meteorological and vegetation-related features.

    ---

    ### 🛠️ Technologies
    | Technology | Purpose |
    |---|---|
    | Python 3.x | Core language |
    | scikit-learn | SVM, Random Forest, metrics |
    | XGBoost | Gradient boosting classifier |
    | Matplotlib / Seaborn | Visualizations |
    | Streamlit | Interactive web UI |
    | Pandas / NumPy | Data handling |

    ---

    ### 📂 Repository Structure
    ```
    Wildfire-Risk-Classification/
    ├── data/
    ├── notebooks/
    ├── models/
    ├── outputs/
    │   ├── plots/
    │   └── predictions/
    ├── src/
    │   ├── visualizations.py
    │   ├── utils.py
    │   └── app.py
    ├── README.md
    ├── requirements.txt
    └── main.py
    ```
    """)

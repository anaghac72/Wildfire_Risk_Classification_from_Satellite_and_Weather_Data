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

from src.utils import (
    FEATURE_COLUMNS, TARGET_COLUMN, MODEL_FILES,
    compute_metrics, assign_risk_levels, get_confusion_matrix,
)

warnings.filterwarnings("ignore")

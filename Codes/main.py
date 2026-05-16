"""
main.py — Main Execution Pipeline
Project: Wildfire Risk Classification from Satellite and Weather Data

This script orchestrates the full inference + visualization pipeline:
  1. Load the cleaned dataset
  2. Train or load pre-trained models (SVM, Random Forest, XGBoost)
  3. Generate predictions on test data
  4. Compute evaluation metrics
  5. Generate all visualizations
  6. Save prediction outputs to CSV

Usage:
    python main.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    load_dataset, get_features_and_target, load_all_models,
    compute_metrics, get_classification_report, save_predictions,
    save_combined_predictions, ensure_directories, print_banner,
    assign_risk_levels, FEATURE_COLUMNS,
)
from src.visualizations import generate_all_visualizations

warnings.filterwarnings("ignore")

# ── Path Configuration ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "cleaned_wildfire_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
PREDS_DIR = os.path.join(BASE_DIR, "outputs", "predictions")

RANDOM_STATE = 42
TEST_SIZE = 0.2


def train_and_save_models(X_train, y_train):
    """
    Train SVM, Random Forest, and XGBoost if saved models don't exist.
    Saves trained models to models/ directory.
    Returns dict of trained model objects.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    models = {}

    # — SVM ——————————————————————————————————————————————
    svm_path = os.path.join(MODELS_DIR, "svm_model.pkl")
    if os.path.exists(svm_path):
        models["SVM"] = joblib.load(svm_path)
        print("[OK] SVM loaded from disk")
    else:
        print("[...] Training SVM...")
        svm = SVC(kernel="rbf", C=1.0, gamma="scale",
                  probability=True, random_state=RANDOM_STATE)
        svm.fit(X_train, y_train)
        joblib.dump(svm, svm_path)
        models["SVM"] = svm
        print("[OK] SVM trained and saved")

    # — Random Forest ———————————————————————————————————
    rf_path = os.path.join(MODELS_DIR, "rf_model.pkl")
    if os.path.exists(rf_path):
        models["Random Forest"] = joblib.load(rf_path)
        print("[OK] Random Forest loaded from disk")
    else:
        print("[...] Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        joblib.dump(rf, rf_path)
        models["Random Forest"] = rf
        print("[OK] Random Forest trained and saved")

    # — XGBoost ————————————————————————————————————————
    xgb_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
    if os.path.exists(xgb_path):
        models["XGBoost"] = joblib.load(xgb_path)
        print("[OK] XGBoost loaded from disk")
    else:
        print("[...] Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE)
        xgb.fit(X_train, y_train)
        joblib.dump(xgb, xgb_path)
        models["XGBoost"] = xgb
        print("[OK] XGBoost trained and saved")

    return models


def run_pipeline():
    """Execute the full prediction and visualization pipeline."""

    print_banner("WILDFIRE RISK CLASSIFICATION PIPELINE")

    # Step 1 — Setup directories
    ensure_directories(BASE_DIR)

    # Step 2 — Load dataset
    print("\n[1/6] Loading dataset...")
    df = load_dataset(DATA_PATH)
    X, y = get_features_and_target(df)

    # Step 3 — Split data
    print("\n[2/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # Step 4 — Load or train models
    print("\n[3/6] Loading / training models...")
    models = train_and_save_models(X_train, y_train)

    # Step 5 — Generate predictions & compute metrics
    print("\n[4/6] Running predictions...")
    metrics_dict = {}
    results = {}
    feature_importances = {}
    predictions_list = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)

        # Metrics
        m = compute_metrics(y_test, y_pred)
        metrics_dict[name] = m
        print(f"\n  --- {name} ---")
        print(f"  Accuracy : {m['accuracy']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall   : {m['recall']:.4f}")
        print(f"  F1-Score : {m['f1_score']:.4f}")
        print(get_classification_report(
            y_test, y_pred, target_names=["Low Risk", "High Risk"]))

        results[name] = {
            "y_true": np.array(y_test),
            "y_pred": y_pred,
            "probabilities": probs,
        }

        # Feature importances (tree-based models)
        if hasattr(model, "feature_importances_"):
            feature_importances[name] = model.feature_importances_

        # Collect for combined CSV
        predictions_list.append({
            "y_true": np.array(y_test),
            "y_pred": y_pred,
            "model_name": name,
        })

        # Save individual predictions
        save_predictions(y_test, y_pred, name, PREDS_DIR, probs)

    # Save combined predictions
    save_combined_predictions(predictions_list, PREDS_DIR)

    # Step 6 — Generate all visualizations
    print("\n[5/6] Generating visualizations...")
    generate_all_visualizations(
        df=df,
        metrics_dict=metrics_dict,
        results=results,
        feature_importances=feature_importances,
        feature_names=FEATURE_COLUMNS,
        output_dir=PLOTS_DIR,
    )

    # Summary
    print_banner("PIPELINE COMPLETE")
    print(f"  Plots saved to      : {PLOTS_DIR}")
    print(f"  Predictions saved to: {PREDS_DIR}")
    print(f"  Models saved to     : {MODELS_DIR}")

    # Print best model
    best = max(metrics_dict, key=lambda m: metrics_dict[m]["accuracy"])
    best_acc = metrics_dict[best]["accuracy"] * 100
    print(f"\n  Best Model: {best} ({best_acc:.2f}% accuracy)")
    print()


if __name__ == "__main__":
    run_pipeline()

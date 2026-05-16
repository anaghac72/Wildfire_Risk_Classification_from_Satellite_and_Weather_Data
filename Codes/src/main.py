import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import (
    load_dataset,
    get_features_and_target,
    ensure_directories,
    print_banner
)

from src.models import train_or_load_models, evaluate_models, save_predictions
from src.visualizations import generate_all_visualizations

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/cleaned_wildfire_dataset.csv")

MODEL_DIR = os.path.join(BASE_DIR, "../models")
OUTPUT_DIR = os.path.join(BASE_DIR, "../outputs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")


def run_pipeline():
    print_banner("🔥 WILDFIRE RISK CLASSIFICATION PIPELINE")

    # Create folders
    ensure_directories([MODEL_DIR, OUTPUT_DIR, PLOT_DIR, PRED_DIR])

    # Load dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = load_dataset(DATA_PATH)

    # Features & target
    X, y = get_features_and_target(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train or load models
    models = train_or_load_models(X_train, y_train, MODEL_DIR)

    # Evaluate models
    results, predictions = evaluate_models(models, X_test, y_test)

    # Save predictions
    save_predictions(predictions, y_test, PRED_DIR)

    # Visualizations
    generate_all_visualizations(results, y_test, predictions, PLOT_DIR)

    # Best model
    best_model = max(results, key=lambda x: results[x]["accuracy"])

    print("\n🏆 BEST MODEL:", best_model)
    print("Accuracy:", results[best_model]["accuracy"])


if __name__ == "__main__":
    run_pipeline()

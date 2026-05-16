import os
import joblib
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# =========================
# MODEL TRAINING
# =========================
def train_or_load_models(X_train, y_train, model_dir):

    os.makedirs(model_dir, exist_ok=True)

    models = {}

    model_configs = {
        "svm": SVC(kernel="rbf", probability=True),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42),
        "xgb": XGBClassifier(
            eval_metric="logloss",
            random_state=42
        )
    }

    for name, model in model_configs.items():
        path = os.path.join(model_dir, f"{name}.pkl")

        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"[LOADED] {name}")
        else:
            model.fit(X_train, y_train)
            joblib.dump(model, path)
            models[name] = model
            print(f"[TRAINED] {name}")

    return models


# =========================
# EVALUATION
# =========================
def evaluate_models(models, X_test, y_test):

    results = {}
    predictions = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "report": classification_report(y_test, y_pred)
        }

    return results, predictions


# =========================
# SAVE OUTPUTS
# =========================
def save_predictions(predictions, y_test, output_dir):

    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame({"Actual": y_test})

    for name, preds in predictions.items():
        df[name + "_pred"] = preds

    df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    print("[SAVED] predictions.csv")

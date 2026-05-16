import pandas as pd

# =========================
# CONFIG CONSTANTS
# =========================

FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "wind_speed",
    "rainfall",
    "vegetation_index"
]

TARGET_COLUMN = "risk_level"


MODEL_FILES = {
    "svm": "svm.pkl",
    "rf": "random_forest.pkl",
    "xgb": "xgboost.pkl"
}


# =========================
# DATA LOADING
# =========================
def load_dataset(path):
    df = pd.read_csv(path)
    df = df.dropna()
    return df


def get_features_and_target(df):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


# =========================
# METRICS (OPTIONAL)
# =========================
def compute_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }


def assign_risk_levels(pred):
    mapping = {
        0: "Low Risk",
        1: "Medium Risk",
        2: "High Risk"
    }
    return mapping.get(pred, "Unknown")

import pandas as pd
import os

FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "wind_speed",
    "rainfall",
    "vegetation_index"
]


def load_dataset(path):
    df = pd.read_csv(path)

    # Drop missing values safely
    df = df.dropna()

    return df


def get_features_and_target(df):
    X = df[FEATURE_COLUMNS]
    y = df["risk_level"]
    return X, y


def ensure_directories(paths):
    if isinstance(paths, str):
        paths = [paths]

    for path in paths:
        os.makedirs(path, exist_ok=True)


def print_banner(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60 + "\n")

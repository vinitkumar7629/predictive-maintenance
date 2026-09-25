"""
feature_engineering.py

Step 3: Turn raw per-cycle sensor readings into rolling-window features.

Why this matters: a single sensor reading is noisy -- engines have small
random fluctuations cycle to cycle even when healthy. What actually signals
"this engine is degrading" is the TREND over recent cycles, not any one
instantaneous value. So for each cycle, instead of just using the raw
sensor value, we compute:
  - a rolling mean (smooths out noise)
  - a rolling standard deviation (rising volatility can itself be a
    degradation symptom)
over the last WINDOW cycles for that engine.

This is the same idea used in real condition-monitoring systems: you don't
alarm on one noisy spike, you alarm on a sustained trend.
"""

import pandas as pd
from data_loader import load_train_data, load_test_data

WINDOW = 5

# The 12 sensors identified in eda.py as carrying real degradation signal.
USEFUL_SENSORS = [
    "sensor_11", "sensor_4", "sensor_15", "sensor_2", "sensor_17", "sensor_3",
    "sensor_8", "sensor_13", "sensor_20", "sensor_21", "sensor_7", "sensor_12",
]


def add_rolling_features(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Add rolling mean/std features per engine, per useful sensor."""
    df = df.sort_values(["engine_id", "cycle"]).copy()

    grouped = df.groupby("engine_id")[USEFUL_SENSORS]

    roll_mean = grouped.rolling(window=window, min_periods=1).mean()
    roll_std = grouped.rolling(window=window, min_periods=1).std().fillna(0)

    roll_mean.columns = [f"{c}_roll_mean" for c in roll_mean.columns]
    roll_std.columns = [f"{c}_roll_std" for c in roll_std.columns]

    roll_mean = roll_mean.reset_index(level=0, drop=True)
    roll_std = roll_std.reset_index(level=0, drop=True)

    df = pd.concat([df, roll_mean, roll_std], axis=1)
    return df


def build_feature_matrix(df: pd.DataFrame):
    """Return (X, y) -- rolling features and the RUL target."""
    df = add_rolling_features(df)
    feature_cols = [c for c in df.columns if c.endswith("_roll_mean") or c.endswith("_roll_std")]
    X = df[feature_cols]
    y = df["RUL"] if "RUL" in df.columns else None
    return X, y, feature_cols


if __name__ == "__main__":
    train_df = load_train_data("../data/train_FD001.txt")
    X_train, y_train, feature_cols = build_feature_matrix(train_df)

    print("Feature matrix shape:", X_train.shape)
    print("Number of features:", len(feature_cols))
    print("\nSample features (first engine, first 5 cycles):")
    print(X_train.head())
    print("\nAny missing values?", X_train.isna().sum().sum())
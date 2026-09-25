"""
train.py

Step 4: Train a model to predict Remaining Useful Life (RUL) from the
rolling-window sensor features, then derive a risk band (healthy / warning
/ critical) from the predicted RUL for the dashboard.

Two modeling decisions worth being able to explain in an interview:

1. RUL CLIPPING (piecewise RUL target):
   Early in an engine's life, RUL doesn't actually correlate with sensor
   readings -- a brand-new engine and a "500 cycles from failure" engine
   look sensor-wise almost identical, because degradation hasn't started
   yet. Trying to predict "480 vs 500" from flat sensor data just adds
   noise the model can't learn from. Standard practice on this dataset
   (established in PHM/CMAPSS literature) is to CAP the RUL target at a
   ceiling, e.g. 125 cycles: "if there's more than 125 cycles left, I
   don't need the exact number, I just need to know it's healthy."
   This turns a noisy unbounded regression into a much more learnable
   problem and is a standard, citable technique -- not a shortcut.

2. TRAIN/VALIDATION SPLIT BY ENGINE, NOT BY ROW:
   If we split randomly by row, cycles from the same engine end up in
   both train and validation, which leaks information (the model
   effectively memorizes that engine's trajectory). We split by engine_id
   instead so validation engines are ones the model has never seen at all
   -- this is the only way to honestly estimate real-world performance.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from data_loader import load_train_data
from feature_engineering import build_feature_matrix, USEFUL_SENSORS

RUL_CAP = 125

# Risk band thresholds (in predicted cycles remaining) for the dashboard.
RISK_THRESHOLDS = {"critical": 20, "warning": 50}  # RUL < 20 -> critical, < 50 -> warning, else healthy


def to_risk_band(rul_value: float) -> str:
    if rul_value < RISK_THRESHOLDS["critical"]:
        return "critical"
    elif rul_value < RISK_THRESHOLDS["warning"]:
        return "warning"
    return "healthy"


def main():
    df = load_train_data("../data/train_FD001.txt")
    df["RUL"] = df["RUL"].clip(upper=RUL_CAP)

    X, y, feature_cols = build_feature_matrix(df)

    # Split by engine, not by row, to avoid leakage.
    rng = np.random.default_rng(42)
    unique_engines = df["engine_id"].unique()
    rng.shuffle(unique_engines)
    n_val = int(len(unique_engines) * 0.2)
    val_engines = set(unique_engines[:n_val])

    val_mask = df["engine_id"].isin(val_engines).values
    X_train, X_val = X[~val_mask], X[val_mask]
    y_train, y_val = y[~val_mask], y[val_mask]

    print(f"Train rows: {len(X_train)} ({len(unique_engines) - n_val} engines)")
    print(f"Val rows:   {len(X_val)} ({n_val} engines)")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"\nValidation MAE:  {mae:.2f} cycles")
    print(f"Validation RMSE: {rmse:.2f} cycles")

    true_bands = y_val.apply(to_risk_band)
    pred_bands = pd.Series(preds, index=y_val.index).apply(to_risk_band)
    band_accuracy = (true_bands == pred_bands).mean()
    print(f"Risk-band classification accuracy: {band_accuracy:.1%}")

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 5 most important features:")
    print(importances.head())

    joblib.dump(model, "../models/rul_model.joblib")
    joblib.dump(feature_cols, "../models/feature_cols.joblib")
    print("\nModel saved to ../models/rul_model.joblib")


if __name__ == "__main__":
    main()
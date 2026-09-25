"""
eda.py

Step 2: Explore the CMAPSS sensor data to find out which sensors actually
carry a degradation signal, and which are flat/uninformative and should be
dropped before modeling.

Why this matters: the raw dataset has 21 sensors + 3 operating settings.
Feeding all of them into a model blindly wastes capacity on noise and makes
the model harder to explain in an interview. A domain-aware first pass --
"does this sensor's value change as the engine degrades?" -- is standard
practice in PdM/PHM literature on this exact dataset.
"""

import pandas as pd
from data_loader import load_train_data

pd.set_option("display.width", 120)


def main():
    df = load_train_data("../data/train_FD001.txt")

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    setting_cols = [c for c in df.columns if c.startswith("op_setting")]

    # 1. Operating settings: FD001 is a SINGLE operating condition subset,
    #    so these should be near-constant. Confirm that.
    print("=== Operating settings (should be ~constant for FD001) ===")
    print(df[setting_cols].describe().loc[["mean", "std"]])

    # 2. Sensor variance: a sensor that never changes carries zero
    #    information for predicting degradation. Flag near-zero-std sensors.
    print("\n=== Sensor standard deviation (near-zero = flat/uninformative) ===")
    sensor_std = df[sensor_cols].std().sort_values()
    print(sensor_std)

    flat_sensors = sensor_std[sensor_std < 1e-3].index.tolist()
    print(f"\nFlat sensors to drop: {flat_sensors}")

    # 3. Correlation with RUL: which sensors actually trend as the engine
    #    approaches failure? Strong (positive or negative) correlation =
    #    a sensor worth keeping.
    print("\n=== Sensor correlation with RUL (degradation signal strength) ===")
    corr_with_rul = df[sensor_cols].corrwith(df["RUL"]).sort_values()
    print(corr_with_rul)

    useful_sensors = corr_with_rul[corr_with_rul.abs() > 0.5].index.tolist()
    print(f"\nSensors with strong degradation signal (|corr| > 0.5): {useful_sensors}")


if __name__ == "__main__":
    main()
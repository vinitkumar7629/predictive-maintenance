"""
data_loader.py

Loads the NASA CMAPSS turbofan engine sensor dataset and computes
Remaining Useful Life (RUL) labels for training.

Why RUL matters for predictive maintenance:
Each engine in the training set runs from a healthy state until failure.
At every cycle, we know exactly how many cycles remain before that engine
fails (because we can see the whole trajectory). That "cycles remaining"
number is the RUL label our model learns to predict from sensor readings
alone -- which is exactly what a real factory system would need to do,
since in production you never get to see the future.
"""

import pandas as pd

# Column names per the CMAPSS dataset documentation.
# 3 operational settings + 21 sensor readings, per engine per cycle.
COLUMN_NAMES = (
    ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_train_data(path: str) -> pd.DataFrame:
    """Load a CMAPSS train_FD00x.txt file and attach RUL labels.

    RUL for a row = (that engine's max cycle) - (current cycle).
    At the last row before failure, RUL = 0.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)

    # Max observed cycle per engine = the cycle at which it failed.
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]

    return df


def load_test_data(path: str, rul_path: str) -> pd.DataFrame:
    """Load a CMAPSS test_FD00x.txt file.

    Unlike training data, test trajectories are cut off BEFORE failure --
    that's what makes it a fair test of the model. The true remaining life
    at the point of cutoff is given separately in RUL_FD00x.txt.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    true_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])
    true_rul["engine_id"] = true_rul.index + 1  # 1-indexed engine IDs

    return df, true_rul


if __name__ == "__main__":
    train_df = load_train_data("../data/train_FD001.txt")
    test_df, test_rul = load_test_data("../data/test_FD001.txt", "../data/RUL_FD001.txt")

    print("Train shape:", train_df.shape)
    print("Number of engines (train):", train_df["engine_id"].nunique())
    print("\nSample rows:")
    print(train_df[["engine_id", "cycle", "sensor_2", "sensor_3", "sensor_4", "RUL"]].head())
    print("\nRUL distribution (train):")
    print(train_df["RUL"].describe())
"""
main.py

FastAPI service that wraps the trained RUL model.

Design choice worth explaining in an interview: the API takes the LAST FEW
CYCLES of sensor readings for one engine (not just a single reading),
because the model was trained on rolling-window features, not raw
instantaneous values. This mirrors how a real deployment would work: a
factory edge system streams recent sensor history to the service, not a
single snapshot, because a snapshot alone can't show a trend.

Endpoints:
  GET  /health            - liveness check
  GET  /engines            - demo engines available for the dashboard
  GET  /engines/{id}/history - full sensor history for one demo engine
  POST /predict/rul        - predict RUL + risk band from recent sensor readings
"""

import sys
import os
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_engineering import USEFUL_SENSORS  # noqa: E402
from train import to_risk_band, RISK_THRESHOLDS  # noqa: E402
from data_loader import load_train_data  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "rul_model.joblib")
FEATURE_COLS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "feature_cols.joblib")

app = FastAPI(
    title="Smart Factory Predictive Maintenance API",
    description="Predicts Remaining Useful Life (RUL) and risk band from recent equipment sensor readings.",
    version="1.0.0",
)

model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEATURE_COLS_PATH)

# Loaded once at startup so the dashboard has real engines to simulate
# against, without needing its own copy of the raw CMAPSS files.
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train_FD001.txt")
_demo_df = load_train_data(DATA_PATH)
DEMO_ENGINE_IDS = [1, 24, 47, 66, 88]  # a mix of short/long-lived engines for the demo dropdown


class SensorReading(BaseModel):
    """One cycle's worth of readings for the sensors the model actually uses."""
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_7: float
    sensor_8: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_15: float
    sensor_17: float
    sensor_20: float
    sensor_21: float


class PredictionRequest(BaseModel):
    engine_id: str = Field(..., description="Identifier for the equipment/engine being monitored")
    readings: List[SensorReading] = Field(
        ..., min_length=1, max_length=5,
        description="Most recent cycles of sensor readings, oldest first (up to 5, matching the model's training window)",
    )


class PredictionResponse(BaseModel):
    engine_id: str
    predicted_rul: float
    risk_band: str
    risk_thresholds: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/engines")
def list_engines():
    """Demo engines available for the dashboard to simulate against."""
    engines = []
    for eid in DEMO_ENGINE_IDS:
        engine_df = _demo_df[_demo_df.engine_id == eid]
        engines.append({"engine_id": str(eid), "total_cycles": int(engine_df.cycle.max())})
    return {"engines": engines}


@app.get("/engines/{engine_id}/history")
def engine_history(engine_id: int):
    """Full run-to-failure sensor history for one demo engine, oldest first."""
    engine_df = _demo_df[_demo_df.engine_id == engine_id].sort_values("cycle")
    if engine_df.empty:
        raise HTTPException(status_code=404, detail=f"No demo data for engine_id={engine_id}")

    cycles = engine_df[["cycle"] + USEFUL_SENSORS].to_dict(orient="records")
    return {"engine_id": str(engine_id), "cycles": cycles}


@app.post("/predict/rul", response_model=PredictionResponse)
def predict_rul(request: PredictionRequest):
    try:
        readings_df = pd.DataFrame([r.model_dump() for r in request.readings])

        roll_mean = readings_df[USEFUL_SENSORS].mean()
        roll_std = readings_df[USEFUL_SENSORS].std(ddof=0).fillna(0)

        feature_row = {}
        for sensor in USEFUL_SENSORS:
            feature_row[f"{sensor}_roll_mean"] = roll_mean[sensor]
            feature_row[f"{sensor}_roll_std"] = roll_std[sensor]

        X = pd.DataFrame([feature_row])[feature_cols]  # enforce training column order
        predicted_rul = float(model.predict(X)[0])
        risk_band = to_risk_band(predicted_rul)

        return PredictionResponse(
            engine_id=request.engine_id,
            predicted_rul=round(predicted_rul, 1),
            risk_band=risk_band,
            risk_thresholds=RISK_THRESHOLDS,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/")
def dashboard():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
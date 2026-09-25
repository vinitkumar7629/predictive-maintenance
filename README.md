# Smart Factory Predictive Maintenance

Predicts a machine's **Remaining Useful Life (RUL)** from recent sensor
readings and classifies it into a **risk band (healthy / warning /
critical)**, served through a FastAPI backend with a live simulation
dashboard.

**Live Demo:** [https://predictive-maintenance-zshu.onrender.com](https://predictive-maintenance-zshu.onrender.com)

## Problem

Unplanned equipment failure is expensive on a factory floor — it halts
production and forces reactive, emergency maintenance. Predictive
maintenance flips this: by watching how sensor readings trend over time, a
model can flag a machine as "at risk" cycles or days before it actually
fails, so maintenance can be scheduled proactively instead of reactively.

## Dataset

NASA C-MAPSS Turbofan Engine Degradation Simulation (FD001 subset) — the
standard public benchmark for run-to-failure prognostics research. 100
simulated turbofan engines, each run from a healthy state to failure, with
21 sensor channels + 3 operating settings recorded per operational cycle.
Sourced via a GitHub mirror of NASA's Prognostics Center of Excellence
data repository.

**Why an aircraft-engine dataset for a "smart factory" project?** There is
no equivalently large, clean, public dataset of real factory-floor sensor
failures. CMAPSS is the closest public analog: multivariate sensor
time-series degrading toward a known failure point. The modeling approach
here (rolling-window feature extraction → RUL regression → risk
classification) transfers directly to real industrial sensor streams.

## Architecture

Raw sensor logs (CMAPSS)
|
data_loader.py -> loads readings, computes RUL labels
|
eda.py -> drops flat sensors, keeps sensors
| correlated with degradation
feature_engineering.py -> rolling mean/std over a 5-cycle window
|
train.py -> Random Forest regressor, RUL capped at 125,
| engine-level train/val split (no leakage)
|
api/main.py -> FastAPI service: /predict/rul, /engines,
| /engines/{id}/history
|
api/static/ -> dashboard: replays real engine sensor
history against the live API


## Key design decisions

- **Signal-quality pass before modeling** — dropped 6 sensors with zero
  variance, kept 12 sensors with meaningful correlation to RUL.
- **Rolling-window features, not raw readings** — a 5-cycle rolling
  mean/std captures trend and volatility, which is what signals
  degradation, not a single noisy reading.
- **RUL capping at 125 cycles** — early-life sensor data doesn't
  meaningfully predict exact remaining life; capping the target turns an
  unlearnable problem into a learnable one (standard CMAPSS/PHM practice).
- **Engine-level train/validation split** — split by `engine_id`, not by
  row, so validation engines are entirely unseen and results aren't
  inflated by leakage across cycles of the same engine.

## Results

| Metric | Value |
|---|---|
| Validation MAE | ~14.4 cycles |
| Validation RMSE | ~19.8 cycles |
| Risk-band accuracy | ~89% |
| Most important feature | `sensor_4` rolling mean |

## Running locally

```bash
pip install -r requirements.txt
cd src && python3 train.py
cd ../api && uvicorn main:app --reload --port 8000
```
Then open `http://localhost:8000` for the dashboard, or `/docs` for API docs.

## Deploying (Render)

1. Push this repo to GitHub.
2. On Render: new **Web Service** from the repo.
3. Build command: `pip install -r requirements.txt && cd src && python3 train.py`
4. Start command: `cd api && uvicorn main:app --host 0.0.0.0 --port $PORT`

## Production gap: from this demo to a real factory deployment

This project demonstrates the ML methodology using public simulation data.
A real factory deployment would differ mainly in the **data ingestion**
layer: instead of replaying a static file, sensor readings would stream
from PLCs/SCADA systems in real time over an industrial protocol like
**OPC-UA** or **Modbus**, into the same feature engineering and prediction
pipeline built here. The model, API, and dashboard logic wouldn't need to
change — only the source feeding `/predict/rul` would move from a
replayed history file to a live OPC-UA/Modbus client.

## Possible extensions

- Swap Random Forest for an LSTM/1D-CNN using full sensor trajectories.
- Add alerting when an engine crosses into `critical`.
- Extend to FD002/FD004 subsets (multiple operating conditions).
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import json
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
LOG_DIR = BASE / "logs"
LOG_DIR.mkdir(exist_ok=True)

manifest = json.loads((MODEL_DIR / "model_manifest.json").read_text(encoding="utf-8"))
models = {
    name: joblib.load(MODEL_DIR / info["file"])
    for name, info in manifest["models"].items()
}

app = FastAPI(
    title="BOREMAN Water Quality Forecast API",
    version="1.0.0",
    description="Controlled demonstration API. Forecasts are not water-safety classifications."
)

class SensorReading(BaseModel):
    tds: Optional[float] = Field(default=None, ge=0)
    humidity: Optional[float] = None
    temperature: Optional[float] = None
    timestamp: Optional[datetime] = None

@app.get("/")
def root():
    return {
        "application": "BOREMAN AI API",
        "status": "running",
        "model_version": manifest["model_version"],
        "warning": "Forecasts are not water-safety classifications."
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "model_version": manifest["model_version"]
    }

@app.get("/models")
def model_list():
    return manifest

@app.post("/forecast")
def forecast(reading: SensorReading):
    timestamp = reading.timestamp or datetime.now(timezone.utc)
    now = datetime.now(timezone.utc)

    if timestamp.tzinfo is None:
        raise HTTPException(status_code=400, detail="Timestamp must include timezone.")
    age_seconds = (now - timestamp.astimezone(timezone.utc)).total_seconds()

    if age_seconds < -60:
        raise HTTPException(status_code=400, detail="Reading timestamp is in the future.")
    if age_seconds > 300:
        raise HTTPException(status_code=400, detail="Reading is stale; fallback required.")

    values = {
        "tds": reading.tds,
        "humidity": reading.humidity,
        "Temperature": reading.temperature
    }

    if all(v is None for v in values.values()):
        raise HTTPException(status_code=400, detail="At least one sensor reading is required.")

    result = {
        "timestamp": timestamp.isoformat(),
        "model_version": manifest["model_version"],
        "forecasts": {},
        "data_quality": {
            "input_age_seconds": round(age_seconds, 1),
            "stale": False,
            "missing_sensors": [k for k, v in values.items() if v is None],
            "status": "PARTIAL_INPUT" if any(v is None for v in values.values()) else "INPUT_RECEIVED"
        },
        "notice": "Experimental sensor forecasts only. Not a drinking-water safety assessment."
    }

    for target, current in values.items():
        result["forecasts"][target] = {}
        for horizon in (5, 15, 30, 60):
            key = f"{target}_{horizon}m"
            if target == "tds" or key not in models:
                prediction = current
                method = "PERSISTENCE"
            else:
                prediction = None
                method = "RIDGE_SHADOW_NOT_YET_INFERRED"
            result["forecasts"][target][f"{horizon}m"] = {
                "value": prediction,
                "method": method,
                "unit": "mg/L" if target == "tds" else ("%" if target == "humidity" else "°C")
            }

    log_file = LOG_DIR / "forecast_requests.jsonl"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": timestamp.isoformat(),
            "received_at": now.isoformat(),
            "inputs": values,
            "model_version": manifest["model_version"],
            "forecast_policy": result["forecasts"]
        }) + "\n")

    return result

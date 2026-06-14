# app/main.py
import sys, json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from src.pipeline import engineer

# ── Logging ──────────────────────────────────────────────────
logger.remove()
Path("logs").mkdir(exist_ok=True)
logger.add("logs/predictions.jsonl", format="{message}", level="INFO", rotation="10 MB")

# ── Load model ───────────────────────────────────────────────
import onnxruntime as rt
artifact  = joblib.load("models/flood_model.pkl")
encoder   = artifact["encoder"]
FEATURES  = artifact["features"]
CAT_COLS  = artifact["cat_cols"]

sess = rt.InferenceSession("models/flood_model.onnx")
input_name = sess.get_inputs()[0].name

# ── App ──────────────────────────────────────────────────────
app = FastAPI(
    title="False Positive's FloodRisk API",
    description="Flood risk score prediction by team False Positives",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"]
)

# ── Schema ───────────────────────────────────────────────────
class LocationInput(BaseModel):
    rainfall_7d_mm:                 float = Field(..., ge=0,   example=85.0)
    elevation_m:                    float = Field(..., ge=0,   example=12.0)
    distance_to_river_m:            float = Field(..., ge=0,   example=350.0)
    drainage_index:                 float = Field(..., ge=0, le=1, example=0.4)
    ndwi:                           float = Field(..., ge=-1, le=1, example=0.3)
    ndvi:                           float = Field(..., ge=-1, le=1, example=0.2)
    historical_flood_count:         int   = Field(..., ge=0,   example=3)
    inundation_area_sqm:            float = Field(..., ge=0,   example=4500.0)
    district:                       str   = Field(...,         example="Colombo")
    landcover:                      str   = Field(...,         example="Urban")
    soil_type:                      str   = Field(...,         example="Clay")
    water_supply:                   str   = Field(...,         example="Piped")
    electricity:                    str   = Field(...,         example="Yes")
    road_quality:                   str   = Field(...,         example="Fair")
    urban_rural:                    str   = Field(...,         example="Urban")
    water_presence_flag:            str   = Field(...,         example="Likely")
    flood_occurrence_current_event: str   = Field(...,         example="Yes")
    is_good_to_live:                str   = Field(...,         example="No")
    generation_date:                str   = Field("2024-06-01", example="2024-06-01")

class PredictionOut(BaseModel):
    flood_risk_score: float
    risk_level:       str
    confidence:       str
    model_version:    str
    timestamp:        str

def risk_label(s: float) -> str:
    if s < 0.25: return "Low"
    if s < 0.50: return "Moderate"
    if s < 0.75: return "High"
    return "Critical"

def confidence(s: float) -> str:
    d = abs(s - 0.5)
    if d > 0.35: return "High"
    if d > 0.15: return "Moderate"
    return "Low"

# ── Routes ───────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_version": "1.0.0"}

@app.post("/predict", response_model=PredictionOut)
def predict(req: LocationInput):
    try:
        df = pd.DataFrame([req.dict()])
        df = engineer(df)

        for f in FEATURES:
            if f not in df.columns:
                df[f] = np.nan
        X = df[FEATURES].copy()
        X[CAT_COLS] = encoder.transform(X[CAT_COLS].astype(str))
        
        X_float32 = X.astype(np.float32).values
        pred_onnx = sess.run(None, {input_name: X_float32})[0]
        score = float(np.clip(pred_onnx.flatten()[0], 0, 1))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    entry = {
        "timestamp":  datetime.utcnow().isoformat(),
        "input":      req.dict(),
        "score":      round(score, 4),
        "risk_level": risk_label(score),
    }
    logger.info(json.dumps(entry))

    return PredictionOut(
        flood_risk_score=round(score, 4),
        risk_level=risk_label(score),
        confidence=confidence(score),
        model_version="1.0.0",
        timestamp=entry["timestamp"],
    )

@app.get("/metrics")
def metrics():
    log_file = Path("logs/predictions.jsonl")
    if not log_file.exists() or log_file.stat().st_size == 0:
        return {"total_predictions": 0, "avg_risk_score": 0,
                "risk_distribution": {}, "recent": []}
    lines = [json.loads(l) for l in log_file.read_text().strip().splitlines() if l]
    scores = [l["score"] for l in lines]
    dist   = {}
    for l in lines:
        dist[l["risk_level"]] = dist.get(l["risk_level"], 0) + 1
    return {
        "total_predictions": len(lines),
        "avg_risk_score":    round(sum(scores) / len(scores), 4),
        "risk_distribution": dist,
        "recent":            lines[-10:],
    }
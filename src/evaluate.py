import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    """Return RMSE/MAE/R2 for regression predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def log_experiment(model_version: str, metrics: dict, params: dict, log_dir: str = "logs") -> Path:
    """Append a structured training log entry to logs/training_runs.jsonl."""
    out_dir = Path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "training_runs.jsonl"

    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": model_version,
        "metrics": metrics,
        "params": params,
    }
    with out_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    return out_file
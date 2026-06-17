"""Serving-consistent retrain for FloodGuard.

The original `train.py` trained on ~45 columns, many of which (the precomputed
``*_log1p`` / ``*_yeojohnson`` / ``*_qmap`` transforms and the district target
encoding) are NOT reproducible at serving time. The API filled them with NaN,
collapsing live predictions into a narrow band (train/serve skew).

This script retrains on ONLY serving-reproducible features, persists the
district target-encoding map so the API can reconstruct it, and relaxes the
over-regularization so the model fits — and spreads — properly.

Run:  python -m src.train_serving
"""
from pathlib import Path
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
import onnxmltools
from onnxmltools.convert import convert_lightgbm
from skl2onnx.common.data_types import FloatTensorType

from src.evaluate import log_experiment, regression_metrics
from src.pipeline import engineer, add_district_te

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('models')
OUTPUT_DIR.mkdir(exist_ok=True)

T = 'flood_risk_score'
N = 5
SMOOTH = 50

# Serving-reproducible features only (everything engineer() + add_district_te()
# can build from a LocationInput / mock_features dict).
RAW_NUM = [
    'rainfall_7d_mm', 'monthly_rainfall_mm', 'elevation_m', 'distance_to_river_m',
    'drainage_index', 'ndvi', 'ndwi', 'historical_flood_count', 'inundation_area_sqm',
]
ENGINEERED = [
    'gen_month_sin', 'gen_month_cos', 'rain_drainage_ratio', 'rain_elev_ratio',
    'river_elev_ratio', 'ndwi_ndvi_diff', 'flood_x_rain', 'inund_log1p',
    'water_likely', 'flood_occurred', 'is_urban', 'road_quality_ord',
    'district_te_mean', 'district_te_std',
]
CAT_COLS = [
    'district', 'landcover', 'soil_type', 'water_supply', 'electricity',
    'road_quality', 'urban_rural', 'water_presence_flag',
    'flood_occurrence_current_event', 'is_good_to_live',
]
FEATURES = RAW_NUM + ENGINEERED + CAT_COLS


def _load_csv(candidates):
    for p in candidates:
        if Path(p).exists():
            return pd.read_csv(p)
    raise FileNotFoundError(f"None of these files exist: {candidates}")


def build_te_map(train_df, smooth, target_col):
    """Smoothed per-district mean/std target-encoding map for serving."""
    gmean = train_df[target_col].mean()
    gstd = train_df[target_col].std()
    agg = train_df.groupby('district')[target_col].agg(['mean', 'std', 'count'])
    mean_sm = (agg['mean'] * agg['count'] + gmean * smooth) / (agg['count'] + smooth)
    std_sm = agg['std'].fillna(gstd)
    return {
        'mean': mean_sm.to_dict(),
        'std': std_sm.to_dict(),
        'global_mean': float(gmean),
        'global_std': float(gstd),
    }


def oof_te(train_df, kf, smooth, target_col):
    """Out-of-fold district TE for training rows (avoids leakage)."""
    om = np.zeros(len(train_df))
    os_ = np.zeros(len(train_df))
    gmean = train_df[target_col].mean()
    gstd = train_df[target_col].std()
    for tr, va in kf.split(train_df):
        agg = train_df.iloc[tr].groupby('district')[target_col].agg(['mean', 'std', 'count'])
        fold_gm = train_df.iloc[tr][target_col].mean()
        msm = (agg['mean'] * agg['count'] + fold_gm * smooth) / (agg['count'] + smooth)
        ssm = agg['std'].fillna(gstd)
        om[va] = train_df.iloc[va]['district'].map(msm).fillna(gmean).values
        os_[va] = train_df.iloc[va]['district'].map(ssm).fillna(gstd).values
    return om, os_


def main():
    train_df = engineer(_load_csv(['./train.csv', './data/train.csv']))

    te_map = build_te_map(train_df, SMOOTH, T)
    kf = KFold(N, shuffle=True, random_state=42)
    om, os_ = oof_te(train_df, kf, SMOOTH, T)
    train_df['district_te_mean'] = om
    train_df['district_te_std'] = os_

    X = train_df[FEATURES].copy()
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[CAT_COLS] = enc.fit_transform(X[CAT_COLS].astype(str))

    model = lgb.LGBMRegressor(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, train_df[T])

    pred = np.clip(model.predict(X), 0, 1)
    metrics = regression_metrics(train_df[T].values, pred)
    print(f"Train metrics: {metrics}")
    print(f"Pred spread: min={pred.min():.3f} max={pred.max():.3f} std={pred.std():.3f}")

    artifact_path = OUTPUT_DIR / 'flood_model.pkl'
    joblib.dump(
        {
            'encoder': enc,
            'features': FEATURES,
            'cat_cols': CAT_COLS,
            'te_map': te_map,
            'global_mean': te_map['global_mean'],
            'model': model,
        },
        artifact_path,
    )

    onnx_path = OUTPUT_DIR / 'flood_model.onnx'
    initial_types = [('float_input', FloatTensorType([None, len(FEATURES)]))]
    onnx_model = convert_lightgbm(model, initial_types=initial_types)
    onnxmltools.utils.save_model(onnx_model, onnx_path)

    log_experiment(
        model_version='v3-serving-consistent',
        metrics=metrics,
        params={
            'target': T, 'n_folds': N, 'smooth': SMOOTH,
            'n_features': len(FEATURES), 'saved_model': 'lightgbm',
            'note': 'serving-reproducible features + persisted district TE',
        },
    )
    print(f"Saved {artifact_path} and {onnx_path}")


if __name__ == '__main__':
    main()

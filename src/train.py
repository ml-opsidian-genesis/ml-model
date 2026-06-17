from pathlib import Path
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer, QuantileTransformer
import onnxmltools
from onnxmltools.convert import convert_lightgbm
from skl2onnx.common.data_types import FloatTensorType

from src.evaluate import log_experiment, regression_metrics
from src.pipeline import engineer

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('models')
OUTPUT_DIR.mkdir(exist_ok=True)

T = 'flood_risk_score'
N = 5
SMOOTH = 50
SEEDS = [42, 123, 789, 2024, 555]

DROP = [T, 'record_id', 'place_name', 'generation_date', 'reason_not_good_to_live', 'is_synthetic']
CAT_COLS = [
    'district', 'landcover', 'soil_type', 'water_supply', 'electricity',
    'road_quality', 'urban_rural', 'water_presence_flag',
    'flood_occurrence_current_event', 'is_good_to_live'
]


def _load_csv(candidates):
    for p in candidates:
        path = Path(p)
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"None of these files exist: {candidates}")


def te_oof_mean(tr_df, te_df, col, kf, smooth, target_col, global_mean):
    om = np.zeros(len(tr_df))
    for tr, va in kf.split(tr_df):
        agg = tr_df.iloc[tr].groupby(col)[target_col].agg(['mean', 'count'])
        gm = tr_df.iloc[tr][target_col].mean()
        sm = (agg['mean'] * agg['count'] + gm * smooth) / (agg['count'] + smooth)
        om[va] = tr_df.iloc[va][col].map(sm).fillna(gm).values
    agg = tr_df.groupby(col)[target_col].agg(['mean', 'count'])
    sm = (agg['mean'] * agg['count'] + global_mean * smooth) / (agg['count'] + smooth)
    return om, te_df[col].map(sm).fillna(global_mean).values


def te_oof_std(tr_df, te_df, col, kf, target_col):
    os = np.zeros(len(tr_df))
    for tr, va in kf.split(tr_df):
        agg = tr_df.iloc[tr].groupby(col)[target_col].std()
        os[va] = tr_df.iloc[va][col].map(agg).fillna(tr_df.iloc[tr][target_col].std()).values
    agg = tr_df.groupby(col)[target_col].std()
    return os, te_df[col].map(agg).fillna(tr_df[target_col].std()).values


def train_and_save_model():
    train_df = _load_csv(['./train.csv', './data/train.csv'])
    test_df = _load_csv(['./test.csv', './data/test.csv'])

    train_df = engineer(train_df)
    test_df = engineer(test_df)
    global_mean = train_df[T].mean()

    # District target-encoding: a simple, static per-district mean/std so the
    # exact same lookup can be replayed at serving time (OOF folds can't be
    # reproduced outside of training, so we don't use them for the saved
    # artifact -- the model is fit on the full training set anyway).
    global_std = train_df[T].std()
    district_te_mean_map = train_df.groupby('district')[T].mean().to_dict()
    district_te_std_map = train_df.groupby('district')[T].std().fillna(global_std).to_dict()

    tr2_save, te2_save = train_df.copy(), test_df.copy()
    tr2_save['district_te_mean'] = tr2_save['district'].map(district_te_mean_map).fillna(global_mean)
    tr2_save['district_te_std'] = tr2_save['district'].map(district_te_std_map).fillna(global_std)
    te2_save['district_te_mean'] = te2_save['district'].map(district_te_mean_map).fillna(global_mean)
    te2_save['district_te_std'] = te2_save['district'].map(district_te_std_map).fillna(global_std)

    # Yeo-Johnson / quantile transforms: fit on train, persist the fitted
    # transformer so serving can replay the exact same mapping on new rows.
    fitted_transforms = {}

    def _fit_transform(df_tr, df_te, src_col, out_col, transformer_cls, **kwargs):
        if src_col not in df_tr.columns:
            fitted_transforms[out_col] = None
            df_tr[out_col] = np.nan
            df_te[out_col] = np.nan
            return
        tf = transformer_cls(**kwargs)
        df_tr[out_col] = tf.fit_transform(df_tr[[src_col]].astype(float).values).flatten()
        df_te[out_col] = tf.transform(df_te[[src_col]].astype(float).values).flatten()
        fitted_transforms[out_col] = tf

    _fit_transform(tr2_save, te2_save, 'elevation_m', 'elevation_m_yeojohnson', PowerTransformer, method='yeo-johnson')
    _fit_transform(tr2_save, te2_save, 'drainage_index', 'drainage_index_yeojohnson', PowerTransformer, method='yeo-johnson')
    _fit_transform(tr2_save, te2_save, 'ndvi', 'ndvi_qmap', QuantileTransformer, n_quantiles=100, output_distribution='uniform')
    _fit_transform(tr2_save, te2_save, 'ndwi', 'ndwi_qmap', QuantileTransformer, n_quantiles=100, output_distribution='uniform')
    _fit_transform(tr2_save, te2_save, 'built_up_percent', 'built_up_percent_qmap', QuantileTransformer, n_quantiles=100, output_distribution='uniform')

    feat_save = [c for c in tr2_save.columns if c not in DROP]
    cat_in_save = [c for c in CAT_COLS if c in feat_save]
    X_save = tr2_save[feat_save]

    enc_save = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    Xe_save = X_save.copy()
    Xe_save[cat_in_save] = enc_save.fit_transform(X_save[cat_in_save].astype(str))

    lgb_save = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.02,
        num_leaves=15,
        max_depth=4,
        min_child_samples=100,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_save.fit(Xe_save, train_df[T])

    # Train-set metrics for tracking only (does not affect model artifact).
    train_pred = np.clip(lgb_save.predict(Xe_save), 0, 1)
    metrics = regression_metrics(train_df[T].values, train_pred)

    artifact_path = OUTPUT_DIR / 'flood_model.pkl'
    joblib.dump(
        {
            'encoder': enc_save,
            'features': feat_save,
            'cat_cols': cat_in_save,
            'district_te_mean_map': district_te_mean_map,
            'district_te_std_map': district_te_std_map,
            'te_global_mean': float(global_mean),
            'te_global_std': float(global_std),
            'fitted_transforms': fitted_transforms,
        },
        artifact_path,
    )

    onnx_path = OUTPUT_DIR / 'flood_model.onnx'
    initial_types = [('float_input', FloatTensorType([None, len(feat_save)]))]
    onnx_model = convert_lightgbm(lgb_save, initial_types=initial_types)
    onnxmltools.utils.save_model(onnx_model, onnx_path)

    log_file = log_experiment(
        model_version='v2-lgb-artifact',
        metrics=metrics,
        params={
            'target': T,
            'n_folds': N,
            'smooth': SMOOTH,
            'seed_list': SEEDS,
            'saved_model_seed': 42,
            'saved_model': 'lightgbm',
        },
    )

    print(f"Training complete. Model saved to {artifact_path}")
    print(f"Run metrics logged to {log_file}")


if __name__ == '__main__':
    train_and_save_model()
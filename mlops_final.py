import pandas as pd, numpy as np, warnings
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import Ridge
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('outputs'); OUTPUT_DIR.mkdir(exist_ok=True)
train = pd.read_csv('./train.csv'); test = pd.read_csv('./test.csv')
sub   = pd.read_csv('./sample_submission.csv'); T = 'flood_risk_score'

# ── Feature engineering ────────────────────────────────────
def engineer(df):
    df = df.copy()
    df['gen_month']      = pd.to_datetime(df['generation_date'], errors='coerce').dt.month
    df['gen_month_sin']  = np.sin(2*np.pi*df['gen_month']/12)
    df['gen_month_cos']  = np.cos(2*np.pi*df['gen_month']/12)
    df['rain_drainage_ratio'] = df['rainfall_7d_mm']      / (df['drainage_index'] + 1e-3)
    df['rain_elev_ratio']     = df['rainfall_7d_mm']      / (df['elevation_m'] + 1)
    df['river_elev_ratio']    = df['distance_to_river_m'] / (df['elevation_m'] + 1)
    df['ndwi_ndvi_diff']      = df['ndwi'] - df['ndvi']
    df['flood_x_rain']        = df['historical_flood_count'] * df['rainfall_7d_mm']
    df['inund_log1p']         = np.log1p(df['inundation_area_sqm'])
    df['water_likely']        = (df['water_presence_flag'] == 'Likely').astype(int)
    df['flood_occurred']      = (df['flood_occurrence_current_event'] == 'Yes').astype(int)
    df['is_urban']            = (df['urban_rural'] == 'Urban').astype(int)
    df['road_quality_ord']    = df['road_quality'].map(
        {'No road access': 0, 'Poor (unpaved)': 1, 'Fair': 2, 'Good (paved)': 3})
    return df
train = engineer(train); test = engineer(test)

DROP = [T, 'record_id', 'place_name', 'generation_date', 'reason_not_good_to_live', 'is_synthetic']
cat_cols = ['district', 'landcover', 'soil_type', 'water_supply', 'electricity',
            'road_quality', 'urban_rural', 'water_presence_flag',
            'flood_occurrence_current_event', 'is_good_to_live']
GLOBAL = train[T].mean()
N = 5                       # folds
SMOOTH = 50                 # target-encoding smoothing
SEEDS = [42, 123, 789, 2024, 555]

# ── Leak-free out-of-fold target encoding for `district` (mean + std) ─────────
def te_oof_mean(tr_df, te_df, col, kf, smooth):
    om = np.zeros(len(tr_df))
    for tr, va in kf.split(tr_df):
        agg = tr_df.iloc[tr].groupby(col)[T].agg(['mean', 'count']); gm = tr_df.iloc[tr][T].mean()
        sm  = (agg['mean']*agg['count'] + gm*smooth) / (agg['count'] + smooth)
        om[va] = tr_df.iloc[va][col].map(sm).fillna(gm).values
    agg = tr_df.groupby(col)[T].agg(['mean', 'count'])
    sm  = (agg['mean']*agg['count'] + GLOBAL*smooth) / (agg['count'] + smooth)
    return om, te_df[col].map(sm).fillna(GLOBAL).values

def te_oof_std(tr_df, te_df, col, kf):
    os = np.zeros(len(tr_df))
    for tr, va in kf.split(tr_df):
        agg = tr_df.iloc[tr].groupby(col)[T].std()
        os[va] = tr_df.iloc[va][col].map(agg).fillna(tr_df.iloc[tr][T].std()).values
    agg = tr_df.groupby(col)[T].std()
    return os, te_df[col].map(agg).fillna(tr_df[T].std()).values

# ── Build one Ridge-stacked ensemble for a given seed + objective ─────────────
def stack_for_seed(SEED, objective):
    """objective: 'L2' -> {lgb,xgb,cat} squared error ; 'L1' -> {lgb,cat} absolute error.
       Returns the averaged-over-folds Ridge-stacked TEST prediction vector."""
    kf = KFold(N, shuffle=True, random_state=SEED)

    tr2, te2 = train.copy(), test.copy()
    dm, dmt   = te_oof_mean(tr2, te2, 'district', kf, SMOOTH)
    dstd, dst = te_oof_std (tr2, te2, 'district', kf)
    tr2['district_te_mean'] = dm;  tr2['district_te_std'] = dstd
    te2['district_te_mean'] = dmt; te2['district_te_std'] = dst

    feat   = [c for c in tr2.columns if c not in DROP]
    cat_in = [c for c in cat_cols if c in feat]
    X, y   = tr2[feat], tr2[T]; X_test = te2[feat].copy()

    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    Xe  = X.copy();      Xe[cat_in]  = enc.fit_transform(X[cat_in].astype(str))
    Xte = X_test.copy(); Xte[cat_in] = enc.transform(X_test[cat_in].astype(str))

    if objective == 'L2':
        keys = ['lgb', 'xgb', 'cat']
        lgb_p = dict(n_estimators=800, learning_rate=0.02, num_leaves=15, max_depth=4,
                     min_child_samples=100, subsample=0.7, colsample_bytree=0.6,
                     reg_alpha=1.0, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1)
        xgb_p = dict(n_estimators=700, learning_rate=0.02, max_depth=4, min_child_weight=25,
                     subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=6.0,
                     gamma=0.1, early_stopping_rounds=80, random_state=SEED, n_jobs=-1,
                     verbosity=0, tree_method='hist')
        cat_p = dict(iterations=800, learning_rate=0.02, depth=4, l2_leaf_reg=10,
                     random_seed=SEED, verbose=0, cat_features=cat_in, early_stopping_rounds=80)
    else:  # L1 / absolute error
        keys = ['lgb', 'cat']
        lgb_p = dict(objective='mae', n_estimators=1500, learning_rate=0.02, num_leaves=15,
                     max_depth=4, min_child_samples=100, subsample=0.7, colsample_bytree=0.6,
                     reg_alpha=1.0, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1)
        cat_p = dict(loss_function='MAE', iterations=1500, learning_rate=0.02, depth=4,
                     l2_leaf_reg=10, random_seed=SEED, verbose=0, cat_features=cat_in)

    oof = {k: np.zeros(len(X)) for k in keys}
    prd = {k: np.zeros(len(X_test)) for k in keys}
    for tr, va in kf.split(X):
        # LightGBM
        m = lgb.LGBMRegressor(**lgb_p)
        m.fit(Xe.iloc[tr], y.iloc[tr], eval_set=[(Xe.iloc[va], y.iloc[va])],
              callbacks=[lgb.early_stopping(80, verbose=False)])
        oof['lgb'][va] = m.predict(Xe.iloc[va]); prd['lgb'] += m.predict(Xte)/N
        # XGBoost (L2 only)
        if objective == 'L2':
            m = xgb.XGBRegressor(**xgb_p)
            m.fit(Xe.iloc[tr], y.iloc[tr], eval_set=[(Xe.iloc[va], y.iloc[va])], verbose=False)
            oof['xgb'][va] = m.predict(Xe.iloc[va]); prd['xgb'] += m.predict(Xte)/N
        # CatBoost (native categoricals)
        Xc_tr, Xc_va, Xc_te = X.iloc[tr].copy(), X.iloc[va].copy(), X_test.copy()
        for c in cat_in:
            Xc_tr[c] = Xc_tr[c].fillna('missing').astype(str)
            Xc_va[c] = Xc_va[c].fillna('missing').astype(str)
            Xc_te[c] = Xc_te[c].fillna('missing').astype(str)
        if objective == 'L2':
            m = CatBoostRegressor(**cat_p); m.fit(Xc_tr, y.iloc[tr], eval_set=(Xc_va, y.iloc[va]))
        else:
            m = CatBoostRegressor(**cat_p)
            m.fit(Xc_tr, y.iloc[tr], eval_set=(Xc_va, y.iloc[va]),
                  use_best_model=True, early_stopping_rounds=80)
        oof['cat'][va] = m.predict(Xc_va); prd['cat'] += m.predict(Xc_te)/N

    # Ridge (non-negative) stack, itself out-of-fold averaged
    S  = np.column_stack([oof[k] for k in keys])
    St = np.column_stack([prd[k] for k in keys])
    sp = np.zeros(len(X_test))
    for tr, va in kf.split(S):
        r = Ridge(alpha=1.0, positive=True); r.fit(S[tr], y.iloc[tr])
        sp += r.predict(St) / N
    return sp

# ── Run every seed once for each objective ───────────────────────────────────
L2_stacks, L1_stacks = [], []
for i, SEED in enumerate(SEEDS):
    print(f"Seed {SEED} ({i+1}/{len(SEEDS)})  L2 ...", flush=True)
    L2_stacks.append(stack_for_seed(SEED, 'L2'))
    print(f"Seed {SEED} ({i+1}/{len(SEEDS)})  L1 ...", flush=True)
    L1_stacks.append(stack_for_seed(SEED, 'L1'))

L2_stacks = np.array(L2_stacks); L1_stacks = np.array(L1_stacks)

# ── Combine ──────────────────────────────────────
v15     = L2_stacks[:3].mean(axis=0)               # 3-seed L2 average
v18_L2  = L2_stacks.mean(axis=0)                   # 5-seed L2 average
v18_L1  = L1_stacks.mean(axis=0)                   # 5-seed L1 average
v18_all = (v18_L2 + v18_L1) / 2
final   = np.clip((v18_all + v15) / 2, 0, 1)

sub['flood_risk_score'] = final
out = OUTPUT_DIR / 'submission_final.csv'
sub.to_csv(out, index=False)
print(f"\nSaved {out.name} | mean={final.mean():.4f} std={final.std():.4f} "
      f"range=[{final.min():.3f},{final.max():.3f}]")

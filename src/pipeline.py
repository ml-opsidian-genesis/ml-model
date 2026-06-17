import pandas as pd
import numpy as np

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact feature engineering used during model training."""
    df = df.copy()
    df['gen_month']      = pd.to_datetime(df['generation_date'], errors='coerce').dt.month
    df['gen_month_sin']  = np.sin(2 * np.pi * df['gen_month'] / 12)
    df['gen_month_cos']  = np.cos(2 * np.pi * df['gen_month'] / 12)
    df['rain_drainage_ratio'] = df['rainfall_7d_mm']      / (df['drainage_index'] + 1e-3)
    df['rain_elev_ratio']     = df['rainfall_7d_mm']      / (df['elevation_m'] + 1)
    df['river_elev_ratio']    = df['distance_to_river_m'] / (df['elevation_m'] + 1)
    df['ndwi_ndvi_diff']      = df['ndwi'] - df['ndvi']
    df['flood_x_rain']        = df['historical_flood_count'] * df['rainfall_7d_mm']
    df['inund_log1p']         = np.log1p(pd.to_numeric(df['inundation_area_sqm'], errors='coerce'))
    df['water_likely']        = (df['water_presence_flag'] == 'Likely').astype(int)
    df['flood_occurred']      = (df['flood_occurrence_current_event'] == 'Yes').astype(int)
    df['is_urban']            = (df['urban_rural'] == 'Urban').astype(int)
    df['road_quality_ord']    = df['road_quality'].map(
        {'No road access': 0, 'Poor (unpaved)': 1, 'Fair': 2, 'Good (paved)': 3})

    # Reproducible log1p transforms -- same formula at train and serve time.
    # pd.to_numeric first: a single-row frame built from a dict with None
    # values can leave the column as object dtype, and np.log1p (a numpy
    # ufunc) fails on object arrays instead of treating None/NaN as missing.
    df['distance_to_river_m_log1p']        = np.log1p(pd.to_numeric(df['distance_to_river_m'], errors='coerce'))
    df['population_density_per_km2_log1p'] = np.log1p(pd.to_numeric(df['population_density_per_km2'], errors='coerce'))
    df['rainfall_7d_mm_log1p']             = np.log1p(pd.to_numeric(df['rainfall_7d_mm'], errors='coerce'))
    df['monthly_rainfall_mm_log1p']        = np.log1p(pd.to_numeric(df['monthly_rainfall_mm'], errors='coerce'))
    df['nearest_hospital_km_log1p']        = np.log1p(pd.to_numeric(df['nearest_hospital_km'], errors='coerce'))
    df['nearest_evac_km_log1p']            = np.log1p(pd.to_numeric(df['nearest_evac_km'], errors='coerce'))
    return df


def add_district_te(df: pd.DataFrame, te_mean_map: dict, te_std_map: dict,
                     global_mean: float, global_std: float) -> pd.DataFrame:
    """Apply persisted district -> target mean/std encodings at serving time.
    Unknown districts fall back to the training-set global stats."""
    df = df.copy()
    df['district_te_mean'] = df['district'].map(te_mean_map).fillna(global_mean)
    df['district_te_std']  = df['district'].map(te_std_map).fillna(global_std)
    return df


_TRANSFORM_SOURCE_COL = {
    'elevation_m_yeojohnson': 'elevation_m',
    'drainage_index_yeojohnson': 'drainage_index',
    'ndvi_qmap': 'ndvi',
    'ndwi_qmap': 'ndwi',
    'built_up_percent_qmap': 'built_up_percent',
}


def apply_fitted_transforms(df: pd.DataFrame, transforms: dict) -> pd.DataFrame:
    """Apply persisted Yeo-Johnson / quantile transformers fit during training.
    `transforms` maps output column -> fitted sklearn transformer (or absent/
    None if that column wasn't fittable during training, in which case the
    output column is left NaN -- same as any other missing feature)."""
    df = df.copy()
    for out_col, src_col in _TRANSFORM_SOURCE_COL.items():
        tf = transforms.get(out_col)
        if tf is None or src_col not in df.columns:
            df[out_col] = np.nan
            continue
        values = df[[src_col]].astype(float).values
        df[out_col] = tf.transform(values).flatten()
    return df
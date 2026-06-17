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
    df['inund_log1p']         = np.log1p(df['inundation_area_sqm'])
    df['water_likely']        = (df['water_presence_flag'] == 'Likely').astype(int)
    df['flood_occurred']      = (df['flood_occurrence_current_event'] == 'Yes').astype(int)
    df['is_urban']            = (df['urban_rural'] == 'Urban').astype(int)
    df['road_quality_ord']    = df['road_quality'].map(
        {'No road access': 0, 'Poor (unpaved)': 1, 'Fair': 2, 'Good (paved)': 3})
    return df


def add_district_te(df: pd.DataFrame, te_map: dict) -> pd.DataFrame:
    """Apply a persisted district target-encoding map (reconstructs at serving
    time the same ``district_te_mean`` / ``district_te_std`` features used in
    training, instead of leaving them NaN)."""
    df = df.copy()
    mean_map = te_map['mean']
    std_map = te_map['std']
    df['district_te_mean'] = df['district'].map(mean_map).fillna(te_map['global_mean'])
    df['district_te_std'] = df['district'].map(std_map).fillna(te_map['global_std'])
    return df

"""Deterministic mock feature generation for the FloodGuard demo.

The competition demo does not wire a live weather feed; instead, for each
location we synthesise plausible daily feature values. Generation is
*deterministic per (location, date)* so a morning run is reproducible, yet it
varies day to day (driven by a random "weather regime") so risk scores move and
alerts fire realistically.

Returned dicts are compatible with the ``LocationInput`` schema used by the
``/predict`` endpoint, so the same model scoring path is reused.
"""
from __future__ import annotations

import hashlib
import random
from datetime import date as date_cls


_LANDCOVER = ["Urban", "Agriculture", "Forest", "Wetland", "Barren", "Grassland"]
_SOIL = ["Clay", "Loamy", "Sandy", "Silt", "Peat"]
_WATER_SUPPLY = ["Piped", "Municipal", "Well", "River"]
_ELECTRICITY = ["Yes", "Mixed", "No"]
_ROAD = ["Good (paved)", "Fair", "Poor (unpaved)", "No road access"]


def _seed(*parts: object) -> int:
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def location_profile(loc_id: str, district: str) -> dict:
    """Stable per-location traits that do not change from day to day."""
    rng = random.Random(_seed("profile", loc_id))
    elevation = round(rng.uniform(2.0, 120.0), 1)
    return {
        "district": district,
        "elevation_m": elevation,
        "distance_to_river_m": round(rng.uniform(40.0, 4200.0), 1),
        "drainage_index": round(rng.uniform(0.15, 0.85), 2),
        "historical_flood_count": rng.randint(0, 9),
        "landcover": rng.choice(_LANDCOVER),
        "soil_type": rng.choice(_SOIL),
        "water_supply": rng.choice(_WATER_SUPPLY),
        "electricity": rng.choice(_ELECTRICITY),
        "road_quality": rng.choice(_ROAD),
        "urban_rural": "Urban" if rng.random() < 0.45 else "Rural",
        "ndvi_base": round(rng.uniform(-0.2, 0.6), 3),
    }


def mock_features(loc_id: str, district: str, on_date: date_cls | None = None) -> dict:
    """Build a full feature dict for one location on one day."""
    on_date = on_date or date_cls.today()
    prof = location_profile(loc_id, district)
    rng = random.Random(_seed("daily", loc_id, on_date.isoformat()))

    # Daily weather regime drives rainfall (the dominant flood signal).
    regime = rng.choices(
        ["dry", "normal", "wet", "storm"],
        weights=[0.22, 0.40, 0.23, 0.15],
    )[0]
    rainfall = {
        "dry": rng.uniform(5, 40),
        "normal": rng.uniform(40, 95),
        "wet": rng.uniform(95, 165),
        "storm": rng.uniform(165, 320),
    }[regime]

    monthly_rainfall = rainfall * rng.uniform(2.5, 4.0)
    # Wetter days raise the water index and inundated area.
    ndwi = max(-1.0, min(1.0, -0.2 + rainfall / 320.0 + rng.uniform(-0.15, 0.15)))
    ndvi = max(-1.0, min(1.0, prof["ndvi_base"] + rng.uniform(-0.1, 0.1)))
    inundation = max(0.0, rainfall * rng.uniform(60, 220) * (1.0 / (prof["elevation_m"] + 5)))

    low_lying = prof["elevation_m"] < 25
    flooded = rainfall > 150 and (low_lying or prof["drainage_index"] < 0.4)

    return {
        "rainfall_7d_mm": round(rainfall, 1),
        "monthly_rainfall_mm": round(monthly_rainfall, 1),
        "elevation_m": prof["elevation_m"],
        "distance_to_river_m": prof["distance_to_river_m"],
        "drainage_index": prof["drainage_index"],
        "ndwi": round(ndwi, 3),
        "ndvi": round(ndvi, 3),
        "historical_flood_count": prof["historical_flood_count"],
        "inundation_area_sqm": round(inundation, 1),
        "district": prof["district"],
        "landcover": prof["landcover"],
        "soil_type": prof["soil_type"],
        "water_supply": prof["water_supply"],
        "electricity": prof["electricity"],
        "road_quality": prof["road_quality"],
        "urban_rural": prof["urban_rural"],
        "water_presence_flag": "Likely" if ndwi > 0.1 else "Unlikely",
        "flood_occurrence_current_event": "Yes" if flooded else "No",
        "is_good_to_live": "No" if flooded else "Yes",
        "generation_date": on_date.isoformat(),
        "_weather_regime": regime,
    }

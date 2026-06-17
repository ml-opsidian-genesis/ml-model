from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200

def test_predict_success():
    payload = {
        "district": "Colombo",
        "latitude": 6.9271,
        "longitude": 79.8612,
        "elevation_m": 10.0,
        "distance_to_river_m": 100.0,
        "landcover": "Urban",
        "soil_type": "Clay",
        "water_supply": "Yes",
        "electricity": "Yes",
        "road_quality": "Good",
        "population_density_per_km2": 5000,
        "built_up_percent": 0.8,
        "urban_rural": "Urban",
        "rainfall_7d_mm": 50.0,
        "monthly_rainfall_mm": 200.0,
        "drainage_index": 0.5,
        "ndvi": 0.3,
        "ndwi": 0.1,
        "water_presence_flag": "1",
        "historical_flood_count": 0,
        "infrastructure_score": 0.8,
        "nearest_hospital_km": 2.0,
        "nearest_evac_km": 1.0,
        "flood_occurrence_current_event": "0",
        "inundation_area_sqm": 0.0,
        "is_good_to_live": "1",
        "generation_date": "2023-01-01"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.json()

def test_predict_missing_fields():
    payload = {
        "district": "Kandy",
        "latitude": 7.2906,
        "longitude": 80.6337
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.json()

def test_predict_batch_success():
    payload = {
        "locations": [
            {
                "id": "loc1",
                "name": "Colombo Fort",
                "district": "Colombo",
                "latitude": 6.9271,
                "longitude": 79.8612,
                "elevation_m": 10.0,
                "distance_to_river_m": 100.0,
                "landcover": "Urban",
                "soil_type": "Clay",
                "water_supply": "Piped",
                "electricity": "Yes",
                "road_quality": "Good (paved)",
                "population_density_per_km2": 5000,
                "built_up_percent": 0.8,
                "urban_rural": "Urban",
                "rainfall_7d_mm": 50.0,
                "monthly_rainfall_mm": 200.0,
                "drainage_index": 0.5,
                "ndvi": 0.3,
                "ndwi": 0.1,
                "water_presence_flag": "Likely",
                "historical_flood_count": 0,
                "infrastructure_score": 0.8,
                "nearest_hospital_km": 2.0,
                "nearest_evac_km": 1.0,
                "flood_occurrence_current_event": "No",
                "inundation_area_sqm": 0.0,
                "is_good_to_live": "Yes",
                "generation_date": "2023-01-01",
            },
            {
                "id": "loc2",
                "name": "Kandy Center",
                "district": "Kandy",
                "latitude": 7.2906,
                "longitude": 80.6337,
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["count"] == 2
    assert len(body["results"]) == 2
    assert body["results"][0]["id"] == "loc1"
    assert "flood_risk_score" in body["results"][0]
    assert "weather_regime" not in body["results"][0]

def test_predict_batch_empty_locations():
    response = client.post("/predict/batch", json={"locations": []})
    assert response.status_code == 200
    assert response.json()["count"] == 0

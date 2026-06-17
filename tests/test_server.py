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
        "record_id": 1,
        "district": "Colombo",
        "place_name": "Test Place",
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
        "built_up_percent": 80.0,
        "urban_rural": "Urban",
        "rainfall_7d_mm": 50.0,
        "monthly_rainfall_mm": 200.0,
        "drainage_index": 0.5,
        "ndvi": 0.3,
        "ndwi": 0.1,
        "water_presence_flag": 1,
        "historical_flood_count": 0,
        "infrastructure_score": 0.8,
        "nearest_hospital_km": 2.0,
        "nearest_evac_km": 1.0,
        "flood_occurrence_current_event": 0,
        "inundation_area_sqm": 0.0,
        "is_good_to_live": 1,
        "reason_not_good_to_live": "",
        "is_synthetic": 0,
        "generation_date": "2023-01-01"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "flood_risk_score" in data
    assert "risk_level" in data
    assert "confidence" in data
    assert "model_version" in data
    assert 0.0 <= data["flood_risk_score"] <= 1.0

def test_predict_missing_fields():
    payload = {
        "record_id": 2,
        "district": "Kandy",
        "latitude": 7.2906,
        "longitude": 80.6337
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "flood_risk_score" in data
    assert data["risk_level"] in ["Low", "Moderate", "High"]

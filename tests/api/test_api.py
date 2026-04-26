"""
Unit tests for API endpoints and model service
Uses pytest and httpx for testing
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.app import app
from src.api.config import settings


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


# ============================================================================
# HEALTH CHECK TESTS
# ============================================================================

def test_health_check_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data


# ============================================================================
# SINGLE PREDICTION TESTS
# ============================================================================

def test_valid_single_prediction(client):
    """Test valid single patient prediction"""
    patient = {
        "age": 55,
        "sex": 1,
        "cp": 0,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 130,
        "exang": 0,
        "oldpeak": 2.6,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }
    
    response = client.post("/api/predict", json=patient)
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "risk_class" in data
    assert "confidence" in data
    assert "decision" in data
    assert data["decision"] in ["Accepted", "Deferred"]
    assert 0 <= data["confidence"] <= 1


def test_prediction_with_low_confidence(client):
    """Test deferred prediction (low confidence)"""
    # Patient data that might result in low confidence
    patient = {
        "age": 50,
        "sex": 0,
        "cp": 1,
        "trestbps": 120,
        "chol": 200,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 0.5,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }
    
    response = client.post("/api/predict", json=patient)
    assert response.status_code == 200
    # Deferred predictions are still successful, just have decision="Deferred"


def test_invalid_age(client):
    """Test validation: invalid age"""
    patient = {
        "age": -5,  # Invalid
        "sex": 1,
        "cp": 0,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 130,
        "exang": 0,
        "oldpeak": 2.6,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }
    
    response = client.post("/api/predict", json=patient)
    assert response.status_code == 422


def test_age_too_high(client):
    """Test validation: age > 150"""
    patient = {
        "age": 200,  # Invalid
        "sex": 1,
        "cp": 0,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 130,
        "exang": 0,
        "oldpeak": 2.6,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }
    
    response = client.post("/api/predict", json=patient)
    assert response.status_code == 422


def test_invalid_sex(client):
    """Test validation: invalid sex"""
    patient = {
        "age": 55,
        "sex": 5,  # Invalid (should be 0-1)
        "cp": 0,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 130,
        "exang": 0,
        "oldpeak": 2.6,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }
    
    response = client.post("/api/predict", json=patient)
    assert response.status_code == 422


def test_invalid_chest_pain(client):
    """Test validation: invalid chest pain type"""
    patient = {
        "age": 55,
        "sex": 1,
        "cp": 5,  # Invalid (should be 0-3)
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 130,
        "exang": 0,
        "oldpeak": 2.6,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }
    
    response = client.post("/api/predict", json=patient)
    assert response.status_code == 422


def test_extreme_cholesterol_valid(client):
    """Test extreme but valid cholesterol"""
    patient = {
        "age": 55,
        "sex": 1,
        "cp": 0,
        "trestbps": 140,
        "chol": 560,  # Max valid value
        "fbs": 0,
        "restecg": 1,
        "thalach": 130,
        "exang": 0,
        "oldpeak": 2.6,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }
    
    response = client.post("/api/predict", json=patient)
    assert response.status_code == 200


def test_missing_required_field(client):
    """Test validation: missing required field"""
    patient = {
        "age": 55,
        "sex": 1,
        "cp": 0,
        # Missing trestbps and other required fields
        "chol": 250,
    }
    
    response = client.post("/api/predict", json=patient)
    assert response.status_code == 422


# ============================================================================
# BATCH PREDICTION TESTS
# ============================================================================

def test_valid_batch_prediction(client):
    """Test valid batch prediction"""
    batch = {
        "patients": [
            {
                "age": 55,
                "sex": 1,
                "cp": 0,
                "trestbps": 140,
                "chol": 250,
                "fbs": 0,
                "restecg": 1,
                "thalach": 130,
                "exang": 0,
                "oldpeak": 2.6,
                "slope": 1,
                "ca": 0,
                "thal": 2
            },
            {
                "age": 45,
                "sex": 0,
                "cp": 1,
                "trestbps": 130,
                "chol": 200,
                "fbs": 0,
                "restecg": 0,
                "thalach": 140,
                "exang": 0,
                "oldpeak": 1.5,
                "slope": 1,
                "ca": 0,
                "thal": 2
            }
        ]
    }
    
    response = client.post("/api/predict/batch", json=batch)
    assert response.status_code == 200
    data = response.json()
    
    assert "batch_id" in data
    assert data["total_predictions"] == 2
    assert len(data["predictions"]) == 2


def test_batch_max_size(client):
    """Test batch max size (100)"""
    patients = [
        {
            "age": 50 + i % 30,
            "sex": i % 2,
            "cp": i % 4,
            "trestbps": 100 + (i % 100),
            "chol": 150 + (i % 400),
            "fbs": i % 2,
            "restecg": i % 3,
            "thalach": 70 + (i % 130),
            "exang": i % 2,
            "oldpeak": round((i % 60) / 10, 1),
            "slope": i % 3,
            "ca": i % 5,
            "thal": i % 4
        }
        for i in range(100)
    ]
    
    batch = {"patients": patients}
    response = client.post("/api/predict/batch", json=batch)
    assert response.status_code == 200


def test_batch_exceeds_max_size(client):
    """Test batch exceeds max size"""
    patients = [
        {
            "age": 50,
            "sex": 1,
            "cp": 0,
            "trestbps": 140,
            "chol": 250,
            "fbs": 0,
            "restecg": 1,
            "thalach": 130,
            "exang": 0,
            "oldpeak": 2.6,
            "slope": 1,
            "ca": 0,
            "thal": 2
        }
        for _ in range(101)  # Exceeds limit
    ]
    
    batch = {"patients": patients}
    response = client.post("/api/predict/batch", json=batch)
    assert response.status_code == 422


def test_empty_batch(client):
    """Test empty batch"""
    batch = {"patients": []}
    response = client.post("/api/predict/batch", json=batch)
    assert response.status_code == 422


# ============================================================================
# MODEL INFO TESTS
# ============================================================================

def test_model_info(client):
    """Test model info endpoint"""
    response = client.get("/api/model/info")
    assert response.status_code == 200
    data = response.json()
    
    assert "model_info" in data
    assert "feature_importance" in data
    assert "api_config" in data


# ============================================================================
# API ROOT TESTS
# ============================================================================

def test_api_root(client):
    """Test API root endpoint"""
    response = client.get("/api")
    assert response.status_code == 200
    data = response.json()
    
    assert "message" in data
    assert "documentation" in data
    assert "endpoints" in data


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_prediction_response_consistency(client):
    """Test that predictions are consistent for same input"""
    patient = {
        "age": 55,
        "sex": 1,
        "cp": 0,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 130,
        "exang": 0,
        "oldpeak": 2.6,
        "slope": 1,
        "ca": 0,
        "thal": 2
    }
    
    resp1 = client.post("/api/predict", json=patient)
    resp2 = client.post("/api/predict", json=patient)
    
    data1 = resp1.json()
    data2 = resp2.json()
    
    # Predictions should be identical for same input
    assert data1["risk_class"] == data2["risk_class"]
    assert data1["confidence"] == pytest.approx(data2["confidence"], abs=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

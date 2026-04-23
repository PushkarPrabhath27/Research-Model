"""
Test suite for GriceBench API
"""

import pytest
from fastapi.testclient import TestClient
from scripts.api import app

client = TestClient(app)


def test_health_check():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data


def test_detect_endpoint():
    """Test violation detection endpoint"""
    request_data = {
        "context": "What is quantum computing?",
        "response": "Quantum computing uses qubits for processing.",
        "threshold": 0.5
    }
    
    response = client.post("/detect", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "violations" in data
    assert "probabilities" in data
    assert "cooperative" in data
    assert all(maxim in data["violations"] for maxim in ["quantity", "quality", "relation", "manner"])


def test_detect_invalid_input():
    """Test detection with invalid input"""
    request_data = {
        "context": "",  # Empty context
        "response": "Test",
        "threshold": 0.5
    }
    
    response = client.post("/detect", json=request_data)
    assert response.status_code == 422  # Validation error


def test_repair_endpoint():
    """Test repair endpoint"""
    request_data = {
        "context": "What is AI?",
        "response": "AI is comp. sys.",
        "violations": ["manner"]
    }
    
    response = client.post("/repair", json=request_data)
    
    # May return 503 if model not loaded, or 200 if loaded
    assert response.status_code in [200, 503]


def test_metrics_endpoint():
    """Test Prometheus metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "gricebench" in response.text


def test_api_documentation():
    """Test API documentation is accessible"""
    response = client.get("/docs")
    assert response.status_code == 200
    
    response = client.get("/redoc")
    assert response.status_code == 200

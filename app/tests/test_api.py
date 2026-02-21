import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# M3: Automated Testing - Unit test for model utility/inference function
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "cats-vs-dogs-classifier"}

def test_predict_endpoint():
    # Test with a dummy file upload
    files = {"file": ("test_cat.jpg", b"dummy image data", "image/jpeg")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 200
    json_response = response.json()
    
    assert "prediction" in json_response
    assert "probability" in json_response
    assert json_response["prediction"] in ["Cat", "Dog"]
    assert 0.0 <= json_response["probability"] <= 1.0

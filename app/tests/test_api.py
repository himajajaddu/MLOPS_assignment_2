import io
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "cats-vs-dogs-classifier"
    assert "model_loaded" in body

def test_predict_endpoint_returns_503_when_model_missing():
    # In CI/without training, model likely isn't present; API should respond gracefully.
    img = Image.new("RGB", (64, 64))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test.jpg", buf.getvalue(), "image/jpeg")}
    response = client.post("/predict", files=files)

    # Either 503 (model not loaded) or 200 (if user trained and model exists)
    assert response.status_code in (200, 503)

    if response.status_code == 200:
        js = response.json()
        assert js["prediction"] in js["class_probabilities"]
        assert 0.0 <= js["probability"] <= 1.0

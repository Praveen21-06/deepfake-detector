import base64
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def encode_sample_image():
    # Create a small blank image for testing
    from PIL import Image
    import io
    img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

@pytest.fixture(scope="module")
def sample_image_b64():
    return encode_sample_image()

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"
    assert "model_version" in json_data

def test_model_info():
    response = client.get("/api/v1/model_info")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["model_name"] == "EfficientNet-B0"
    assert json_data["input_size"] == 224
    assert isinstance(json_data["classes"], list)

def test_predict_success(sample_image_b64):
    payload = {"image_data": sample_image_b64}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert "label" in json_data
    assert "confidence" in json_data
    assert "latency" in json_data

def test_predict_invalid():
    payload = {"image_data": "invalid_base64_string"}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 400

def test_batch_predict(sample_image_b64):
    payload = {"images": [sample_image_b64, sample_image_b64]}
    response = client.post("/api/v1/batch_predict", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert isinstance(json_data, list)
    assert len(json_data) == 2
    for item in json_data:
        assert "label" in item
        assert "confidence" in item
        assert "latency" in item

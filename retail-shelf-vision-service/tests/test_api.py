from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def create_test_image():
    image = Image.new("RGB", (224, 224), color=(255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint():
    image_buffer = create_test_image()

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", image_buffer, "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert "predicted_label" in body
    assert "confidence" in body
    assert "model_name" in body
    assert "model_version" in body
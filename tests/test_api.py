"""Automated Pytest Test Suite for BioVision AI API."""

import base64
import io

import numpy as np

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def create_sample_leaf_image(color=(34, 139, 34), size=(256, 256)) -> bytes:
    """Generates a synthetic RGB leaf image in memory using PIL or OpenCV."""
    if HAS_CV2:
        # OpenCV uses BGR
        bgr_color = (color[2], color[1], color[0])
        img_arr = np.full((size[1], size[0], 3), bgr_color, dtype=np.uint8)
        _success, buffer = cv2.imencode(".jpg", img_arr)
        return buffer.tobytes()
    elif HAS_PIL:
        img = Image.new("RGB", size, color=color)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()
    else:
        raise RuntimeError(
            "Neither OpenCV nor Pillow available for test image creation."
        )


def test_root_endpoint():
    """Validates root index metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "BioVision AI"
    assert "version" in data


def test_health_check():
    """Validates liveness probe."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data
    assert "inference_provider" in data


def test_list_classes():
    """Validates catalog of 38 classes."""
    response = client.get("/classes")
    assert response.status_code == 200
    classes = response.json()
    assert len(classes) == 38
    assert any(c["crop"] == "Tomato" for c in classes)


def test_predict_endpoint():
    """Validates /predict inference pipeline with multipart image upload."""
    img_bytes = create_sample_leaf_image()
    files = {"file": ("test_leaf.jpg", img_bytes, "image/jpeg")}

    response = client.post("/predict?top_k=3", files=files)
    assert response.status_code == 200
    data = response.json()

    assert "crop" in data
    assert "disease" in data
    assert "confidence" in data
    assert "severity" in data
    assert "yield_loss_risk_pct" in data
    assert "agronomic_recommendations" in data
    assert len(data["top_k_predictions"]) == 3
    assert data["inference_latency_ms"] > 0


def test_explain_gradcam():
    """Validates /explain Grad-CAM heatmap generation."""
    img_bytes = create_sample_leaf_image()
    files = {"file": ("test_leaf.jpg", img_bytes, "image/jpeg")}

    response = client.post("/explain", files=files)
    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert "heatmap_image_base64" in data
    # Verify base64 decodeable
    raw_heat = base64.b64decode(data["heatmap_image_base64"])
    assert len(raw_heat) > 100


def test_invalid_mime_type():
    """Validates that non-image file uploads are rejected with 415."""
    files = {"file": ("malicious.txt", b"Hello World", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 415


def test_prometheus_metrics():
    """Validates /metrics endpoint outputs Prometheus format."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "biovision_inference_total" in response.text
    assert "biovision_uptime_seconds" in response.text

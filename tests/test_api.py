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


def test_ui_endpoint():
    """Validates that the interactive Web UI is accessible."""
    response = client.get("/ui")
    assert response.status_code == 200
    assert "BioVision AI" in response.text


def test_sample_leaves_endpoint():
    """Validates that preset sample leaves are served correctly."""
    response = client.get("/sample_leaves/tomato_late_blight.jpg")
    assert response.status_code == 200
    assert len(response.content) > 1000


def test_drift_endpoint():
    """Validates that /drift returns valid status and distribution metrics."""
    response = client.get("/drift")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "drift_detected" in data
    assert "metrics" in data
    assert "sample_count" in data
    assert "recommendation" in data


def test_drift_telemetry_recording():
    """Validates that predictions dynamically update drift telemetry."""
    initial_drift = client.get("/drift").json()
    initial_count = initial_drift["sample_count"]

    # Trigger inference
    img_bytes = create_sample_leaf_image()
    files = {"file": ("test_leaf.jpg", img_bytes, "image/jpeg")}
    pred_res = client.post("/predict?top_k=3", files=files)
    assert pred_res.status_code == 200

    updated_drift = client.get("/drift").json()
    assert updated_drift["sample_count"] == initial_count + 1


def test_offline_disease_database():
    """Validates that the offline rural agronomic knowledge base is served and populated."""
    response = client.get("/ui/disease_db.json")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 38
    assert "Tomato___Late_blight" in data
    assert "immediate_actions" in data["Tomato___Late_blight"]


def test_offline_service_worker():
    """Validates that the PWA service worker script is accessible for client-side caching."""
    response = client.get("/ui/sw.js")
    assert response.status_code == 200
    assert "biovision-offline-v1" in response.text


def test_export_tflite_pipeline(tmp_path):
    """Validates that the TFLite edge export pipeline produces mobile artifacts."""
    from scripts.export_tflite import export_labels_and_metadata, export_tflite_models

    out_dir = str(tmp_path / "models")
    export_labels_and_metadata(out_dir)
    fp32_path, int8_path = export_tflite_models(out_dir)

    assert (tmp_path / "models" / "labels.txt").exists()
    assert (tmp_path / "models" / "metadata.json").exists()
    assert (tmp_path / "models" / "crop_disease_model.tflite").exists()
    assert (tmp_path / "models" / "crop_disease_model_quant.tflite").exists()

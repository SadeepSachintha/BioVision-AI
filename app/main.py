"""BioVision AI: Crop Disease Intelligence & Yield Optimizer
Production FastAPI Application Server.
"""

import io
import os
import time

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

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

from app.disease_db import get_agronomic_info
from app.drift_monitor import drift_monitor
from app.gradcam import gradcam_engine
from app.model_engine import CLASSES, engine
from app.schemas import (
    AgronomicRecommendations,
    DiseaseClassInfo,
    DriftReportResponse,
    GradCAMResponse,
    HealthResponse,
    PredictionItem,
    PredictResponse,
)

START_TIME = time.time()
INFERENCE_COUNT = 0
TOTAL_LATENCY_MS = 0.0

app = FastAPI(
    title="🌾 BioVision AI",
    description="Edge-optimized Crop Disease Intelligence & Yield Optimizer with Grad-CAM Explainability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Enable CORS for mobile and web agronomy dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mounts for Web Demo UI and sample leaves
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

SAMPLE_LEAVES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "sample_leaves"
)
if os.path.exists(SAMPLE_LEAVES_DIR):
    app.mount(
        "/sample_leaves", StaticFiles(directory=SAMPLE_LEAVES_DIR), name="sample_leaves"
    )


def validate_image_file(file: UploadFile):
    """Validates MIME type, file size, and parses raw image stream."""
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/bmp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format '{file.content_type}'. Must be one of: {allowed_types}",
        )

    try:
        contents = file.file.read()
        if len(contents) > 10 * 1024 * 1024:  # 10 MB limit
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File size exceeds maximum permitted payload (10 MB).",
            )
        if HAS_PIL:
            img = Image.open(io.BytesIO(contents))
            img.verify()
            return Image.open(io.BytesIO(contents))
        elif HAS_CV2:
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode failed to parse image stream")
            return img
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No image decoder backend (OpenCV or PIL) available on server.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupt or invalid image stream: {e!s}",
        )


@app.get("/", tags=["General"])
async def root():
    """Service metadata and root welcome index."""
    return {
        "service": "BioVision AI",
        "description": "Crop Disease Intelligence & Yield Optimizer",
        "version": "1.0.0",
        "web_ui": "/ui",
        "documentation": "/docs",
        "health_check": "/health",
        "classes_endpoint": "/classes",
        "drift_monitoring": "/drift",
        "offline_rural_database": "/ui/disease_db.json",
        "offline_service_worker": "/ui/sw.js",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health & Operations"])
async def health():
    """Liveness and readiness probe for Kubernetes / Docker orchestrators."""
    uptime = time.time() - START_TIME
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=engine.is_loaded,
        inference_provider=engine.provider,
        model_path=engine.model_path,
        uptime_seconds=round(uptime, 2),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(
    file: UploadFile = File(
        ..., description="Foliar photograph of the affected crop leaf"
    ),
    top_k: int = Query(
        3, ge=1, le=10, description="Number of candidate classes to return"
    ),
):
    """Executes high-speed ONNX inference, returning disease classification and intervention protocols."""
    global INFERENCE_COUNT, TOTAL_LATENCY_MS

    image = validate_image_file(file)
    predictions, latency_ms = engine.predict(image, top_k=top_k)

    INFERENCE_COUNT += 1
    TOTAL_LATENCY_MS += latency_ms

    top_pred = predictions[0]

    # Asynchronously record inference telemetry for drift monitoring
    all_probs = [p["confidence"] for p in predictions]
    drift_monitor.record_inference(image, top_pred["confidence"], all_probs)

    agronomic_info = get_agronomic_info(top_pred["class_name"])

    recommendations = AgronomicRecommendations(
        immediate_actions=agronomic_info.get("immediate_actions", []),
        chemical_control=agronomic_info.get(
            "chemical_control", "Consult registered agronomist."
        ),
        organic_bio_control=agronomic_info.get(
            "organic_bio_control", "Consult organic certification guidelines."
        ),
        preventative_practices=agronomic_info.get(
            "preventative_practices", "Maintain good sanitation and rotation."
        ),
    )

    items = [
        PredictionItem(
            class_name=p["class_name"],
            crop=p["crop"],
            condition=p["condition"],
            confidence=p["confidence"],
        )
        for p in predictions
    ]

    return PredictResponse(
        crop=agronomic_info["crop"],
        disease=agronomic_info["disease"],
        scientific_name=agronomic_info.get("scientific_name"),
        confidence=top_pred["confidence"],
        top_k_predictions=items,
        severity=agronomic_info.get("severity", "MODERATE"),
        yield_loss_risk_pct=agronomic_info.get("yield_loss_risk_pct", "Undetermined"),
        agronomic_recommendations=recommendations,
        inference_latency_ms=latency_ms,
    )


@app.post("/explain", response_model=GradCAMResponse, tags=["Explainability"])
async def explain(
    file: UploadFile = File(
        ..., description="Foliar leaf image to analyze via Grad-CAM"
    )
):
    """Executes Grad-CAM visual explainability, producing a base64 thermodynamic attention heatmap overlay."""
    image = validate_image_file(file)
    predictions, latency_ms = engine.predict(image, top_k=1)
    top_pred = predictions[0]

    heatmap_b64 = gradcam_engine.overlay_heatmap(image, alpha=0.45)

    note = (
        f"Grad-CAM visualizer isolated focal lesions supporting '{top_pred['condition']}' "
        f"on {top_pred['crop']} leaf surface. High temperature regions (red/yellow) indicate primary activation zones."
    )

    return GradCAMResponse(
        prediction=top_pred["class_name"],
        confidence=top_pred["confidence"],
        inference_latency_ms=latency_ms,
        heatmap_image_base64=heatmap_b64,
        explanation_note=note,
    )


@app.get("/classes", response_model=list[DiseaseClassInfo], tags=["Catalog"])
async def list_classes():
    """Enumerates all 38 agricultural crop and pathogen classes supported by the model."""
    catalog = []
    for cls_name in CLASSES:
        info = get_agronomic_info(cls_name)
        catalog.append(
            DiseaseClassInfo(
                class_name=cls_name,
                crop=info["crop"],
                condition=info["disease"],
                severity=info.get("severity", "MODERATE"),
            )
        )
    return catalog


@app.get("/drift", response_model=DriftReportResponse, tags=["MLOps & Monitoring"])
async def drift():
    """Evaluates foliar covariate shift and prediction entropy drift over rolling inference window."""
    return drift_monitor.evaluate_drift()


@app.get("/metrics", response_class=PlainTextResponse, tags=["Telemetry"])
async def metrics():
    """Prometheus-compatible plain text telemetry metrics."""
    avg_latency = (TOTAL_LATENCY_MS / INFERENCE_COUNT) if INFERENCE_COUNT > 0 else 0.0
    uptime = time.time() - START_TIME
    drift_eval = drift_monitor.evaluate_drift()

    lines = [
        "# HELP biovision_uptime_seconds Total runtime of the FastAPI server",
        "# TYPE biovision_uptime_seconds gauge",
        f"biovision_uptime_seconds {uptime:.2f}",
        "# HELP biovision_inference_total Number of leaf inference requests processed",
        "# TYPE biovision_inference_total counter",
        f"biovision_inference_total {INFERENCE_COUNT}",
        "# HELP biovision_avg_latency_ms Average latency in milliseconds",
        "# TYPE biovision_avg_latency_ms gauge",
        f"biovision_avg_latency_ms {avg_latency:.2f}",
        "# HELP biovision_model_loaded Binary indicator if ONNX model is active",
        "# TYPE biovision_model_loaded gauge",
        f"biovision_model_loaded {1 if engine.is_loaded else 0}",
        "# HELP biovision_drift_detected Binary indicator if covariate or concept drift detected",
        "# TYPE biovision_drift_detected gauge",
        f"biovision_drift_detected {1 if drift_eval['drift_detected'] else 0}",
        "# HELP biovision_rolling_entropy Current rolling Shannon prediction entropy",
        "# TYPE biovision_rolling_entropy gauge",
        f"biovision_rolling_entropy {drift_eval['metrics']['rolling_mean_entropy']:.3f}",
    ]
    return "\n".join(lines) + "\n"

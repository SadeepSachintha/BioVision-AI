from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    organization: str = Field(default="SS Labz", examples=["SS Labz"])
    version: str = Field(..., examples=["1.0.0"])
    model_loaded: bool = Field(..., examples=[True])
    inference_provider: str = Field(..., examples=["CPUExecutionProvider"])
    model_path: str = Field(..., examples=["models/crop_disease_model.onnx"])
    uptime_seconds: float = Field(..., examples=[120.45])


class PredictionItem(BaseModel):
    class_name: str
    crop: str
    condition: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class AgronomicRecommendations(BaseModel):
    immediate_actions: list[str]
    chemical_control: str
    organic_bio_control: str
    preventative_practices: str


class PredictResponse(BaseModel):
    crop: str
    disease: str
    scientific_name: str | None = None
    confidence: float
    top_k_predictions: list[PredictionItem]
    severity: str
    yield_loss_risk_pct: str
    agronomic_recommendations: AgronomicRecommendations
    inference_latency_ms: float


class GradCAMResponse(BaseModel):
    prediction: str
    confidence: float
    inference_latency_ms: float
    heatmap_image_base64: str
    explanation_note: str


class DiseaseClassInfo(BaseModel):
    class_name: str
    crop: str
    condition: str
    severity: str


class DriftMetrics(BaseModel):
    rolling_mean_confidence: float
    baseline_confidence: float | None = None
    rolling_mean_entropy: float
    baseline_entropy: float | None = None
    rolling_green_ratio: float
    baseline_green_ratio: float | None = None
    rolling_luminance: float
    baseline_luminance: float | None = None


class DriftReportResponse(BaseModel):
    status: str
    sample_count: int
    window_capacity: int
    drift_detected: bool
    message: str
    metrics: DriftMetrics
    ks_covariate_shift: float
    ks_luminance_shift: float | None = None
    recommendation: str

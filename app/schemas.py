from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
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
    immediate_actions: List[str]
    chemical_control: str
    organic_bio_control: str
    preventative_practices: str


class PredictResponse(BaseModel):
    crop: str
    disease: str
    scientific_name: Optional[str] = None
    confidence: float
    top_k_predictions: List[PredictionItem]
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

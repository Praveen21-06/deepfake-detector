from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    image_data: str  # Base64-encoded image

class BatchPredictRequest(BaseModel):
    images: List[str]  # List of base64-encoded images

class PredictResponse(BaseModel):
    label: str
    confidence: float
    latency: float

class HealthResponse(BaseModel):
    status: str
    model_version: str

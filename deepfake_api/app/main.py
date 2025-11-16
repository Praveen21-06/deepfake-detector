from fastapi import FastAPI, HTTPException
from typing import List
from app.schemas import PredictRequest, BatchPredictRequest, PredictResponse, HealthResponse
from app.model import predict_image

app = FastAPI(
    title="Deepfake Detection API",
    version="1.0.0",
    description="API for image-level deepfake detection using EfficientNet-B0"
)

@app.get("/api/v1/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", model_version="1.0.0")

@app.get("/api/v1/model_info")
def model_info():
    return {
        "model_name": "EfficientNet-B0",
        "input_size": 224,
        "classes": ["real", "fake"]
    }

@app.post("/api/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        result = predict_image(req.image_data)
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

@app.post("/api/v1/batch_predict", response_model=List[PredictResponse])
def batch_predict(req: BatchPredictRequest):
    results = []
    for b64 in req.images:
        try:
            res = predict_image(b64)
        except Exception:
            res = {"label": "error", "confidence": 0.0, "latency": 0.0}
        results.append(PredictResponse(**res))
    return results

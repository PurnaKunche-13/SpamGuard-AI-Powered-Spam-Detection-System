from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.predictor import PredictorManager
from backend.app.schemas import PredictionRequest, PredictionResponse, TrainingResponse
from ml.src.train import train_model


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "ml" / "data" / "sample_messages.csv"
FRONTEND_INDEX = BASE_DIR / "website" / "index.html"
STATIC_DIR = BASE_DIR / "website" / "public"

app = FastAPI(
    title="SpamGuard API",
    description="AI powered spam detection service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = PredictorManager()
app.mount("/public", StaticFiles(directory=STATIC_DIR), name="public")


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "model_ready": predictor.is_ready()}


@app.get("/")
def home() -> FileResponse:
    return FileResponse(FRONTEND_INDEX)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Model artifacts not found. Train the model first.")

    result = predictor.load().predict(payload.message)
    return PredictionResponse(**result)


@app.post("/train", response_model=TrainingResponse)
def train() -> TrainingResponse:
    if not DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="Training dataset not found.")

    metrics = train_model(DATA_PATH)
    predictor.reload()
    return TrainingResponse(
        status="trained",
        accuracy=metrics["accuracy"],
        samples=metrics["samples"],
    )

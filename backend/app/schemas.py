from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Message text to classify")


class PredictionResponse(BaseModel):
    label: str
    spam_probability: float
    ham_probability: float


class TrainingResponse(BaseModel):
    status: str
    accuracy: float
    samples: int


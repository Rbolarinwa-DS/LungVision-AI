from pydantic import BaseModel
from typing import Dict


class DiseasePrediction(BaseModel):
    probability: float
    detected: bool


class PredictionResponse(BaseModel):
    predictions: Dict[str, DiseasePrediction]
    gradcam: str
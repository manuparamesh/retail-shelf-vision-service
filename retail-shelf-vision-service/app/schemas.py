from pydantic import BaseModel


class PredictionResponse(BaseModel):
    predicted_label: str
    confidence: float
    model_name: str
    model_version: str
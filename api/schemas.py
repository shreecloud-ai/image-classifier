from pydantic import BaseModel
from typing import Optional

class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    latency_ms: float

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

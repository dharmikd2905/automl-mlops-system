from typing import Any, Dict, Optional

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    prediction: Any
    raw_value: Optional[float] = None
    task_type: str
    confidence: Optional[float] = None
    model_used: str

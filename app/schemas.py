from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    engine_id: int
    window: List[List[float]]

class PredictResponse(BaseModel):
    predicted_RUL: float
    engine_id: int



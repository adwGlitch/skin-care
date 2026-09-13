from pydantic import BaseModel, Field
from typing import List, Optional

class PredictionClass(BaseModel):
    class_name: str = Field(alias="class")
    name: str
    probability: float

class ImageQuality(BaseModel):
    score: float
    status: str
    messages: List[str]

class AnalysisResponse(BaseModel):
    status: str
    message: Optional[str] = None
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    top_predictions: Optional[List[PredictionClass]] = None
    image_quality: Optional[ImageQuality] = None
    uncertainty: Optional[str] = None
    heatmap_url: Optional[str] = None
    model_version: Optional[str] = None
    recommendation_level: Optional[str] = None
    scan_id: Optional[str] = None

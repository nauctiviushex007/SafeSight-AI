from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

Status = Literal["pending_review", "confirmed", "rejected"]


class IncidentCreate(BaseModel):
    category: Literal["no_helmet", "no_vest", "restricted_zone"]
    confidence: float = Field(ge=0, le=1)
    certainty: Literal["likely", "uncertain"] = "likely"
    source: str | None = None
    evidence_ref: str | None = None
    description: str = ""
    recommended_action: str | None = None


class IncidentReview(BaseModel):
    status: Status
    reviewer_notes: str | None = None


class IncidentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    timestamp: datetime
    category: str
    severity: str
    confidence: float
    certainty: str
    evidence_ref: str | None
    source: str | None
    description: str
    status: str
    reviewer_notes: str | None
    recommended_action: str | None
    reviewed_at: datetime | None


class DetectionIn(BaseModel):
    label: Literal["person", "helmet", "vest"]
    confidence: float = Field(ge=0, le=1)
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


class EvaluateRequest(BaseModel):
    detections: list[DetectionIn]
    restricted_zones: list[list[tuple[float, float]]] = []

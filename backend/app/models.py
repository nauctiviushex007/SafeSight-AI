import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Float, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from .database import Base


def new_incident_id() -> str:
    return "INC-" + uuid.uuid4().hex[:8].upper()


def utcnow():
    return datetime.now(timezone.utc)


class Incident(Base):
    """A potential violation. Starts as machine-detected; a human must confirm it."""
    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_incident_id)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    category: Mapped[str] = mapped_column(String, index=True)      # no_helmet / no_vest / restricted_zone
    severity: Mapped[str] = mapped_column(String)                  # from config.SEVERITY_RULES
    confidence: Mapped[float] = mapped_column(Float)
    certainty: Mapped[str] = mapped_column(String, default="likely")  # likely | uncertain
    evidence_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    source: Mapped[str | None] = mapped_column(String, nullable=True)  # file name or camera id
    description: Mapped[str] = mapped_column(Text, default="")
    # auto_detected -> pending_review -> confirmed | rejected
    status: Mapped[str] = mapped_column(String, default="pending_review", index=True)
    reviewer_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    recommended_action: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

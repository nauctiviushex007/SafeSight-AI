import csv, io
from datetime import datetime, timezone
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import select, func, or_
from sqlalchemy.orm import Session

from . import config, models, schemas, rules_engine
from .database import Base, engine, get_db

Base.metadata.create_all(engine)

app = FastAPI(title="SafeSight AI", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

FOLLOW_UP = {
    "no_helmet": "Verify on footage; if confirmed, brief the worker and supervisor per your site safety procedure.",
    "no_vest": "Verify on footage; if confirmed, remind the worker of high-visibility requirements.",
    "restricted_zone": "Verify zone boundary and footage; if confirmed, notify the area supervisor promptly.",
}


@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


# ---------- Incidents ----------
@app.post("/api/incidents", response_model=schemas.IncidentOut, status_code=201)
def create_incident(body: schemas.IncidentCreate, db: Session = Depends(get_db)):
    inc = models.Incident(
        **body.model_dump(exclude={"recommended_action"}),
        severity=config.SEVERITY_RULES[body.category],
        recommended_action=body.recommended_action or FOLLOW_UP[body.category],
    )
    db.add(inc); db.commit(); db.refresh(inc)
    return inc


@app.get("/api/incidents", response_model=list[schemas.IncidentOut])
def list_incidents(q: str | None = None, status: str | None = None, category: str | None = None,
                   severity: str | None = None,
                   sort: str = Query("-timestamp", pattern="^-?(timestamp|confidence|severity)$"),
                   db: Session = Depends(get_db)):
    stmt = select(models.Incident)
    if q:
        like = f"%{q}%"
        stmt = stmt.where(or_(models.Incident.id.ilike(like), models.Incident.description.ilike(like),
                              models.Incident.source.ilike(like)))
    if status: stmt = stmt.where(models.Incident.status == status)
    if category: stmt = stmt.where(models.Incident.category == category)
    if severity: stmt = stmt.where(models.Incident.severity == severity)
    col = getattr(models.Incident, sort.lstrip("-"))
    stmt = stmt.order_by(col.desc() if sort.startswith("-") else col.asc())
    return db.scalars(stmt).all()


@app.get("/api/incidents/export/csv")
def export_csv(db: Session = Depends(get_db)):
    rows = db.scalars(select(models.Incident).order_by(models.Incident.timestamp.desc())).all()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "timestamp", "category", "severity", "confidence", "certainty", "status", "source", "reviewer_notes"])
    for r in rows:
        w.writerow([r.id, r.timestamp.isoformat(), r.category, r.severity, round(r.confidence, 3),
                    r.certainty, r.status, r.source or "", r.reviewer_notes or ""])
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=incidents.csv"})


@app.get("/api/incidents/{incident_id}", response_model=schemas.IncidentOut)
def get_incident(incident_id: str, db: Session = Depends(get_db)):
    inc = db.get(models.Incident, incident_id)
    if not inc: raise HTTPException(404, "Incident not found")
    return inc


@app.patch("/api/incidents/{incident_id}/review", response_model=schemas.IncidentOut)
def review_incident(incident_id: str, body: schemas.IncidentReview, db: Session = Depends(get_db)):
    """Human review step: only a person can mark an incident confirmed or rejected."""
    inc = db.get(models.Incident, incident_id)
    if not inc: raise HTTPException(404, "Incident not found")
    inc.status = body.status
    inc.reviewer_notes = body.reviewer_notes
    inc.reviewed_at = datetime.now(timezone.utc) if body.status != "pending_review" else None
    db.commit(); db.refresh(inc)
    return inc


# ---------- Dashboard (real DB data only) ----------
@app.get("/api/dashboard/summary")
def dashboard_summary(db: Session = Depends(get_db)):
    I = models.Incident
    count = lambda *c: db.scalar(select(func.count()).select_from(I).where(*c)) or 0
    by_cat = dict(db.execute(select(I.category, func.count()).group_by(I.category)).all())
    by_sev = dict(db.execute(select(I.severity, func.count()).group_by(I.severity)).all())
    by_day = db.execute(select(func.date(I.timestamp), func.count()).group_by(func.date(I.timestamp))
                        .order_by(func.date(I.timestamp))).all()
    return {
        "total_events": count(),
        "ppe_violations": count(I.category.in_(["no_helmet", "no_vest"])),
        "restricted_zone_alerts": count(I.category == "restricted_zone"),
        "pending_review": count(I.status == "pending_review"),
        "confirmed": count(I.status == "confirmed"),
        "by_category": by_cat,
        "by_severity": by_sev,
        "events_over_time": [{"date": str(d), "count": c} for d, c in by_day],
        "has_data": count() > 0,
    }


# ---------- Rules engine endpoint ----------
@app.post("/api/rules/evaluate")
def evaluate_rules(body: schemas.EvaluateRequest):
    dets = [rules_engine.Detection(d.label, d.confidence, d.bbox) for d in body.detections]
    result = rules_engine.evaluate(dets, body.restricted_zones)
    return {"violations": [v.__dict__ for v in result],
            "note": "Potential violations only; human review required."}


# ---------- Phase 2+: vision, RAG, SVM, frontend ----------
import os
from fastapi.staticfiles import StaticFiles
from .extra import r as extra_router
app.include_router(extra_router)
os.makedirs("data/outputs", exist_ok=True)
app.mount("/files", StaticFiles(directory="data/outputs"), name="files")
_fe = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
app.mount("/", StaticFiles(directory=_fe, html=True), name="frontend")

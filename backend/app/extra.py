import os, time
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from . import models, config, vision, rag, svm_module
from .database import get_db

r = APIRouter(prefix="/api")
MAX = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024
IMG, VID = (".jpg", ".jpeg", ".png", ".bmp", ".webp"), (".mp4", ".avi", ".mov", ".mkv")
FOLLOW = {"no_helmet": "Verify footage, then brief worker/supervisor per site procedure.",
          "no_vest": "Verify footage, then remind worker of hi-vis requirements.",
          "restricted_zone": "Verify zone and footage, then notify area supervisor."}


async def _read(f: UploadFile, exts):
    if not f.filename.lower().endswith(exts): raise HTTPException(415, f"Unsupported file type. Allowed: {', '.join(exts)}")
    data = await f.read()
    if len(data) > MAX: raise HTTPException(413, "File too large")
    return data


@r.post("/vision/analyze")
async def analyze(file: UploadFile = File(...), conf: float = Form(vision.CONF), save: bool = Form(True), db: Session = Depends(get_db)):
    data = await _read(file, IMG + VID)
    try:
        res = (vision.analyze_image if file.filename.lower().endswith(IMG) else vision.analyze_video)(data, conf)
    except ValueError as e: raise HTTPException(400, str(e))
    except RuntimeError as e: raise HTTPException(503, str(e))
    except Exception as e: raise HTTPException(500, f"Inference failed: {e}")
    ids = []
    if save:
        for v in res["violations"]:
            inc = models.Incident(category=v.category, severity=v.severity, confidence=v.confidence, certainty=v.certainty,
                                  source=file.filename, evidence_ref=res["output"], description=v.description,
                                  recommended_action=FOLLOW[v.category])
            db.add(inc); db.flush(); ids.append(inc.id)
        db.commit()
    res["violations"] = [v.__dict__ for v in res["violations"]]; res["incident_ids"] = ids
    return res


@r.get("/models/status")
def model_status():
    # Diagnostics must never initialize ML runtimes or read evaluation artifacts.
    # YOLO/PyTorch initialization happens only when an inference request needs it.
    yolo = vision.status(check_load=False)
    return {
        "yolo": yolo,
        "class_map": vision.CLASS_MAP,
        "svm": None,
        "svm_status": "Metrics are not loaded by this lightweight endpoint. Open Model Evaluation to train or inspect SVM results.",
        "limitations": [
            "Generic COCO models cannot detect helmets/vests.",
            "PPE-to-person association is geometric and can be wrong.",
            "Zones assume a fixed camera; perspective is not corrected."
        ]
    }


@r.post("/svm/train")
def svm_train():
    try: return svm_module.train()
    except Exception as e: raise HTTPException(400, str(e))


@r.post("/rag/upload")
async def rag_upload(file: UploadFile = File(...)):
    data = await _read(file, (".pdf", ".txt"))
    try: return {"document": file.filename, "chunks": rag.ingest(file.filename, data)}
    except ValueError as e: raise HTTPException(400, str(e))


@r.get("/rag/documents")
def rag_docs(): return rag.documents()


@r.post("/rag/ask")
def rag_ask(body: dict): 
    q = (body.get("question") or "").strip()
    if not q: raise HTTPException(400, "Question required")
    return rag.answer(q)


@r.get("/analytics")
def analytics(days: int = 30, category: str | None = None, db: Session = Depends(get_db)):
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
    q = select(models.Incident).where(models.Incident.timestamp >= since)
    if category: q = q.where(models.Incident.category == category)
    rows = db.scalars(q).all(); cnt = lambda key: {k: sum(1 for x in rows if key(x) == k) for k in {key(x) for x in rows}}
    return {"total": len(rows), "by_day": dict(sorted(cnt(lambda x: x.timestamp.date().isoformat()).items())),
            "by_category": cnt(lambda x: x.category), "by_severity": cnt(lambda x: x.severity), "by_status": cnt(lambda x: x.status)}

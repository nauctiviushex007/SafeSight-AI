"""YOLO inference. Class names from YOUR model are mapped to app categories via CLASS_MAP."""
import os, json, time, tempfile, uuid
import cv2, numpy as np
from . import config
from .rules_engine import Detection, evaluate

MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/ppe.pt")
CONF = float(os.getenv("YOLO_CONF", "0.35"))
EVERY = int(os.getenv("VIDEO_SAMPLE_EVERY", "15"))
CLASS_MAP = json.loads(os.getenv("CLASS_MAP", json.dumps({
    "person": "person", "helmet": "helmet", "hardhat": "helmet", "safety helmet": "helmet",
    "vest": "vest", "safety vest": "vest", "safety-vest": "vest"})))
OUT = "data/outputs"; os.makedirs(OUT, exist_ok=True)
_model = None


def load():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"No YOLO model at '{MODEL_PATH}'. Train/download a PPE model (.pt) and set YOLO_MODEL_PATH.")
        from ultralytics import YOLO
        _model = YOLO(MODEL_PATH)
    return _model


def status():
    try:
        m = load()
        names = list(m.names.values())
        sup = sorted({CLASS_MAP[n.lower()] for n in names if n.lower() in CLASS_MAP})
        return {"loaded": True, "path": MODEL_PATH, "classes": names, "supported_categories": sup}
    except Exception as e:
        return {"loaded": False, "path": MODEL_PATH, "error": str(e), "classes": [], "supported_categories": []}


def detect(frame, conf=CONF):
    r = load().predict(frame, conf=conf, verbose=False)[0]
    return [{"raw": r.names[int(b.cls)], "label": CLASS_MAP.get(r.names[int(b.cls)].lower()),
             "confidence": round(float(b.conf), 3), "bbox": [round(float(x), 1) for x in b.xyxy[0]]} for b in r.boxes]


def violations_for(dets, zones):
    sup = status()["supported_categories"]
    ds = [Detection(d["label"], d["confidence"], tuple(d["bbox"])) for d in dets if d["label"]]
    return evaluate(ds, zones, supports=tuple(sup))


def annotate(frame, dets, viols):
    bad = {v.person_bbox for v in viols}
    for d in dets:
        x1, y1, x2, y2 = map(int, d["bbox"])
        col = (60, 60, 255) if tuple(d["bbox"]) in bad else (200, 160, 40)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, f'{d["raw"]} {d["confidence"]:.2f}', (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    return frame


def analyze_image(data: bytes, conf=CONF, zones=None):
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Corrupt or unsupported image file")
    t = time.time()
    dets = detect(frame, conf); viols = violations_for(dets, zones)
    name = f"{uuid.uuid4().hex[:10]}.jpg"
    cv2.imwrite(os.path.join(OUT, name), annotate(frame, dets, viols))
    return {"detections": dets, "violations": viols, "output": f"/files/{name}",
            "ms": round((time.time() - t) * 1000)}


def analyze_video(data: bytes, conf=CONF, zones=None):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); tmp.write(data); tmp.close()
    cap = cv2.VideoCapture(tmp.name)
    if not cap.isOpened():
        raise ValueError("Corrupt or unsupported video file")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    name = f"{uuid.uuid4().hex[:10]}.mp4"; writer = None
    all_d, best, i, t = [], {}, 0, time.time()
    while i < 3000:
        ok, f = cap.read()
        if not ok: break
        if i % EVERY == 0:
            dets = detect(f, conf); v = violations_for(dets, zones)
            for x in v:
                if x.category not in best or x.confidence > best[x.category].confidence: best[x.category] = x
            all_d += dets
            f = annotate(f, dets, v)
            if writer is None:
                h, w = f.shape[:2]
                writer = cv2.VideoWriter(os.path.join(OUT, name), cv2.VideoWriter_fourcc(*"mp4v"), max(fps / EVERY, 1), (w, h))
            writer.write(f)
        i += 1
    cap.release(); writer and writer.release()
    os.unlink(tmp.name)
    return {"detections": all_d[:500], "violations": list(best.values()), "output": f"/files/{name}" if writer else None,
            "ms": round((time.time() - t) * 1000), "frames_read": i}

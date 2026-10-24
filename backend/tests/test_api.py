import os
os.environ["DATABASE_URL"] = "sqlite:///./test.db"
from fastapi.testclient import TestClient
from app.main import app

c = TestClient(app)


def test_incident_review_flow():
    r = c.post("/api/incidents", json={"category": "no_helmet", "confidence": 0.8, "source": "cam1"})
    assert r.status_code == 201 and r.json()["status"] == "pending_review"
    iid = r.json()["id"]
    r = c.patch(f"/api/incidents/{iid}/review", json={"status": "confirmed", "reviewer_notes": "ok"})
    assert r.json()["status"] == "confirmed"
    assert c.get("/api/dashboard/summary").json()["has_data"] is True
    assert c.get("/api/incidents/export/csv").status_code == 200

# SafeSight AI — Phase 1: Backend Core

## Run (Windows PowerShell)
```
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs for the interactive API.

## Test
```
pytest -q
```

## What's here
- `app/rules_engine.py` — detections -> potential violations (pure Python, unit-tested)
- `app/models.py` — Incident table with human review workflow
- `app/main.py` — incidents CRUD/review, dashboard stats, CSV export, rules endpoint
- Dashboard numbers come only from the database; an empty DB returns `has_data: false`.

## Roadmap
2 YOLO vision pipeline · 3 RAG assistant · 4 SVM module · 5 React frontend · 6 Evaluation, Docker

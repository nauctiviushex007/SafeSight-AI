# SafeSight AI — full project

## Run (Windows PowerShell)
```
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000 (UI) and /docs (API).

## Enable detection
Put a PPE-trained YOLO model at `backend/models/ppe.pt` (e.g. train on a Roboflow/Kaggle "PPE / hard hat" dataset with `yolo detect train`). If its class names differ, set `CLASS_MAP` in `.env`, e.g. `{"hard-hat":"helmet","person":"person"}`. A generic COCO model only detects "person", so helmet/vest rules stay disabled automatically.

## Other
- SVM: `python -m app.svm_module` (synthetic data by default; add `data/svm_dataset.csv` with columns person_conf,height_ratio,ppe_overlap,brightness,needs_review for real data)
- RAG: upload a PDF/TXT on the Safety Assistant page; set ANTHROPIC_API_KEY for LLM answers, otherwise extractive answers are used.
- Tests: `pytest -q`

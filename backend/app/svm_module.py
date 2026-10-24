"""
SVM experiment: given engineered features of a (person, PPE) detection, predict whether the
event NEEDS HUMAN REVIEW (1) or is a confident detection (0). It complements YOLO, it does not replace it.
DEFAULT DATA IS SYNTHETIC (pipeline test only) -> results are NOT real-world safety performance.
Replace data/svm_dataset.csv with real labelled data (same columns) to train properly.
Run: python -m app.svm_module
"""
import os, json
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix

FEATURES = ["person_conf", "height_ratio", "ppe_overlap", "brightness"]
DATA, MODEL, METRICS = "data/svm_dataset.csv", "models/svm.joblib", "data/svm_metrics.json"


def make_synthetic(n=600, seed=0):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({"person_conf": r.uniform(.3, 1, n), "height_ratio": r.uniform(.05, .8, n),
                      "ppe_overlap": r.uniform(0, 1, n), "brightness": r.uniform(20, 220, n)})
    score = (d.person_conf < .6) * 1 + (d.height_ratio < .15) * 1 + (d.brightness < 70) * 1 + (d.ppe_overlap.between(.2, .5)) * 1
    d["needs_review"] = ((score + r.normal(0, .5, n)) >= 1.5).astype(int)
    return d


def train():
    os.makedirs("data", exist_ok=True); os.makedirs("models", exist_ok=True)
    synthetic = not os.path.exists(DATA)
    df = make_synthetic() if synthetic else pd.read_csv(DATA)
    missing = set(FEATURES + ["needs_review"]) - set(df.columns)
    if missing or df.isna().any().any() or df.needs_review.nunique() < 2:
        raise ValueError(f"Invalid dataset (missing={missing}, NaNs or single class)")
    Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], df.needs_review, test_size=.25, stratify=df.needs_review, random_state=42)
    # Scaler lives INSIDE the pipeline -> fitted on training data only (no leakage)
    svm = Pipeline([("scale", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True, random_state=42))]).fit(Xtr, ytr)
    base = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    rep = lambda m: classification_report(yte, m.predict(Xte), output_dict=True, zero_division=0)
    out = {"dataset": "SYNTHETIC (pipeline test only)" if synthetic else DATA, "n_train": len(Xtr), "n_test": len(Xte),
           "svm": rep(svm), "baseline": rep(base), "confusion_matrix": confusion_matrix(yte, svm.predict(Xte)).tolist(),
           "limitations": "Synthetic labels follow hand-written rules; high scores only prove the pipeline works."}
    joblib.dump(svm, MODEL); json.dump(out, open(METRICS, "w"), indent=1)
    return out


def metrics():
    return json.load(open(METRICS)) if os.path.exists(METRICS) else None


def predict(feats: dict):
    if not os.path.exists(MODEL): raise RuntimeError("SVM not trained yet")
    m = joblib.load(MODEL); X = pd.DataFrame([feats])[FEATURES]
    return {"needs_review": int(m.predict(X)[0]), "probability": round(float(m.predict_proba(X)[0][1]), 3)}


if __name__ == "__main__":
    r = train(); print("SVM F1:", round(r["svm"]["weighted avg"]["f1-score"], 3), "| baseline F1:", round(r["baseline"]["weighted avg"]["f1-score"], 3))

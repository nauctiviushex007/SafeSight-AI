"""RAG: ingest PDF/TXT -> chunk -> TF-IDF vectors -> cosine search -> cited answer.
TF-IDF keeps setup offline and deterministic. To use neural embeddings, replace `search()` with
sentence-transformers + FAISS (same chunk format)."""
import os, io, json, re
import httpx
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

STORE = "data/rag_chunks.json"; SIZE, OVERLAP, MIN_SCORE = 120, 30, 0.08  # words


def _load(): return json.load(open(STORE)) if os.path.exists(STORE) else []


def documents():
    c = _load(); names = sorted({x["doc"] for x in c})
    return [{"name": n, "chunks": sum(1 for x in c if x["doc"] == n)} for n in names]


def ingest(name: str, data: bytes):
    if name.lower().endswith(".pdf"):
        pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(io.BytesIO(data)).pages)]
    elif name.lower().endswith(".txt"):
        pages = [(None, data.decode("utf-8", "ignore"))]
    else:
        raise ValueError("Only PDF and TXT files are supported")
    new = []
    for pg, text in pages:
        w = text.split()
        for s in range(0, max(len(w), 1), SIZE - OVERLAP):
            piece = " ".join(w[s:s + SIZE])
            if len(piece) > 20: new.append({"doc": name, "page": pg, "text": piece})
    if not new: raise ValueError("No extractable text found (scanned PDF?)")
    chunks = [c for c in _load() if c["doc"] != name] + new
    os.makedirs("data", exist_ok=True); json.dump(chunks, open(STORE, "w"))
    return len(new)


def search(q, k=4):
    ch = _load()
    if not ch: return []
    v = TfidfVectorizer(stop_words="english").fit([c["text"] for c in ch] + [q])
    sc = linear_kernel(v.transform([q]), v.transform([c["text"] for c in ch]))[0]
    top = sorted(range(len(ch)), key=lambda i: -sc[i])[:k]
    return [{**ch[i], "score": round(float(sc[i]), 3)} for i in top if sc[i] >= MIN_SCORE]


def answer(q):
    if not _load():
        return {"answer": "No documents are indexed yet. Upload a PDF or TXT safety document first.", "sources": []}
    hits = search(q)
    if not hits:
        return {"answer": "The indexed documents do not contain enough information to answer this question.", "sources": []}
    src = [{"doc": h["doc"], "page": h["page"], "snippet": h["text"][:300], "score": h["score"]} for h in hits]
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        try:
            ctx = "\n\n".join(f'[{i+1}] ({h["doc"]}, p.{h["page"]}) {h["text"]}' for i, h in enumerate(hits))
            r = httpx.post("https://api.anthropic.com/v1/messages", timeout=40,
                           headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                           json={"model": os.getenv("LLM_MODEL", "claude-sonnet-4-6"), "max_tokens": 600,
                                 "system": "Answer ONLY from the numbered sources, cite as [n]. If they are insufficient, say so. Never claim a document is a legal requirement.",
                                 "messages": [{"role": "user", "content": f"Sources:\n{ctx}\n\nQuestion: {q}"}]})
            r.raise_for_status()
            return {"answer": r.json()["content"][0]["text"], "sources": src, "mode": "llm"}
        except Exception as e:
            pass  # fall through to extractive answer
    sents = re.split(r"(?<=[.!?])\s+", hits[0]["text"])
    return {"answer": "Most relevant passages (extractive mode, no LLM configured):\n\n" + " ".join(sents[:3]), "sources": src, "mode": "extractive"}

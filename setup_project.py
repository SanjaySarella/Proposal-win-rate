import os, base64

ROOT = os.path.dirname(os.path.abspath(__file__))
print(f"Setting up project at: {ROOT}")

def write(path, content):
    full = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  wrote: {path}")

def b(s):
    return base64.b64decode(s).decode("utf-8")

# ── frontend/package.json ──────────────────────────────────────────────────────
write("frontend/package.json", b("ewogICJuYW1lIjogImZlZGVyYWwtY29udHJhY3QtYXBwIiwKICAidmVyc2lvbiI6ICIxLjAuMCIsCiAgInByaXZhdGUiOiB0cnVlLAogICJkZXBlbmRlbmNpZXMiOiB7CiAgICAiYXhpb3MiOiAiXjEuNi4wIiwKICAgICJyZWFjdCI6ICJeMTguMi4wIiwKICAgICJyZWFjdC1kb20iOiAiXjE4LjIuMCIsCiAgICAicmVhY3Qtc2NyaXB0cyI6ICI1LjAuMSIKICB9LAogICJzY3JpcHRzIjogewogICAgInN0YXJ0IjogInJlYWN0LXNjcmlwdHMgc3RhcnQiLAogICAgImJ1aWxkIjogInJlYWN0LXNjcmlwdHMgYnVpbGQiLAogICAgInRlc3QiOiAicmVhY3Qtc2NyaXB0cyB0ZXN0IgogIH0sCiAgImJyb3dzZXJzbGlzdCI6IHsKICAgICJwcm9kdWN0aW9uIjogWyI+MC4yJSIsICJub3QgZGVhZCIsICJub3Qgb3BfbWluaSBhbGwiXSwKICAgICJkZXZlbG9wbWVudCI6IFsibGFzdCAxIGNocm9tZSB2ZXJzaW9uIl0KICB9Cn0K"))

# ── frontend/public/index.html ─────────────────────────────────────────────────
write("frontend/public/index.html", b("PCFET0NUWVBFIGh0bWw+CjxodG1sIGxhbmc9ImVuIj4KICA8aGVhZD4KICAgIDxtZXRhIGNoYXJzZXQ9InV0Zi04IiAvPgogICAgPG1ldGEgbmFtZT0idmlld3BvcnQiIGNvbnRlbnQ9IndpZHRoPWRldmljZS13aWR0aCwgaW5pdGlhbC1zY2FsZT0xIiAvPgogICAgPHRpdGxlPlNhbmpheVNhcmVsbGEncyBGZWRlcmFsIENvbnRyYWN0IFdpbiBSYXRlIEludGVsbGlnZW5jZTwvdGl0bGU+CiAgPC9oZWFkPgogIDxib2R5PgogICAgPG5vc2NyaXB0PllvdSBuZWVkIHRvIGVuYWJsZSBKYXZhU2NyaXB0IHRvIHJ1biB0aGlzIGFwcC48L25vc2NyaXB0PgogICAgPGRpdiBpZD0icm9vdCI+PC9kaXY+CiAgPC9ib2R5Pgo8L2h0bWw+Cg=="))

# ── frontend/src/index.js ──────────────────────────────────────────────────────
write("frontend/src/index.js", b("aW1wb3J0IFJlYWN0IGZyb20gJ3JlYWN0JzsKaW1wb3J0IFJlYWN0RE9NIGZyb20gJ3JlYWN0LWRvbS9jbGllbnQnOwppbXBvcnQgQXBwIGZyb20gJy4vQXBwJzsKCmNvbnN0IHJvb3QgPSBSZWFjdERPTS5jcmVhdGVSb290KGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdyb290JykpOwpyb290LnJlbmRlcig8UmVhY3QuU3RyaWN0TW9kZT48QXBwIC8+PC9SZWFjdC5TdHJpY3RNb2RlPik7Cg=="))

# ── frontend/src/api.js ────────────────────────────────────────────────────────
write("frontend/src/api.js", b("aW1wb3J0IGF4aW9zIGZyb20gJ2F4aW9zJzsKCmNvbnN0IEFQSV9CQVNFID0gJyc7CgpleHBvcnQgY29uc3QgcHJlZGljdFdpblJhdGUgPSBhc3luYyAoY29udHJhY3REYXRhKSA9PiB7CiAgY29uc3QgcmVzcG9uc2UgPSBhd2FpdCBheGlvcy5wb3N0KGAke0FQSV9CQVNFfS9wcmVkaWN0YCwgY29udHJhY3REYXRhKTsKICByZXR1cm4gcmVzcG9uc2UuZGF0YTsKfTsKCmV4cG9ydCBjb25zdCBnZXRNb2RlbEluZm8gPSBhc3luYyAoKSA9PiB7CiAgY29uc3QgcmVzcG9uc2UgPSBhd2FpdCBheGlvcy5nZXQoYCR7QVBJX0JBU0V9L21vZGVsLWluZm9gKTsKICByZXR1cm4gcmVzcG9uc2UuZGF0YTsKfTsK"))

# ── frontend/src/App.jsx ───────────────────────────────────────────────────────
app_jsx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "App.jsx")
if os.path.exists(app_jsx_path):
    with open(app_jsx_path, "r", encoding="utf-8") as f:
        write("frontend/src/App.jsx", f.read())
else:
    print("  WARNING: App.jsx not found — skipping")

# ── backend/__init__.py ────────────────────────────────────────────────────────
write("backend/__init__.py", "")

# ── backend/requirements.txt ───────────────────────────────────────────────────
write("backend/requirements.txt",
"""fastapi==0.111.0
uvicorn==0.29.0
pydantic==2.7.1
scikit-learn==1.4.2
pandas==2.2.2
numpy==1.26.4
shap==0.45.1
langgraph==0.1.19
langchain-core==0.2.5
groq==0.9.0
chromadb==0.5.3
sentence-transformers==2.7.0
""")

# ── backend/main.py ────────────────────────────────────────────────────────────
write("backend/main.py",
"""from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Federal Contract Win Rate Intelligence")

from backend.model import ModelManager
from backend.agent import run_pipeline

model_manager = ModelManager()

@app.on_event("startup")
async def startup():
    logger.info("Loading model on startup...")
    model_manager.load()
    logger.info(f"Model loaded. ROC-AUC: {model_manager.roc_auc:.4f}")
    logger.info("Model ready.")

class ContractInput(BaseModel):
    agency:     str
    amount:     int
    duration:   int
    experience: int
    isdc:       bool
    isq4:       bool
    isdod:      bool

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    return {
        "model":             "Random Forest Classifier",
        "roc_auc":           model_manager.roc_auc,
        "contracts_trained": 969,
        "pipeline":          "LangGraph 4-node",
    }

@app.post("/predict")
async def predict(data: ContractInput):
    try:
        return await run_pipeline(data, model_manager)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

static_dir = os.path.join(os.path.dirname(__file__), "..", "static")

if os.path.exists(static_dir):
    app.mount(
        "/static",
        StaticFiles(directory=os.path.join(static_dir, "static")),
        name="static_assets",
    )

    @app.get("/")
    async def root():
        return FileResponse(os.path.join(static_dir, "index.html"))

    @app.get("/{full_path:path}")
    async def catch_all(full_path: str):
        return FileResponse(os.path.join(static_dir, "index.html"))
""")

# ── backend/model.py ───────────────────────────────────────────────────────────
write("backend/model.py",
"""import pickle, math, os, datetime
import pandas as pd

FEATURE_NAMES = [
    "log_award_amount", "is_dod", "contract_duration_days",
    "recipient_experience", "agency_activity",
    "is_dc", "award_year", "is_q4", "amount_bucket_encoded",
]

DISPLAY_NAMES = {
    "log_award_amount":       "Award Amount",
    "is_dod":                 "DoD Agency",
    "contract_duration_days": "Contract Duration",
    "recipient_experience":   "Vendor Experience",
    "agency_activity":        "Agency Activity",
    "is_dc":                  "DC Location",
    "award_year":             "Award Year",
    "is_q4":                  "Q4 Submission",
    "amount_bucket_encoded":  "Amount Bucket",
}

class ModelManager:
    def __init__(self):
        self.model     = None
        self.explainer = None
        self.roc_auc   = 0.74

    def load(self):
        base = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(base, "models", "contract_model.pkl"), "rb") as f:
            self.model = pickle.load(f)
        with open(os.path.join(base, "models", "shap_explainer.pkl"), "rb") as f:
            self.explainer = pickle.load(f)

    def predict(self, inp):
        features = self._features(inp)
        df = pd.DataFrame([features], columns=FEATURE_NAMES)
        proba = self.model.predict_proba(df)[0][1]
        pct   = int(round(proba * 100))

        sv = self.explainer.shap_values(df)
        sv = sv[1][0] if isinstance(sv, list) else sv[0]

        shap_list = sorted(
            [{"name": DISPLAY_NAMES.get(n, n), "value": round(float(v), 3)}
             for n, v in zip(FEATURE_NAMES, sv)],
            key=lambda x: abs(x["value"]), reverse=True
        )[:6]

        return pct, shap_list

    def _features(self, d):
        amt    = int(d.amount)
        bucket = (0 if amt < 500_000 else
                  1 if amt < 2_000_000 else
                  2 if amt < 5_000_000 else 3)
        return [
            math.log1p(amt),
            1 if d.isdod else 0,
            int(d.duration),
            int(d.experience),
            7.5,
            1 if d.isdc else 0,
            datetime.datetime.now().year,
            1 if d.isq4 else 0,
            bucket,
        ]
""")

# ── backend/rag.py ─────────────────────────────────────────────────────────────
write("backend/rag.py",
"""import os, logging
logger = logging.getLogger(__name__)
_collection = None

def get_collection():
    global _collection
    if _collection is not None:
        return _collection
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        base       = os.path.dirname(os.path.dirname(__file__))
        chroma_dir = os.path.join(base, "data", "chroma_db")
        os.makedirs(chroma_dir, exist_ok=True)
        ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=chroma_dir)
        _collection = client.get_or_create_collection("federal_contracts", embedding_function=ef)
        if _collection.count() == 0:
            _seed(base, _collection)
        logger.info(f"ChromaDB ready: {_collection.count()} contracts indexed")
    except Exception as e:
        logger.warning(f"ChromaDB init failed: {e}")
        _collection = None
    return _collection

def _seed(base, col):
    import pandas as pd
    csv = os.path.join(base, "data", "processed", "contracts_features.csv")
    if not os.path.exists(csv):
        return
    df = pd.read_csv(csv).head(500)
    docs, ids, metas = [], [], []
    for i, row in df.iterrows():
        text = (f"Agency: {row.get('agency_name','unknown')} | "
                f"Amount: {row.get('award_amount',0):.0f} | "
                f"Duration: {row.get('contract_duration_days',0)} days | "
                f"Win: {row.get('is_won',0)}")
        docs.append(text); ids.append(f"c_{i}"); metas.append({"i": int(i)})
    col.add(documents=docs, ids=ids, metadatas=metas)
    logger.info(f"Seeded {len(docs)} contracts into ChromaDB")

def query_rag(contract_input, pct):
    col = get_collection()
    if col is None or col.count() == 0:
        return ""
    try:
        q = f"{contract_input.agency} amount {contract_input.amount} duration {contract_input.duration} win {pct}%"
        r = col.query(query_texts=[q], n_results=3)
        return " | ".join(r.get("documents", [[]])[0][:3])
    except Exception as e:
        logger.warning(f"RAG query failed: {e}")
        return ""
""")

# ── backend/agent.py ───────────────────────────────────────────────────────────
write("backend/agent.py",
"""import os, json, logging
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from groq import Groq

logger      = logging.getLogger(__name__)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

class State(TypedDict):
    inp:     Any
    pct:     int
    shap:    List[Dict]
    context: str
    points:  List[str]

def node_predict(s):
    return s

def node_retrieve(s):
    try:
        from backend.rag import query_rag
        s["context"] = query_rag(s["inp"], s["pct"])
    except Exception as e:
        logger.warning(f"RAG failed: {e}")
        s["context"] = ""
    return s

def node_analyze(s):
    return s

def node_strategy(s):
    inp, pct, shap = s["inp"], s["pct"], s["shap"]
    context = s.get("context", "")
    factors = "\\n".join([
        f"- {f['name']}: {'increases' if f['value'] > 0 else 'decreases'} win probability (impact: {abs(f['value']):.2f})"
        for f in shap[:4]
    ])
    prompt = (
        "You are a federal procurement strategy advisor.\\n\\n"
        f"Contract: {inp.agency}, ${inp.amount:,}, {inp.duration} days\\n"
        f"Vendor experience: {inp.experience} prior contracts\\n"
        f"Win probability: {pct}%\\n"
        f"Q4 submission: {inp.isq4} | DC location: {inp.isdc}\\n\\n"
        f"Key factors:\\n{factors}\\n\\n"
        f"Historical context: {context[:400] if context else 'N/A'}\\n\\n"
        "Return ONLY a JSON array of exactly 4 strings. "
        "Each is one strategic recommendation, 1-2 sentences, direct, actionable. "
        "No markdown, no numbering. "
        'Example: ["point 1", "point 2", "point 3", "point 4"]'
    )
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        text   = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        points = json.loads(text)
        if isinstance(points, list) and len(points) >= 2:
            s["points"] = points[:4]
            return s
    except Exception as e:
        logger.error(f"Groq failed: {e}")

    outlook = "favorable" if pct >= 60 else "moderate" if pct >= 40 else "challenging"
    s["points"] = [
        f"Win probability of {pct}% indicates a {outlook} position. {'Proceed with a full proposal effort.' if pct >= 50 else 'Evaluate bid/no-bid carefully before committing resources.'}",
        f"Vendor experience is the single strongest predictor. Your {inp.experience} prior contracts {'is a meaningful advantage.' if inp.experience >= 10 else 'is below the threshold where win rates improve significantly — teaming with an experienced prime is worth considering.'}",
        f"{'Q4 submissions historically reduce win rates by ~15% due to budget exhaustion. Shift to Q1-Q2 if the timeline allows.' if inp.isq4 else 'Non-Q4 timing is a genuine advantage — agencies have fuller budgets and fewer competing bids at this point in the fiscal year.'}",
        f"DoD contracts in the $1M-$5M range show the highest win probability for vendors with proven delivery records. {'Your amount falls in this range.' if 1_000_000 <= int(inp.amount) <= 5_000_000 else 'Adjust scope to target this range if the proposal allows.'}",
    ]
    return s

def _build():
    g = StateGraph(State)
    for name, fn in [("predict", node_predict), ("retrieve", node_retrieve),
                     ("analyze", node_analyze), ("strategy", node_strategy)]:
        g.add_node(name, fn)
    g.set_entry_point("predict")
    g.add_edge("predict",  "retrieve")
    g.add_edge("retrieve", "analyze")
    g.add_edge("analyze",  "strategy")
    g.add_edge("strategy", END)
    return g.compile()

pipeline = _build()

async def run_pipeline(contract_input, model_manager):
    pct, shap = model_manager.predict(contract_input)
    result = pipeline.invoke({"inp": contract_input, "pct": pct, "shap": shap, "context": "", "points": []})
    return {"pct": result["pct"], "features": result["shap"], "points": result["points"]}
""")

# ── Dockerfile ─────────────────────────────────────────────────────────────────
write("Dockerfile",
"""FROM node:18-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package.json ./
RUN npm install --legacy-peer-deps
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/
COPY models/ ./models/
COPY data/ ./data/
COPY --from=frontend-build /frontend/build ./static
ENV PYTHONPATH=/app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
""")

# ── .dockerignore ──────────────────────────────────────────────────────────────
write(".dockerignore",
""".venv
__pycache__
*.pyc
.env
.git
frontend/node_modules
frontend/build
src/
app.py
plots/
*.md
setup_project.py
App.jsx
""")

# ── .gitignore ─────────────────────────────────────────────────────────────────
write(".gitignore",
""".venv/
__pycache__/
*.pyc
.env
frontend/node_modules/
frontend/build/
data/raw/
data/chroma_db/
build_log*.txt
App.jsx
setup_project.py
""")

print("")
print("=" * 55)
print("All files written successfully.")
print("=" * 55)
print("")
print("Run these commands next:")
print("  cd frontend")
print("  npm install --legacy-peer-deps")
print("  cd ..")
print("  git add .")
print('  git commit -m "feat: React + FastAPI rebuild with LangGraph"')
print("  git push origin main")

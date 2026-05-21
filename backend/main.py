from fastapi import FastAPI, HTTPException
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

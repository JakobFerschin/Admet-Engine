"""
ADMET-Engine — FastAPI REST Server
Phase 4, Month 7–9: Backend & Workflow Infrastructure

Provides a production-ready REST API for batch ADMET prediction.
On startup the DMPNN model is loaded from models/dmpnn_v1.pt and kept
in memory for low-latency inference.

Endpoints:
  POST /predict/admet  — batch prediction for up to 1000 SMILES

Architecture (Infrastructure Layer, Section 4.1):
  FastAPI + OpenAPI/Swagger → PostgreSQL + pgvector (future)
"""

from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Ensure src/ is importable whether running from project root or within src/api
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from torch_geometric.data import Batch

from models.dmpnn import ADMETPredictor
from data_pipeline import standardize_molecule
from featurizer import smiles_batch_to_data_list

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH: Path = _PROJECT_ROOT / "models" / "dmpnn_v1.pt"
MAX_BATCH_SIZE: int = 1000
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ADMET endpoint labels (12 tasks, Section 2.2)
TASK_LABELS: list[str] = [
    "hepatotoxicity",
    "hERG_inhibition",
    "AMES_mutagenicity",
    "CYP3A4_inhibition",
    "nephrotoxicity",
    "CYP2D6_inhibition",
    "CYP2C9_inhibition",
    "blood_brain_barrier",
    "plasma_protein_binding",
    "caco2_permeability",
    "logD_logP",
    "P_gp_substrate_inhibitor",
]

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class BatchPredictRequest(BaseModel):
    """Request body for batch ADMET prediction."""
    smiles: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description="List of SMILES strings to predict (1–1000).",
        examples=[["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]],
    )


class MoleculePrediction(BaseModel):
    """Prediction result for a single molecule."""
    smiles: str = Field(..., description="Input SMILES string.")
    status: str = Field(..., description="'success' or 'error'.")
    predictions: Optional[dict[str, float]] = Field(
        None,
        description="Map from endpoint name → predicted value (only when status=success).",
    )
    error: Optional[str] = Field(
        None,
        description="Error message (only when status=error).",
    )


class BatchPredictResponse(BaseModel):
    """Response wrapper for batch ADMET prediction."""
    results: list[MoleculePrediction] = Field(
        ..., description="Per-molecule prediction results."
    )
    model_version: str = Field("dmpnn_v1", description="Model checkpoint version.")
    prediction_time_ms: float = Field(
        ..., description="Total inference wall time in milliseconds."
    )


# ---------------------------------------------------------------------------
# Global model reference (set at startup)
# ---------------------------------------------------------------------------

_model: Optional[ADMETPredictor] = None


def load_model(path: Path) -> ADMETPredictor:
    """Instantiate the ADMETPredictor and load weights from disk."""
    model = ADMETPredictor(
        node_dim=9, edge_dim=4, hidden_dim=300,
        num_tasks=12, T=3, set2set_steps=3, dropout=0.1,
    )
    state = torch.load(str(path), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Lifespan — model loading
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; clean up on shutdown."""
    global _model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_PATH}. "
            f"Run 'python scripts/save_dummy_model.py' first."
        )
    _model = load_model(MODEL_PATH)
    print(f"[startup] ADMETPredictor loaded on {DEVICE} ({MODEL_PATH.name})")
    yield
    # Shutdown: nothing to clean for now
    _model = None
    print("[shutdown] Model unloaded.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ADMET-Engine API",
    description="Batch ADMET prediction for drug-like molecules via DMPNN.",
    version="0.1.0",
    lifespan=lifespan,
)

# --- CORS MIDDLEWARE HINZUGEFÜGT ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Readiness probe."""
    return {"status": "ok", "model_loaded": _model is not None, "device": DEVICE}


@app.post("/predict/admet", response_model=BatchPredictResponse)
async def predict_admet(request: BatchPredictRequest):
    """
    Predict 12 ADMET endpoints for a batch of up to 1000 SMILES.

    Each SMILES is standardised via the RDKit pipeline, featurised into a
    molecular graph, and passed through the DMPNN model.  Invalid SMILES
    are reported per-molecule without crashing the entire batch.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again in a moment.")

    t0 = time.perf_counter()

    # 1. Standardise + featurise each SMILES
    valid_data, errors = smiles_batch_to_data_list(request.smiles)

    # 2. Build results array with placeholders
    n_total = len(request.smiles)
    results: list[MoleculePrediction] = [None] * n_total  # type: ignore[assignment]

    # Fill error slots first
    for err in errors:
        results[err["index"]] = MoleculePrediction(
            smiles=err["smiles"],
            status="error",
            error=err["error"],
        )

    # 3. Batch inference on valid molecules
    if valid_data:
        batch = Batch.from_data_list(valid_data)
        batch = batch.to(DEVICE)

        with torch.no_grad():
            preds_tensor = _model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
        # preds_tensor: [num_valid_molecules, 12]

        for i, data in enumerate(valid_data):
            pred_vec = preds_tensor[i].cpu().tolist()
            pred_dict = {
                TASK_LABELS[j]: round(pred_vec[j], 6)
                for j in range(len(TASK_LABELS))
            }
            results[data.orig_index] = MoleculePrediction(
                smiles=data.smiles,
                status="success",
                predictions=pred_dict,
            )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return BatchPredictResponse(
        results=results,
        model_version="dmpnn_v1",
        prediction_time_ms=round(elapsed_ms, 2),
    )
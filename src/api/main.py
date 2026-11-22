"""
ADMET-Engine — FastAPI Scientific Inference Server
Serves the scaffold-validated ChEMBL v37 Multi-Modal Ensemble
(Chemprop One-Hot DMPNN + Residual ECFP4/PhysChem MLP + ExtraTrees + Linear QSAR)
alongside deterministic RDKit medicinal chemistry profiling (2D SVG rendering,
BOILED-Egg HIA/BBB permeation, PAINS/Brenk toxicophore screening, QED, Fsp3,
Lipinski/Veber/Pfizer 3-75/GSK 4-400 rules, and Tanimoto Applicability Domain).

Endpoints:
  GET  /health          — readiness & loaded model info
  GET  /model/metadata  — Bemis-Murcko scaffold validation metrics & ablation table
  POST /predict/admet   — batch ADMET prediction + full cheminformatics profile (1–1000 SMILES)
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

# Ensure src/ is importable whether running from project root or within src/api
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from torch_geometric.data import Batch

from cheminformatics import analyze_molecule
from data_pipeline import standardize_molecule
from featurizer import mol_to_data
from models.chembl_ensemble import ChEMBLADMETEnsemble
from models.dmpnn import ADMETPredictor

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

MODELS_DIR: Path = _PROJECT_ROOT / "models"
MODEL_PATH: Path = MODELS_DIR / "dmpnn_v1.pt"  # Legacy fallback path
MAX_BATCH_SIZE: int = 1000
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

LEGACY_TASK_LABELS: list[str] = [
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
# Pydantic Schemas
# ---------------------------------------------------------------------------

class BatchPredictRequest(BaseModel):
    """Request body for batch ADMET & medicinal chemistry analysis."""
    smiles: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description="List of SMILES strings to analyze (1–1000).",
        examples=[["CC(=O)Oc1ccccc1C(=O)O", "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1"]],
    )
    include_svg: bool = Field(
        True,
        description="Whether to include 2D RDKit SVG structure drawings in the response.",
    )


class MoleculePrediction(BaseModel):
    """Comprehensive prediction & medicinal chemistry profile for one molecule."""
    smiles: str = Field(..., description="Input SMILES string.")
    status: str = Field(..., description="'success' or 'error'.")
    predictions: Optional[dict[str, float]] = Field(
        None,
        description="Calibrated probabilities in [0, 1] per ML endpoint.",
    )
    ml_endpoints: Optional[dict[str, Any]] = Field(
        None,
        description="Detailed ML outputs: probability, epistemic std, 95% CI, risk tier, and per-model breakdown.",
    )
    applicability_domain: Optional[dict[str, Any]] = Field(
        None,
        description="Tanimoto Applicability Domain against ChEMBL v37 training corpus and nearest neighbor.",
    )
    cheminformatics: Optional[dict[str, Any]] = Field(
        None,
        description="Deterministic RDKit identifiers, physchem descriptors, rules, PAINS/Brenk alerts, BOILED-Egg, and 2D SVG.",
    )
    error: Optional[str] = Field(
        None,
        description="Error message (only when status=error).",
    )


class BatchPredictResponse(BaseModel):
    """Response wrapper for batch ADMET prediction."""
    results: list[MoleculePrediction] = Field(
        ..., description="Per-molecule prediction and cheminformatics results."
    )
    model_version: str = Field(..., description="Active model artifact identifier.")
    prediction_time_ms: float = Field(
        ..., description="Total wall-clock inference & cheminformatics time in milliseconds."
    )


# ---------------------------------------------------------------------------
# Global model state (initialized in lifespan)
# ---------------------------------------------------------------------------

_ensemble: Optional[ChEMBLADMETEnsemble] = None
_legacy_model: Optional[ADMETPredictor] = None
_active_version: str = "unloaded"


def _load_legacy_model(path: Path) -> ADMETPredictor:
    """Instantiate legacy 12-task ADMETPredictor and load weights."""
    model = ADMETPredictor(
        node_dim=9, edge_dim=4, hidden_dim=300,
        num_tasks=12, T=3, set2set_steps=3, dropout=0.1,
    )
    state = torch.load(str(path), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def _ensemble_bundle_exists(models_dir: Path) -> bool:
    required = (
        "chembl_v2_metrics.json",
        "chembl_v2_bundle.joblib",
        "dmpnn_chembl_v2.pt",
        "fp_mlp_chembl_v2.pt",
        "chembl_v2_ad.npz",
    )
    return all((models_dir / f).exists() for f in required)


# ---------------------------------------------------------------------------
# Lifespan — startup & shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ChEMBL v37 multi-modal ensemble on startup (fallback to legacy if absent)."""
    global _ensemble, _legacy_model, _active_version

    force_legacy = os.environ.get("ADMET_USE_LEGACY", "0") == "1"
    if not force_legacy and _ensemble_bundle_exists(MODELS_DIR):
        _ensemble = ChEMBLADMETEnsemble(MODELS_DIR, device=DEVICE)
        _legacy_model = None
        _active_version = _ensemble.metadata.get("model_version", "chembl_v37_ensemble_v2")
        print(f"[startup] ChEMBL v37 Multi-Modal Ensemble loaded on {DEVICE} ({_active_version})")
    elif MODEL_PATH.exists():
        _ensemble = None
        _legacy_model = _load_legacy_model(MODEL_PATH)
        _active_version = "dmpnn_v1"
        print(f"[startup] Legacy ADMETPredictor loaded on {DEVICE} ({MODEL_PATH.name})")
    else:
        raise FileNotFoundError(
            f"No trained model bundle found in {MODELS_DIR}. "
            f"Run 'python src/training/train_chembl_ensemble.py' first."
        )

    yield

    _ensemble = None
    _legacy_model = None
    _active_version = "unloaded"
    print("[shutdown] Models unloaded.")


# ---------------------------------------------------------------------------
# FastAPI application & spec-compliant CORS
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ADMET-Engine Scientific API",
    description=(
        "Scaffold-validated ChEMBL v37 Multi-Modal ADMET Ensemble (DMPNN + ECFP4/PhysChem) "
        "with Applicability Domain and deterministic RDKit medicinal chemistry profiling."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Spec-compliant CORS: wildcard origins must not combine with allow_credentials=True
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Readiness probe and active model metadata summary."""
    loaded = (_ensemble is not None) or (_legacy_model is not None)
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "model_version": _active_version,
        "mode": "chembl_v37_ensemble" if _ensemble is not None else "legacy_dmpnn",
        "device": DEVICE,
    }


@app.get("/model/metadata")
async def get_model_metadata():
    """Return Bemis-Murcko scaffold validation metrics, ablation table, and provenance."""
    if _ensemble is not None:
        return _ensemble.metadata
    if _legacy_model is not None:
        return {
            "model_version": _active_version,
            "mode": "legacy_dmpnn",
            "note": "Legacy 12-task placeholder model active; train ChEMBL v37 ensemble for validated predictions.",
        }
    raise HTTPException(status_code=503, detail="Model not loaded yet.")


@app.post("/predict/admet", response_model=BatchPredictResponse)
async def predict_admet(request: BatchPredictRequest):
    """
    Predict calibrated ADMET toxicity endpoints and compute full medicinal
    chemistry profiles for a batch of 1–1000 SMILES strings.

    Each SMILES is standardized via the RDKit MolStandardize pipeline; invalid
    SMILES are isolated per-molecule without failing the rest of the batch.
    """
    if _ensemble is None and _legacy_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again in a moment.")

    t0 = time.perf_counter()
    n_total = len(request.smiles)
    results: list[Optional[MoleculePrediction]] = [None] * n_total

    valid_indices: list[int] = []
    valid_smiles: list[str] = []
    valid_mols: list = []

    # 1. Standardize every SMILES individually so bad entries don't crash the batch
    for idx, smi in enumerate(request.smiles):
        try:
            mol = standardize_molecule(smi)
            if mol is None or mol.GetNumAtoms() == 0:
                raise ValueError("Empty molecular structure after standardization.")
            valid_indices.append(idx)
            valid_smiles.append(smi)
            valid_mols.append(mol)
        except (ValueError, RuntimeError) as exc:
            results[idx] = MoleculePrediction(
                smiles=smi,
                status="error",
                error=f"Invalid or unparsable SMILES: '{smi}' ({exc})",
            )

    # 2. Batch ML inference + deterministic RDKit medicinal chemistry profiling
    if valid_mols:
        if _ensemble is not None:
            ml_outputs = _ensemble.predict_mols(valid_mols)
            for i, (orig_idx, smi, mol) in enumerate(zip(valid_indices, valid_smiles, valid_mols)):
                chem_profile = analyze_molecule(mol, include_svg=request.include_svg)
                ml = ml_outputs[i]
                results[orig_idx] = MoleculePrediction(
                    smiles=smi,
                    status="success",
                    predictions=ml["predictions"],
                    ml_endpoints=ml["ml_endpoints"],
                    applicability_domain=ml["applicability_domain"],
                    cheminformatics=chem_profile,
                )
        else:
            # Legacy 12-task DMPNN fallback — apply sigmoid so outputs are calibrated probabilities in [0, 1]
            valid_graphs = [mol_to_data(m, rich=False) for m in valid_mols]
            batch = Batch.from_data_list(valid_graphs).to(DEVICE)
            with torch.no_grad():
                logits = _legacy_model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch
                )
                probs = torch.sigmoid(logits).cpu().tolist()

            for i, (orig_idx, smi, mol) in enumerate(zip(valid_indices, valid_smiles, valid_mols)):
                pred_dict = {
                    LEGACY_TASK_LABELS[j]: round(probs[i][j], 6)
                    for j in range(len(LEGACY_TASK_LABELS))
                }
                chem_profile = analyze_molecule(mol, include_svg=request.include_svg)
                results[orig_idx] = MoleculePrediction(
                    smiles=smi,
                    status="success",
                    predictions=pred_dict,
                    cheminformatics=chem_profile,
                )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return BatchPredictResponse(
        results=[r for r in results if r is not None],
        model_version=_active_version,
        prediction_time_ms=round(elapsed_ms, 2),
    )
"""
ADMET-Engine — FastAPI Scientific Inference Server
Serves the scaffold-validated ChEMBL v37 Multi-Modal Ensemble
(Chemprop One-Hot DMPNN + Residual ECFP4/PhysChem MLP + ExtraTrees + Linear QSAR)
alongside deterministic RDKit medicinal chemistry profiling (2D SVG rendering,
BOILED-Egg HIA/BBB permeation, PAINS/Brenk toxicophore screening, QED, Fsp3,
Lipinski/Veber/Pfizer 3-75/GSK 4-400 rules, and Tanimoto Applicability Domain).

Endpoints:
  GET  /health               — readiness & loaded model info
  GET  /model/metadata       — Bemis-Murcko scaffold validation metrics & ablation table
  POST /predict/admet        — batch ADMET prediction + full cheminformatics profile (1–1000 SMILES)
  POST /api/v3/simulate-cipa — CardioPhys-3D ORd-CiPA cardiomyocyte action potential simulation
  POST /api/v3/repair        — CardioPhys-3D closed-loop MMP molecular repair agent
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

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from rdkit import Chem
from torch_geometric.data import Batch

from cheminformatics import analyze_molecule
from data_pipeline import standardize_molecule
from featurizer import mol_to_data
from models.chembl_ensemble import ChEMBLADMETEnsemble
from models.dmpnn import ADMETPredictor
from physiology.cipa28 import get_cipa28_drug
from physiology.ohara_rudy import CardiomyocyteSimulator, ChannelBlockProfile
from repair.molecular_repair import ClosedLoopRepairAgent

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


# ===========================================================================
# CardioPhys-3D v3.0 — Electrophysiology Simulation & Closed-Loop Repair
# ===========================================================================

class SimulateCiPARequest(BaseModel):
    """
    Request body for ORd-CiPA ventricular cardiomyocyte action potential simulation.
    Provide either `drug_name` (FDA-CiPA 28 panel lookup) OR custom ion-channel
    potencies (`pic50_herg`, `pic50_cav12`, `pic50_nav15_late`) + `conc_uM`.
    """
    drug_name: Optional[str] = Field(
        None,
        description="Reference drug name in the official FDA-CiPA 28 panel (case-insensitive).",
        examples=["Verapamil", "Dofetilide", "Ranolazine"],
    )
    cmax_multiple: float = Field(
        1.0,
        gt=0.0,
        le=100.0,
        description="Multiplier applied to EFTPC (free Cmax) when using `drug_name` without explicit `conc_uM`.",
    )
    pic50_herg: Optional[float] = Field(
        None,
        ge=1.0,
        le=12.0,
        description="Custom hERG (I_Kr) potency as pIC50 (-log10 IC50 in M).",
    )
    pic50_cav12: Optional[float] = Field(
        None,
        ge=1.0,
        le=12.0,
        description="Custom Cav1.2 (I_CaL) potency as pIC50 (-log10 IC50 in M).",
    )
    pic50_nav15_late: Optional[float] = Field(
        None,
        ge=1.0,
        le=12.0,
        description="Custom Nav1.5 late (I_NaL) potency as pIC50 (-log10 IC50 in M).",
    )
    pic50_nav15_peak: Optional[float] = Field(
        None,
        ge=1.0,
        le=12.0,
        description="Optional custom Nav1.5 peak (I_Na) potency as pIC50 (-log10 IC50 in M).",
    )
    conc_uM: Optional[float] = Field(
        None,
        gt=0.0,
        le=1e5,
        description="Free drug concentration in micromolar (uM). Required when `drug_name` is not provided.",
    )
    n_beats: int = Field(
        1,
        ge=1,
        le=5,
        description="Number of 1 Hz ventricular pacing beats to integrate (default 1, max 5).",
    )

    @model_validator(mode="after")
    def _validate_mode(self) -> "SimulateCiPARequest":
        if self.drug_name is None:
            if self.conc_uM is None:
                raise ValueError("Provide either `drug_name` or `conc_uM` with custom pIC50 values.")
            if (
                self.pic50_herg is None
                and self.pic50_cav12 is None
                and self.pic50_nav15_late is None
                and self.pic50_nav15_peak is None
            ):
                raise ValueError(
                    "Provide at least one custom channel potency (`pic50_herg`, `pic50_cav12`, or `pic50_nav15_late`) when `drug_name` is omitted."
                )
        return self


class ActionPotentialWaveform(BaseModel):
    """Downsampled (~200 points) final-beat ventricular action potential trace."""
    t_ms: list[float] = Field(..., description="Time grid in milliseconds.")
    v_mV: list[float] = Field(..., description="Transmembrane potential in millivolts.")


class SimulateCiPAResponse(BaseModel):
    """ORd-CiPA cardiomyocyte simulation biomarkers and action potential waveform."""
    mode: str = Field(..., description="'cipa28_lookup' or 'custom_pic50'.")
    drug_name: Optional[str] = Field(None, description="Canonical FDA-CiPA 28 drug name (if looked up).")
    cipa_clinical_risk: Optional[str] = Field(None, description="FDA-CiPA clinical TdP risk category (if looked up).")
    conc_uM: float = Field(..., description="Simulated free concentration in uM.")
    n_beats: int = Field(..., description="Number of integrated 1 Hz pacing beats.")
    channel_conductances: dict[str, float] = Field(
        ..., description="Fractional remaining channel conductances (g_kr, g_cal, g_nal, g_na) in (0, 1]."
    )
    qnet_uC_uF: float = Field(..., description="FDA-CiPA net charge biomarker qNet (uC/uF).")
    apd90_ms: float = Field(..., description="Action potential duration at 90% repolarization (ms).")
    apd50_ms: float = Field(..., description="Action potential duration at 50% repolarization (ms).")
    tdp_risk_score: float = Field(..., description="Smooth calibrated Torsades de Pointes risk score in (0, 1).")
    has_ead: bool = Field(..., description="Whether an Early Afterdepolarization (EAD) was detected.")
    risk_tier: str = Field(..., description="Predicted TdP risk tier: 'Low', 'Intermediate', or 'High'.")
    waveform: ActionPotentialWaveform = Field(..., description="Downsampled (~200-point) action potential waveform.")
    simulation_time_ms: float = Field(..., description="Wall-clock ODE integration time in milliseconds.")


class RepairRequest(BaseModel):
    """Request body for WP5 closed-loop MMP molecular repair."""
    smiles: str = Field(
        ...,
        min_length=1,
        description="Seed SMILES string of the lead compound to detoxify.",
        examples=["CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1"],
    )
    n_rounds: int = Field(2, ge=1, le=3, description="Number of iterative MMP edit rounds (1–3).")
    beam_width: int = Field(4, ge=1, le=6, description="Beam search width per round (1–6).")
    min_tanimoto: float = Field(
        0.55, ge=0.0, le=1.0, description="Minimum ECFP4 Tanimoto similarity to preserve seed scaffold."
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of top-ranked repaired analogs to return.")


class SeedProfileResponse(BaseModel):
    """Baseline cardiotoxicity and medicinal chemistry profile of the seed compound."""
    smiles: str
    risk_score: float
    qed: float
    sa_score: float
    clogp: float
    tpsa: float


class RepairedAnalogResponse(BaseModel):
    """Ranked repaired analog generated by the closed-loop MMP agent."""
    smiles: str
    risk_score: float
    delta_risk: float
    tanimoto: float
    qed: float
    sa_score: float
    clogp: float
    tpsa: float
    pains_count: int
    edit_path: list[str]
    objective: float


class RepairResponse(BaseModel):
    """Response payload for POST /api/v3/repair."""
    seed: SeedProfileResponse
    total_evaluated: int
    best_delta_risk: float
    repaired_candidates: list[RepairedAnalogResponse]
    oracle: str
    repair_time_ms: float


def _classify_tdp_risk_tier(tdp_risk_score: float, has_ead: bool) -> str:
    """Map continuous qNet-derived TdP risk score and EAD flag to FDA-CiPA tiers."""
    if has_ead or tdp_risk_score >= 0.65:
        return "High"
    if tdp_risk_score >= 0.35:
        return "Intermediate"
    return "Low"


def _downsample_waveform(
    time_ms: torch.Tensor,
    v_trace: torch.Tensor,
    target_points: int = 200,
) -> ActionPotentialWaveform:
    """Uniformly downsample an action potential trace to ~target_points for JSON transport."""
    t_arr = time_ms.detach().cpu().numpy()
    v_arr = v_trace.detach().cpu().numpy()
    n = len(t_arr)
    if n > target_points:
        idx = np.linspace(0, n - 1, target_points, dtype=int)
        t_arr = t_arr[idx]
        v_arr = v_arr[idx]
    return ActionPotentialWaveform(
        t_ms=[round(float(x), 2) for x in t_arr],
        v_mV=[round(float(y), 3) for y in v_arr],
    )


@app.post("/api/v3/simulate-cipa", response_model=SimulateCiPAResponse)
async def simulate_cipa_v3(request: SimulateCiPARequest):
    """
    Simulate a human ventricular cardiomyocyte action potential (ORd-CiPA ODE)
    for either an official FDA-CiPA 28 reference drug or custom multi-channel
    pIC50 potencies, returning qNet, APD90/APD50, EAD detection, TdP risk tier,
    and a ~200-point action potential waveform.
    """
    t0 = time.perf_counter()

    if request.drug_name is not None:
        try:
            drug = get_cipa28_drug(request.drug_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        conc_val = float(request.conc_uM) if request.conc_uM is not None else float(drug.cmax_free_uM * request.cmax_multiple)
        conc_t = torch.tensor([conc_val], dtype=torch.float32)

        def _opt_ic50(val: Optional[float]) -> Optional[torch.Tensor]:
            return None if val is None else torch.tensor([float(val)], dtype=torch.float32)

        block = ChannelBlockProfile.from_ic50_uM(
            conc_uM=conc_t,
            ic50_herg_uM=_opt_ic50(drug.ic50_herg_uM),
            ic50_cav12_uM=_opt_ic50(drug.ic50_cav12_uM),
            ic50_nav15_late_uM=_opt_ic50(drug.ic50_nav15_late_uM),
            ic50_nav15_peak_uM=_opt_ic50(drug.ic50_nav15_peak_uM),
            h_herg=drug.h_herg,
            h_cav12=drug.h_cav12,
            h_nav15_late=drug.h_nav15_late,
            h_nav15_peak=drug.h_nav15_peak,
        )
        mode = "cipa28_lookup"
        canon_drug_name: Optional[str] = drug.name
        cipa_clinical_risk: Optional[str] = drug.tdp_risk
    else:
        conc_val = float(request.conc_uM)
        conc_t = torch.tensor([conc_val], dtype=torch.float32)

        def _opt_pic50(val: Optional[float]) -> Optional[torch.Tensor]:
            return None if val is None else torch.tensor([float(val)], dtype=torch.float32)

        block = ChannelBlockProfile.from_pic50(
            conc_uM=conc_t,
            pic50_herg=_opt_pic50(request.pic50_herg),
            pic50_cav12=_opt_pic50(request.pic50_cav12),
            pic50_nav15_late=_opt_pic50(request.pic50_nav15_late),
            pic50_nav15_peak=_opt_pic50(request.pic50_nav15_peak),
        )
        mode = "custom_pic50"
        canon_drug_name = None
        cipa_clinical_risk = None

    sim = CardiomyocyteSimulator(dt_ms=0.3, n_beats=request.n_beats)
    with torch.no_grad():
        res = sim.simulate(block, return_traces=True, trace_stride=2)

    qnet_v = float(res.qnet[0].item())
    apd90_v = float(res.apd90_ms[0].item())
    apd50_v = float(res.apd50_ms[0].item())
    tdp_v = float(res.tdp_risk_score[0].item())
    ead_v = bool(res.has_ead[0].item())
    risk_tier = _classify_tdp_risk_tier(tdp_v, ead_v)

    waveform = _downsample_waveform(res.time_ms, res.v_trace[0], target_points=200)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return SimulateCiPAResponse(
        mode=mode,
        drug_name=canon_drug_name,
        cipa_clinical_risk=cipa_clinical_risk,
        conc_uM=round(conc_val, 6),
        n_beats=request.n_beats,
        channel_conductances={
            "g_kr": round(float(block.g_kr[0].item()), 4),
            "g_cal": round(float(block.g_cal[0].item()), 4),
            "g_nal": round(float(block.g_nal[0].item()), 4),
            "g_na": round(float(block.g_na[0].item()), 4),
        },
        qnet_uC_uF=round(qnet_v, 5),
        apd90_ms=round(apd90_v, 2),
        apd50_ms=round(apd50_v, 2),
        tdp_risk_score=round(tdp_v, 4),
        has_ead=ead_v,
        risk_tier=risk_tier,
        waveform=waveform,
        simulation_time_ms=round(elapsed_ms, 2),
    )


@app.post("/api/v3/repair", response_model=RepairResponse)
async def repair_molecule_v3(request: RepairRequest):
    """
    Run the CardioPhys-3D WP5 Closed-Loop Molecular Repair Agent on a seed SMILES
    using the loaded ChEMBL v37 Multi-Modal Ensemble (`hERG_inhibition`) as the
    calibrated cardiotoxicity oracle, returning Pareto-ranked structural analogs.
    """
    if _ensemble is None:
        raise HTTPException(
            status_code=503,
            detail="ChEMBL v37 Multi-Modal Ensemble is not loaded; /api/v3/repair requires the v2/v3 ensemble.",
        )

    try:
        seed_mol = standardize_molecule(request.smiles)
        if seed_mol is None or seed_mol.GetNumAtoms() == 0:
            raise ValueError("Empty molecular structure after standardization.")
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid or unparsable seed SMILES: '{request.smiles}' ({exc})",
        ) from exc

    t0 = time.perf_counter()

    def _herg_oracle(smiles_list: list[str]) -> list[float]:
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        preds = _ensemble.predict_mols(mols)
        return [float(p["predictions"]["hERG_inhibition"]) for p in preds]

    agent = ClosedLoopRepairAgent(
        risk_oracle=_herg_oracle,
        min_tanimoto=request.min_tanimoto,
        max_sa_score=5.0,
        min_qed=0.25,
        require_zero_pains=True,
    )
    report = agent.repair(
        seed_smiles=request.smiles,
        n_rounds=request.n_rounds,
        beam_width=request.beam_width,
        top_k=request.top_k,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    seed_dict = report["seed"]
    candidates = [
        RepairedAnalogResponse(
            smiles=c.smiles,
            risk_score=c.risk_score,
            delta_risk=c.delta_risk,
            tanimoto=c.tanimoto,
            qed=c.qed,
            sa_score=c.sa_score,
            clogp=c.clogp,
            tpsa=c.tpsa,
            pains_count=c.pains_count,
            edit_path=c.edit_path,
            objective=c.objective,
        )
        for c in report["repaired_candidates"]
    ]

    return RepairResponse(
        seed=SeedProfileResponse(**seed_dict),
        total_evaluated=int(report["total_evaluated"]),
        best_delta_risk=float(report["best_delta_risk"]),
        repaired_candidates=candidates,
        oracle="ChEMBLADMETEnsemble.hERG_inhibition",
        repair_time_ms=round(elapsed_ms, 2),
    )
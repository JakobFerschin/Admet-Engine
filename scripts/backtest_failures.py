#!/usr/bin/env python
"""
ADMET-Engine — Historical Backtester
Phase 5, Month 10–12: "Killer-Slide" Backtesting

Loads the trained (or dummy) DMPNN model and runs it against
10 real-world drug candidates that failed clinical trials due
to ADMET/Tox issues (Section 10, "Woche 37–38").

For each candidate the script prints:
  - Drug name, company, year, failure reason
  - Predicted toxicity scores for all 12 ADMET endpoints
  - A "FLAG" column marking endpoints where prediction exceeds a threshold

Candidate list from the ADMET-Engine Project Report, Woche 37–38,
augmented with known small-molecule failures.

Author: Lead MLOps Engineer & Data Scientist (Persona 2/3 Hybrid)
"""

import sys
from pathlib import Path
from typing import Optional

# Ensure src/ is importable
_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Batch

from models.dmpnn import ADMETPredictor
from featurizer import smiles_to_data
from dataset import ENDPOINT_LABELS, NUM_ENDPOINTS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = _SRC.parent
MODEL_PATH = PROJECT_ROOT / "models" / "dmpnn_best.pt"
FALLBACK_PATH = PROJECT_ROOT / "models" / "dmpnn_v1.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Threshold for "flagged" toxicity prediction
WARN_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# 10 Failed Drug Candidates (Section 10, Woche 37–38)
# SMILES sourced from PubChem / DrugBank / literature.
# Bapineuzumab & TGN1412 (biologics) replaced with small-molecule failures.
# ---------------------------------------------------------------------------

FAILED_CANDIDATES: list[dict] = [
    {
        "name": "BIA 10-2474",
        "company": "Bial",
        "year": 2016,
        "phase": "Phase I",
        "failure_reason": "Neurotoxicity / off-target FAAH inhibition",
        "smiles": "O=C(N1CCN(CC1)C(=O)Nc2ccccc2)Cc3ccccc3",
    },
    {
        "name": "Fialuridine (FIAU)",
        "company": "Eli Lilly",
        "year": 1993,
        "phase": "Phase II",
        "failure_reason": "Hepatotoxicity (mtDNA depletion)",
        "smiles": "O=C1C=CN([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2F)C(=O)N1I",
    },
    {
        "name": "Torcetrapib",
        "company": "Pfizer",
        "year": 2006,
        "phase": "Phase III",
        "failure_reason": "Cardiotoxicity / increased mortality",
        "smiles": "CCOC(=O)N1CCN(CC1)C(=O)[C@@H](CC(C)C)NC(=O)c2ccc(cc2)C(F)(F)F",
    },
    {
        "name": "Semagacestat",
        "company": "Eli Lilly",
        "year": 2011,
        "phase": "Phase III",
        "failure_reason": "Hepatotoxicity + worsened cognition",
        "smiles": "CC(C)(C)[C@H](NC(=O)[C@@H](N)Cc1ccccc1)C(=O)N[C@@H](C)C(=O)N[C@@H](Cc2ccccc2)C(=O)O",
    },
    {
        "name": "Troglitazone (Rezulin)",
        "company": "Warner-Lambert / Pfizer",
        "year": 1997,
        "phase": "Post-market",
        "failure_reason": "Hepatotoxicity (idiosyncratic DILI)",
        "smiles": "CC1=C(C2=C(CCC(O2)(C)CCC(=O)N(CC3=CC=CC=C3)CC4=CC=C(C=C4)O)C(=C1C)O)O",
    },
    {
        "name": "Rofecoxib (Vioxx)",
        "company": "Merck",
        "year": 2004,
        "phase": "Post-market",
        "failure_reason": "Cardiotoxicity (COX-2 CV risk)",
        "smiles": "CS(=O)(=O)c1ccc(cc1)-c2c(c3ccccc3oc2=O)CCCC",
    },
    {
        "name": "Cerivastatin (Baycol)",
        "company": "Bayer",
        "year": 2001,
        "phase": "Post-market",
        "failure_reason": "Rhabdomyolysis / myotoxicity",
        "smiles": "CC(C)c1c(C(=O)NCc2ccccc2)c(c(n1C(C)C)c3ccc(F)cc3)CO[C@H](CC(=O)O)CC(O)O",
    },
    {
        "name": "Terfenadine (Seldane)",
        "company": "Hoechst Marion Roussel",
        "year": 1997,
        "phase": "Post-market",
        "failure_reason": "hERG inhibition / Torsades de Pointes",
        "smiles": "CC(C)(C)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(c3ccccc3)c4ccccc4",
    },
    {
        "name": "Mibefradil (Posicor)",
        "company": "Roche",
        "year": 1998,
        "phase": "Post-market",
        "failure_reason": "CYP3A4 / CYP2D6 drug-drug interactions",
        "smiles": "COc1cccc2c1OC(CC2)C(CN(C)CCCC3=NC4=C(N3)C=CC(=C4)F)C(C)C",
    },
    {
        "name": "Verubecestat",
        "company": "Merck",
        "year": 2017,
        "phase": "Phase III",
        "failure_reason": "No efficacy + worsened cognition (BACE1)",
        "smiles": "Cc1cnc(s1)NC(=O)[C@@H]2C[C@H]([C@@H](CN2C)C(=O)N)N3C=NC=N3",
    },
]


# ---------------------------------------------------------------------------
# Backtesting logic
# ---------------------------------------------------------------------------

def load_model(path: Optional[Path] = None) -> ADMETPredictor:
    """Load the best available checkpoint."""
    p = path or MODEL_PATH
    if not p.exists():
        p = FALLBACK_PATH
        print(f"[WARN] Best model not found, falling back to dummy: {p}")
    if not p.exists():
        raise FileNotFoundError(f"No model found at {MODEL_PATH} or {FALLBACK_PATH}")

    model = ADMETPredictor(
        node_dim=9, edge_dim=4, hidden_dim=300,
        num_tasks=NUM_ENDPOINTS, T=3, set2set_steps=3, dropout=0.1,
    )
    state = torch.load(str(p), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"[INFO] Model loaded: {p} ({sum(p.numel() for p in model.parameters()):,} params)")
    return model


def predict_candidates(model: ADMETPredictor, candidates: list[dict]) -> pd.DataFrame:
    """Run batch inference on all candidates and return a results DataFrame."""
    rows: list[dict] = []
    errors: list[dict] = []

    for cand in candidates:
        data = smiles_to_data(cand["smiles"])
        if data is None:
            errors.append({"name": cand["name"], "error": "SMILES parsing failed"})
            continue
        data.smiles = cand["smiles"]
        rows.append((cand, data))

    if not rows:
        return pd.DataFrame(errors)

    # Batch inference
    batch = Batch.from_data_list([r[1] for r in rows]).to(DEVICE)
    with torch.no_grad():
        preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    preds = torch.sigmoid(preds).cpu().numpy()

    # Build results
    results = []
    for i, (cand, _) in enumerate(rows):
        row = {
            "Drug": cand["name"],
            "Company": cand["company"],
            "Year": cand["year"],
            "Phase": cand["phase"],
            "Failure Reason": cand["failure_reason"],
        }
        for j, label in enumerate(ENDPOINT_LABELS):
            row[label] = round(float(preds[i, j]), 4)
            # Flag high-risk predictions
            if preds[i, j] > WARN_THRESHOLD:
                row[f"FLAG_{label}"] = "***"
            else:
                row[f"FLAG_{label}"] = ""
        results.append(row)

    for err in errors:
        results.append({
            "Drug": err["name"],
            "Company": "—",
            "Year": "—",
            "Phase": "—",
            "Failure Reason": err["error"],
            **{label: np.nan for label in ENDPOINT_LABELS},
        })

    return pd.DataFrame(results)


def print_table(df: pd.DataFrame):
    """Nicely formatted console output."""
    # Select key columns for display
    key_endpoints = [
        "hepatotoxicity", "hERG_inhibition", "CYP3A4_inhibition",
        "P_gp_substrate_inhibitor", "nephrotoxicity", "AMES_mutagenicity",
    ]
    disp_cols = ["Drug", "Failure Reason"] + key_endpoints
    disp = df[disp_cols].copy()

    # Extract flag columns
    flag_cols = [f"FLAG_{e}" for e in key_endpoints]
    for j, ep in enumerate(key_endpoints):
        if f"FLAG_{ep}" in df.columns:
            disp[ep] = disp[ep].astype(str) + " " + df[f"FLAG_{ep}"].astype(str)

    print("\n" + "=" * 120)
    print("  ADMET-ENGINE BACKTEST REPORT  —  10 Failed Drug Candidates")
    print("  Predictions with dummy/random model weights — for pipeline validation only")
    print("=" * 120)
    print(disp.to_string(index=False))
    print("=" * 120)
    print(f"  *** = Predicted toxicity risk > {WARN_THRESHOLD}")
    print(f"  Model: {'dmpnn_best.pt' if MODEL_PATH.exists() else 'dmpnn_v1.pt (dummy)'}")
    print("=" * 120 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model = load_model()
    df = predict_candidates(model, FAILED_CANDIDATES)
    print_table(df)

    # Also save full CSV
    out_path = PROJECT_ROOT / "data" / "processed" / "backtest_results.csv"
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[INFO] Full results saved to {out_path}")


if __name__ == "__main__":
    main()

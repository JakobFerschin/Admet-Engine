#!/usr/bin/env python
"""
ADMET-Engine — Clinical Drug Failure & Safe Control Backtester
Evaluates the scaffold-validated ChEMBL v37 Multi-Modal Ensemble against
withdrawn clinical drugs (hERG cardiotoxicity & CYP3A4 DDI failures) and
safe clinical reference controls using IUPAC-verified SMILES.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

import pandas as pd

from cheminformatics import analyze_molecule
from data_pipeline import standardize_molecule
from models.chembl_ensemble import ChEMBLADMETEnsemble

PROJECT_ROOT = _SRC.parent
MODELS_DIR = PROJECT_ROOT / "models"

CLINICAL_CANDIDATES: list[dict] = [
    {
        "name": "Terfenadine (Seldane)",
        "year": 1997,
        "failure_reason": "Withdrawn — hERG / Torsades de Pointes",
        "smiles": "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1",
    },
    {
        "name": "Astemizole (Hismanal)",
        "year": 1999,
        "failure_reason": "Withdrawn — Potent hERG Blocker (IC50 ~1 nM)",
        "smiles": "COc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc3ccc(F)cc3)CC2)cc1",
    },
    {
        "name": "Cisapride (Propulsid)",
        "year": 2000,
        "failure_reason": "Withdrawn — hERG QT Prolongation",
        "smiles": "COc1cc(N)c(Cl)cc1C(=O)NC1CCN(CCCOc2ccc(F)cc2)CC1OC",
    },
    {
        "name": "Ketoconazole",
        "year": 2013,
        "failure_reason": "FDA Index Strong CYP3A4 Inhibitor (Ki ~15 nM)",
        "smiles": "CC(=O)N1CCN(c2ccc(OC[C@H]3CO[C@](Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1",
    },
    {
        "name": "Mibefradil (Posicor)",
        "year": 1998,
        "failure_reason": "Withdrawn — Mechanism-Based CYP3A4 DDI",
        "smiles": "COCC(=O)O[C@]1(CCN(C)CCc2nc3ccccc3[nH]2)CCc2cc(F)ccc2[C@@H]1C(C)C",
    },
    {
        "name": "Haloperidol",
        "year": 2007,
        "failure_reason": "FDA QT Warning — Known hERG Liability",
        "smiles": "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1",
    },
    {
        "name": "Troglitazone (Rezulin)",
        "year": 2000,
        "failure_reason": "Withdrawn — Idiosyncratic Hepatotoxicity (DILI)",
        "smiles": "Cc1c(C)c2c(c(C)c1O)CCC(C)(COc1ccc(CC3SC(=O)NC3=O)cc1)O2",
    },
    {
        "name": "Rofecoxib (Vioxx)",
        "year": 2004,
        "failure_reason": "Withdrawn — COX-2 Cardiovascular Risk",
        "smiles": "CS(=O)(=O)c1ccc(C2=C(c3ccccc3)C(=O)OC2)cc1",
    },
    {
        "name": "Aspirin (Control)",
        "year": 1899,
        "failure_reason": "Safe Clinical Control — No hERG/CYP3A4 Liability",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
    },
    {
        "name": "Caffeine (Control)",
        "year": 1819,
        "failure_reason": "Safe Clinical Control — CYP1A2 Substrate",
        "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    },
]


def main() -> None:
    ensemble = ChEMBLADMETEnsemble(MODELS_DIR, device="cpu")
    mols = [standardize_molecule(c["smiles"]) for c in CLINICAL_CANDIDATES]
    preds = ensemble.predict_mols(mols)

    rows = []
    for cand, mol, pred in zip(CLINICAL_CANDIDATES, mols, preds):
        chem = analyze_molecule(mol, include_svg=False)
        h_ep = pred["ml_endpoints"]["hERG_inhibition"]
        c_ep = pred["ml_endpoints"]["CYP3A4_inhibition"]
        ad = pred["applicability_domain"]
        rows.append({
            "Drug": cand["name"],
            "Year": cand["year"],
            "Clinical_Profile": cand["failure_reason"],
            "hERG_prob": h_ep["probability"],
            "hERG_std": h_ep["epistemic_std"],
            "hERG_tier": h_ep["risk_tier"],
            "CYP3A4_prob": c_ep["probability"],
            "CYP3A4_std": c_ep["epistemic_std"],
            "CYP3A4_tier": c_ep["risk_tier"],
            "Max_Tanimoto": ad["max_tanimoto"],
            "AD_Status": ad["status"],
            "QED": chem["physchem"]["qed"],
            "BOILED_Egg": chem["boiled_egg"]["region"],
            "Alerts": chem["alerts"]["total_alerts"],
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 115)
    print("  ADMET-ENGINE v2.0 — CLINICAL PHARMACOLOGY BACKTEST (ChEMBL v37 Scaffold-Validated Ensemble)")
    print("=" * 115)
    print(df[["Drug", "hERG_prob", "hERG_tier", "CYP3A4_prob", "CYP3A4_tier", "Max_Tanimoto", "AD_Status", "QED"]].to_string(index=False))
    print("=" * 115 + "\n")

    out_path = PROJECT_ROOT / "data" / "processed" / "backtest_results.csv"
    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved backtest results to {out_path}")


if __name__ == "__main__":
    main()

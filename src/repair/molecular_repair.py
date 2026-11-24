"""
ADMET-Engine v3.0 (CardioPhys-3D) — WP5: Closed-Loop Molecular Repair Agent
===========================================================================
Automated medicinal chemistry optimization agent that structurally detoxifies
cardiotoxic leads (e.g., hERG / TdP liabilities) via curated Matched Molecular
Pair (MMP) RDKit reaction transforms grounded in Jamieson et al. (J. Med. Chem.
2006) hERG optimization strategies:
  - Aza-scanning (phenyl -> pyridine / pyrimidine) to reduce cLogP & disrupt
    Phe656 pi-stacking
  - Polar capping & hydroxylation (-OH, -CN, -CONH2) to raise TPSA out of
    Pfizer 3/75 promiscuity space
  - Basic amine pKa attenuation (N-acetylation, morpholine replacement,
    hydroxyethylation) to weaken Tyr652 cation-pi binding
  - Metabolic blocking (para-fluorination) & O-demethylation

Multi-objective beam search optimizes:
  1. Minimize cardiotoxicity / TdP risk score P(TdP) from any pluggable oracle
     (ChEMBL v37 ensemble, CardioPhys-3D WP4->WP2 ODE pipeline, or custom scorer)
  2. Preserve core scaffold & pharmacophore: ECFP4 Tanimoto(orig, new) >= min_tanimoto
  3. Maintain oral drug-likeness (QED), synthetic accessibility (SA-Score <= 4.5),
     and zero PAINS alerts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, GraphDescriptors, QED, rdMolDescriptors

from cheminformatics import compute_morgan_fp, screen_structural_alerts
from data_pipeline import standardize_molecule

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Synthetic Accessibility (SA) Score Estimator (1.0 = easy, 10.0 = very hard)
# ---------------------------------------------------------------------------

def compute_sa_score(mol: Chem.Mol) -> float:
    """
    Deterministic, self-contained Synthetic Accessibility (SA) complexity score
    inspired by Ertl & Schuffenhauer (J. Cheminform. 2009), bounded in [1.0, 10.0].
    Penalizes excessive molecular size, stereocenters, spiro/bridgehead atoms,
    macrocycles (>8-membered rings), and high topological complexity (BertzCT).
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return 10.0

    n_heavy = mol.GetNumHeavyAtoms()
    n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    ring_info = mol.GetRingInfo()
    n_macro = sum(1 for r in ring_info.AtomRings() if len(r) > 8)
    bertz = GraphDescriptors.BertzCT(mol) if n_heavy > 1 else 0.0

    size_pen = max(0.0, (n_heavy - 20) * 0.045)
    stereo_pen = math.log1p(n_stereo) * 0.55
    spiro_pen = math.log1p(n_spiro) * 0.85
    bridge_pen = math.log1p(n_bridge) * 0.90
    macro_pen = n_macro * 1.15
    ct_pen = min(2.2, math.log1p(max(bertz, 0.0)) * 0.22)

    raw = 1.25 + size_pen + stereo_pen + spiro_pen + bridge_pen + macro_pen + ct_pen
    return round(float(max(1.0, min(10.0, raw))), 3)


# ---------------------------------------------------------------------------
# Curated Medicinal Chemistry hERG-Detoxification MMP Transformations
# ---------------------------------------------------------------------------

_REPAIR_TRANSFORMS: list[tuple[str, str]] = [
    # 1. Aza-scan: unsubstituted aromatic CH -> aromatic N (Phenyl -> Pyridine)
    ("Aza-scan (Phenyl->Pyridine)", "[c;H1;r6:1]1[c:2][c:3][c:4][c:5][c:6]1>>[n:1]1[c:2][c:3][c:4][c:5][c:6]1"),
    # 2. Aromatic hydroxylation (raises TPSA +20.2 A^2, drops cLogP ~0.7)
    ("Aromatic Hydroxylation (+OH)", "[c;H1;r6:1]>>[c:1]O"),
    # 3. Aromatic fluorination (blocks CYP oxidation & lowers neighboring amine pKa)
    ("Aromatic Fluorination (+F)", "[c;H1;r6:1]>>[c:1]F"),
    # 4. Aromatic cyanation (electron-withdrawing, +23.8 A^2 TPSA)
    ("Aromatic Cyanation (+CN)", "[c;H1;r6:1]>>[c:1]C#N"),
    # 5. Aromatic carboxamide (+43.1 A^2 TPSA, strong hERG drop)
    ("Aromatic Amidation (+CONH2)", "[c;H1;r6:1]>>[c:1]C(=O)N"),
    # 6. O-Demethylation (Ar-OMe -> Ar-OH)
    ("O-Demethylation (OMe->OH)", "[c:1][O:2][CH3]>>[c:1][OH:2]"),
    # 7. N-Demethylation + Acetylation (attenuates basic amine cation-pi with Tyr652)
    ("N-Acetylation (Amine->Amide)", "[NX3;H0;!$(NC=O):1]([CH3])([#6:2])[#6:3]>>[N:1](C(=O)C)([#6:2])[#6:3]"),
    # 8. Basic N-Me -> N-Hydroxyethyl (drops pKa & cLogP)
    ("N-Hydroxyethylation", "[NX3;H0;!$(NC=O):1]([CH3])([#6:2])[#6:3]>>[N:1](CCO)([#6:2])[#6:3]"),
    # 9. Piperidine CH2 -> Morpholine O (drops piperidine pKa by ~2 log units!)
    ("Piperidine->Morpholine", "[N:1]1[C:2][C:3][CH2:4][C:5][C:6]1>>[N:1]1[C:2][C:3][O:4][C:5][C:6]1"),
    # 10. Secondary amine -> Sulfonamide (neutralizes basicity)
    ("N-Mesylation (Amine->Sulfonamide)", "[NX3;H1;!$(NC=O);!$(NS=O):1]>>[N:1]S(=O)(=O)C"),
    # 11. tert-Butyl -> Methoxy (removes lipophilic hotspot)
    ("tBu->OMe Polar Replacement", "[c:1]C(C)(C)C>>[c:1]OC"),
    # 12. Aliphatic CH2 -> C=O adjacent to N (lactamization / basicity removal)
    ("Alpha-Oxidation (Amine->Lactam)", "[N;r6:1][CH2;r6:2][C;r6:3]>>[N:1][C:2](=O)[C:3]"),
]

_COMPILED_REPAIR_RXNS: list[tuple[str, AllChem.ChemicalReaction]] = [
    (name, AllChem.ReactionFromSmarts(smarts))
    for name, smarts in _REPAIR_TRANSFORMS
]


# ---------------------------------------------------------------------------
# Candidate Dataclass & Closed-Loop Beam-Search Agent
# ---------------------------------------------------------------------------

@dataclass
class RepairCandidate:
    smiles: str
    risk_score: float
    delta_risk: float
    tanimoto: float
    qed: float
    sa_score: float
    clogp: float
    tpsa: float
    pains_count: int
    edit_path: list[str] = field(default_factory=list)
    objective: float = 0.0


class ClosedLoopRepairAgent:
    """
    Multi-objective closed-loop molecular repair agent.

    Takes any batch scoring callable `risk_oracle(smiles_list) -> list[float]`
    returning a cardiotoxicity / TdP risk score in [0, 1] (lower = safer),
    such as:
      - `CardioPhysEndToEndPipeline` (WP4->WP2 ORd-CiPA P(TdP))
      - `ChEMBLADMETEnsemble` (v2.0 calibrated hERG P(active))
    """

    def __init__(
        self,
        risk_oracle: Callable[[list[str]], list[float]],
        min_tanimoto: float = 0.55,
        max_sa_score: float = 4.8,
        min_qed: float = 0.30,
        require_zero_pains: bool = True,
    ):
        self.risk_oracle = risk_oracle
        self.min_tanimoto = min_tanimoto
        self.max_sa_score = max_sa_score
        self.min_qed = min_qed
        self.require_zero_pains = require_zero_pains

    def _expand_one_step(
        self,
        mol: Chem.Mol,
        max_per_rxn: int = 4,
    ) -> list[tuple[str, Chem.Mol, str]]:
        """Apply all curated MMP transforms and return unique sanitized neighbors."""
        neighbors: dict[str, tuple[Chem.Mol, str]] = {}
        for rxn_name, rxn in _COMPILED_REPAIR_RXNS:
            try:
                prods = rxn.RunReactants((mol,))
            except Exception:
                continue
            added = 0
            for prod_tuple in prods:
                if added >= max_per_rxn:
                    break
                cand = prod_tuple[0]
                try:
                    Chem.SanitizeMol(cand)
                    smi = Chem.MolToSmiles(cand, canonical=True)
                    if "." in smi or smi in neighbors:
                        continue
                    neighbors[smi] = (cand, rxn_name)
                    added += 1
                except Exception:
                    continue
        return [(smi, m, name) for smi, (m, name) in neighbors.items()]

    def repair(
        self,
        seed_smiles: str,
        n_rounds: int = 2,
        beam_width: int = 6,
        top_k: int = 5,
    ) -> dict[str, object]:
        """
        Run closed-loop iterative structural repair on `seed_smiles`.

        Returns a structured report containing the seed baseline metrics and the
        top-k Pareto-optimal repaired analogs ranked by multi-objective fitness.
        """
        seed_mol = standardize_molecule(seed_smiles)
        if seed_mol is None or seed_mol.GetNumAtoms() == 0:
            raise ValueError(f"Invalid seed SMILES: {seed_smiles}")

        seed_can = Chem.MolToSmiles(seed_mol, canonical=True)
        seed_fp = compute_morgan_fp(seed_mol)
        seed_risk = float(self.risk_oracle([seed_can])[0])

        seed_profile = {
            "smiles": seed_can,
            "risk_score": round(seed_risk, 4),
            "qed": round(float(QED.qed(seed_mol)), 3),
            "sa_score": compute_sa_score(seed_mol),
            "clogp": round(float(Crippen.MolLogP(seed_mol)), 2),
            "tpsa": round(float(rdMolDescriptors.CalcTPSA(seed_mol)), 1),
        }

        visited: set[str] = {seed_can}
        beam: list[tuple[Chem.Mol, list[str]]] = [(seed_mol, [])]
        all_valid: list[RepairCandidate] = []

        for _ in range(n_rounds):
            pool: list[tuple[str, Chem.Mol, list[str], float, float, float, float, float, int]] = []

            for curr_mol, path in beam:
                for smi, cand_mol, rxn_name in self._expand_one_step(curr_mol):
                    if smi in visited:
                        continue
                    visited.add(smi)

                    # Fast medicinal chemistry constraint filtering before oracle call
                    fp = compute_morgan_fp(cand_mol)
                    sim = float(DataStructs.TanimotoSimilarity(seed_fp, fp))
                    if sim < self.min_tanimoto:
                        continue

                    qed_v = float(QED.qed(cand_mol))
                    sa_v = compute_sa_score(cand_mol)
                    if qed_v < self.min_qed or sa_v > self.max_sa_score:
                        continue

                    alerts = screen_structural_alerts(cand_mol)
                    if self.require_zero_pains and alerts["pains_count"] > 0:
                        continue

                    clogp_v = float(Crippen.MolLogP(cand_mol))
                    tpsa_v = float(rdMolDescriptors.CalcTPSA(cand_mol))
                    pool.append(
                        (smi, cand_mol, path + [rxn_name], sim, qed_v, sa_v, clogp_v, tpsa_v, alerts["pains_count"])
                    )

            if not pool:
                break

            # Batch-score entire candidate pool with the toxicity/TdP oracle
            risks = self.risk_oracle([p[0] for p in pool])
            round_candidates: list[tuple[RepairCandidate, Chem.Mol]] = []

            for item, risk_v in zip(pool, risks):
                smi, cand_mol, path, sim, qed_v, sa_v, clogp_v, tpsa_v, pains_c = item
                risk_f = float(risk_v)
                delta = risk_f - seed_risk
                # Multi-objective fitness (higher = better):
                # strongly reward risk reduction + preserve QED + similarity + low SA
                obj = (-delta) * 2.5 + 0.35 * qed_v + 0.25 * sim - 0.05 * sa_v
                rc = RepairCandidate(
                    smiles=smi,
                    risk_score=round(risk_f, 4),
                    delta_risk=round(delta, 4),
                    tanimoto=round(sim, 3),
                    qed=round(qed_v, 3),
                    sa_score=sa_v,
                    clogp=round(clogp_v, 2),
                    tpsa=round(tpsa_v, 1),
                    pains_count=pains_c,
                    edit_path=path,
                    objective=round(obj, 4),
                )
                all_valid.append(rc)
                round_candidates.append((rc, cand_mol))

            # Advance top `beam_width` candidates to next edit round
            round_candidates.sort(key=lambda x: x[0].objective, reverse=True)
            beam = [(m, rc.edit_path) for rc, m in round_candidates[:beam_width]]

        # Rank strictly by risk reduction first among candidates improving safety
        all_valid.sort(key=lambda c: (c.risk_score, -c.objective))
        best = all_valid[:top_k]

        return {
            "seed": seed_profile,
            "total_evaluated": len(all_valid),
            "repaired_candidates": best,
            "best_delta_risk": best[0].delta_risk if best else 0.0,
        }

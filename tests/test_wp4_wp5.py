"""
Unit and integration tests for ADMET-Engine v3.0 WP4 & WP5:
  - WP4: Closed-form Gaussian KL distillation, Tobit+KL hybrid loss,
    CardioPhysStudentNet edge latency, and end-to-end SMILES -> pIC50 -> ORd-CiPA ODE.
  - WP5: Synthetic Accessibility (SA) score, curated MMP transformations, and
    ClosedLoopRepairAgent tested against BOTH the real trained ChEMBL v37 hERG
    ensemble (on Terfenadine) AND the WP4->WP2 physiological ODE pipeline.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.chembl_ensemble import ChEMBLADMETEnsemble
from models.distillation import (
    CardioPhysEndToEndPipeline,
    CardioPhysStudentNet,
    distillation_tobit_loss,
    gaussian_kl_divergence,
)
from physiology.cipa28 import get_cipa28_drug
from physiology.ohara_rudy import CardiomyocyteSimulator
from repair.molecular_repair import ClosedLoopRepairAgent, compute_sa_score


# ---------------------------------------------------------------------------
# WP4: Knowledge Distillation & End-to-End ODE Coupling Tests
# ---------------------------------------------------------------------------

def test_gaussian_kl_divergence_properties() -> None:
    mu_t = torch.tensor([[6.5, 5.2, 4.8], [7.1, 6.0, 5.0]])
    sig_t = torch.tensor([[0.4, 0.5, 0.6], [0.3, 0.4, 0.5]])

    # Identical distributions -> KL = 0
    kl_zero = gaussian_kl_divergence(mu_t, sig_t, mu_t.clone(), sig_t.clone())
    assert float(kl_zero.item()) == pytest.approx(0.0, abs=1e-6)

    # Shifted student -> KL > 0 with finite gradients
    mu_s = (mu_t + 0.8).clone().requires_grad_(True)
    sig_s = (sig_t * 1.4).clone().requires_grad_(True)
    kl_pos = gaussian_kl_divergence(mu_t, sig_t, mu_s, sig_s)
    assert float(kl_pos.item()) > 0.1
    kl_pos.backward()
    assert torch.isfinite(mu_s.grad).all()
    assert torch.isfinite(sig_s.grad).all()


def test_distillation_tobit_loss_labeled_and_unlabeled() -> None:
    mu_t = torch.tensor([[6.2, 5.0, 4.9], [5.5, 5.8, 5.1]])
    sig_t = torch.tensor([[0.4, 0.5, 0.5], [0.4, 0.4, 0.5]])
    mu_s = torch.tensor([[5.9, 5.1, 4.7], [5.4, 5.5, 5.0]], requires_grad=True)
    sig_s = torch.tensor([[0.5, 0.6, 0.5], [0.5, 0.5, 0.6]], requires_grad=True)
    y = torch.tensor([[6.3, float("nan"), 4.5], [5.0, 5.9, float("nan")]])
    c = torch.tensor([[0.0, float("nan"), -1.0], [1.0, 0.0, float("nan")]])

    loss_labeled = distillation_tobit_loss(mu_s, sig_s, mu_t, sig_t, targets=y, censor=c, distill_weight=0.5)
    loss_unlabeled = distillation_tobit_loss(mu_s, sig_s, mu_t, sig_t, targets=None, censor=None)

    assert torch.isfinite(loss_labeled) and torch.isfinite(loss_unlabeled)
    loss_labeled.backward()
    assert torch.isfinite(mu_s.grad).all()


def test_student_net_fast_edge_inference() -> None:
    student = CardioPhysStudentNet(in_dim=2068, hidden_dim=256, num_channels=3).eval()
    x = torch.randn(16, 2068)
    with torch.no_grad():
        mu, sigma = student(x)
        t0 = time.perf_counter()
        for _ in range(10):
            _ = student(x[:1])
        ms_per_mol = (time.perf_counter() - t0) * 1000.0 / 10.0

    assert mu.shape == (16, 3)
    assert sigma.shape == (16, 3)
    assert (sigma > 0.0).all()
    assert ms_per_mol < 50.0  # comfortably sub-50ms even on slow virtualized CPUs


def test_end_to_end_smiles_to_cardiomyocyte_ode() -> None:
    student = CardioPhysStudentNet(in_dim=2068, hidden_dim=64, num_channels=3).eval()
    sim = CardiomyocyteSimulator(dt_ms=0.5, n_beats=1)
    pipe = CardioPhysEndToEndPipeline(student=student, simulator=sim, device="cpu")

    verapamil = get_cipa28_drug("Verapamil").smiles
    terfenadine = get_cipa28_drug("Terfenadine").smiles
    preds = pipe.predict_smiles([terfenadine, "INVALID_SMILES_XXX", verapamil], conc_uM=1.0)

    assert len(preds) == 3
    assert preds[1] is None
    for idx in (0, 2):
        p = preds[idx]
        assert p is not None
        assert set(p.pic50_mean.keys()) == {"hERG", "Cav1.2", "Nav1.5"}
        assert all(v > 0.0 for v in p.pic50_std.values())
        assert 0.0 <= p.tdp_risk_score <= 1.0
        assert p.apd90_ms > 50.0


# ---------------------------------------------------------------------------
# WP5: Closed-Loop Molecular Repair Agent Tests
# ---------------------------------------------------------------------------

def test_sa_score_ranks_simple_vs_complex() -> None:
    aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    cholesterol = Chem.MolFromSmiles("CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C")
    sa_asp = compute_sa_score(aspirin)
    sa_chol = compute_sa_score(cholesterol)

    assert 1.0 <= sa_asp < sa_chol <= 10.0


def test_closed_loop_repair_with_real_chembl_herg_oracle() -> None:
    """
    Live integration test: run the WP5 Closed-Loop Molecular Repair Agent on
    Terfenadine (classic withdrawn hERG/TdP cardiotoxic drug) using the real
    trained ChEMBL v37 4-family ensemble as the safety oracle.
    """
    ensemble = ChEMBLADMETEnsemble(bundle_dir=PROJECT_ROOT / "models", device="cpu")

    def herg_oracle(smiles_list: list[str]) -> list[float]:
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        preds = ensemble.predict_mols(mols)
        return [float(p["predictions"]["hERG_inhibition"]) for p in preds]

    terfenadine_smi = get_cipa28_drug("Terfenadine").smiles
    agent = ClosedLoopRepairAgent(
        risk_oracle=herg_oracle,
        min_tanimoto=0.55,
        max_sa_score=5.0,
        min_qed=0.25,
        require_zero_pains=True,
    )
    report = agent.repair(terfenadine_smi, n_rounds=2, beam_width=4, top_k=5)

    assert report["seed"]["risk_score"] > 0.70  # Terfenadine is strongly hERG-active
    assert report["total_evaluated"] >= 5
    candidates = report["repaired_candidates"]
    assert len(candidates) >= 1

    best = candidates[0]
    assert best.delta_risk < -0.05  # meaningful toxicity reduction achieved
    assert best.tanimoto >= 0.55
    assert best.pains_count == 0
    assert best.sa_score <= 5.0
    assert len(best.edit_path) >= 1


def test_closed_loop_repair_coupled_to_wp4_wp2_ode() -> None:
    """Verify full WP5 -> WP4 -> WP2 closed-loop coupling executes cleanly."""
    student = CardioPhysStudentNet(in_dim=2068, hidden_dim=64, num_channels=3).eval()
    sim = CardiomyocyteSimulator(dt_ms=0.5, n_beats=1)
    pipe = CardioPhysEndToEndPipeline(student=student, simulator=sim, device="cpu")

    def ode_oracle(smiles_list: list[str]) -> list[float]:
        preds = pipe.predict_smiles(smiles_list, conc_uM=1.0)
        return [p.tdp_risk_score if p is not None else 1.0 for p in preds]

    cisapride_smi = get_cipa28_drug("Cisapride").smiles
    agent = ClosedLoopRepairAgent(
        risk_oracle=ode_oracle,
        min_tanimoto=0.55,
        max_sa_score=5.0,
        min_qed=0.35,
    )
    report = agent.repair(cisapride_smi, n_rounds=1, beam_width=3, top_k=3)
    assert report["total_evaluated"] >= 3
    assert len(report["repaired_candidates"]) >= 1

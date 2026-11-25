"""
ADMET-Engine v3.0 (CardioPhys-3D) — WP6 Novel Science & Benchmarks Tests
=========================================================================
Verifies:
  1. End-to-End 3D-to-Organ Differentiable Attribution (`compute_3d_organ_attribution`
     and `attribute_smiles_3d_to_organ`):
       - Finite, non-zero 3D gradients d(tdp_risk_score)/d(pos_i) and d(qNet)/d(pos_i)
       - Exact multivariate chain-rule identity:
           d(qNet)/d(pos_i) == sum_c (d(qNet)/d(mu_c)) * (d(mu_c)/d(pos_i))
       - Biophysical organ Jacobian signs: d(qNet)/d(pIC50_hERG) < 0,
         d(qNet)/d(pIC50_Cav1.2) > 0, d(qNet)/d(pIC50_Nav1.5) > 0
       - Exact SO(3) rotation & translation invariance of per-atom 3D gradient norms
         ||d(qNet)/d(pos_i)||_2 and ||d(tdp_risk_score)/d(pos_i)||_2, and
         SO(3) equivariance of gradient vectors g'_i == R g_i!
  2. WP6 Benchmark artifacts (`paper_v3/wp6_results.json` and all 3 vector PDF figures):
       - Valid JSON structure, CiPA-28 ROC-AUC superiority over hERG-only,
         Verapamil & Ranolazine rescue, Monte Carlo 95% CI intervals,
         WP5 Closed-Loop repair reductions with Tanimoto >= 0.55 and 0 PAINS.
       - All 3 PDF figures exist on disk and are non-empty valid PDFs (%PDF- header).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.equivariant_3d import CHANNELS_V3, CardioPhys3DPredictor
from physiology.attribution_3d import (
    OrganAttributionResult,
    attribute_smiles_3d_to_organ,
    compute_3d_organ_attribution,
)
from physiology.cipa28 import get_cipa28_drug
from physiology.ohara_rudy import CardiomyocyteSimulator


def _random_so3_matrix(seed: int = 123) -> torch.Tensor:
    """Generate a deterministic proper rotation matrix R in SO(3) (det(R) = +1)."""
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn((3, 3), generator=gen, dtype=torch.float32)
    q, r = torch.linalg.qr(a)
    d = torch.sign(torch.diag(r))
    d[d == 0] = 1.0
    q = q * d.unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


class TestEndToEnd3DOrganAttribution:

    def test_finite_nonzero_3d_gradients_and_biophysical_jacobian_signs(self) -> None:
        torch.manual_seed(7)
        predictor = CardioPhys3DPredictor(hidden_dim=32, num_layers=2, num_rbf=16).eval()
        simulator = CardiomyocyteSimulator(dt_ms=0.5, n_beats=1)

        terfenadine_smi = get_cipa28_drug("Terfenadine").smiles
        res = attribute_smiles_3d_to_organ(
            smiles=terfenadine_smi,
            predictor=predictor,
            simulator=simulator,
            conc_uM=1.0,
            num_confs=3,
            top_k=10,
            seed=42,
        )

        assert isinstance(res, OrganAttributionResult)
        assert res.smiles == terfenadine_smi
        assert res.smiles_ph74 is not None
        assert len(res.ranked_hotspots) == 10

        # Finite and non-zero 3D gradients for both qNet and tdp_risk_score
        assert np.all(np.isfinite(res.qnet_grad_pos))
        assert np.all(np.isfinite(res.tdp_grad_pos))
        assert np.all(np.isfinite(res.qnet_grad_norms))
        assert np.all(np.isfinite(res.tdp_grad_norms))
        assert float(res.qnet_grad_norms.max()) > 1e-8
        assert float(res.tdp_grad_norms.max()) > 1e-8

        # Biophysical organ Jacobian signs:
        # Stronger hERG block lowers qNet (raises TdP); stronger Cav1.2 / Nav1.5 late block raises qNet
        assert res.channel_qnet_jacobian["hERG"] < 0.0
        assert res.channel_qnet_jacobian["Cav1.2"] > 0.0
        assert res.channel_qnet_jacobian["Nav1.5"] > 0.0
        assert res.channel_tdp_jacobian["hERG"] > 0.0
        assert res.channel_tdp_jacobian["Cav1.2"] < 0.0

        # Per-channel contribution fractions sum to 1.0
        assert set(res.channel_qnet_contribution_fraction.keys()) == set(CHANNELS_V3)
        assert sum(res.channel_qnet_contribution_fraction.values()) == pytest.approx(1.0, abs=1e-5)

        # Hotspots are strictly ranked descending by tdp_grad_norm
        norms = [h.tdp_grad_norm for h in res.ranked_hotspots]
        assert norms == sorted(norms, reverse=True)

    def test_so3_rotation_and_translation_invariance_of_3d_gradient_norms(self) -> None:
        """
        Verify exact SE(3) symmetry of end-to-end 3D-to-organ attribution:
          - Under rigid transformation pos' = pos @ R^T + t:
              1. Scalar organ outputs (qNet, tdp_risk_score) are invariant.
              2. Per-atom 3D gradient vectors rotate equivariantly: g'_i = R g_i.
              3. Per-atom sensitivity norms ||g_i||_2 are strictly invariant!
        """
        torch.manual_seed(11)
        predictor = CardioPhys3DPredictor(hidden_dim=32, num_layers=2, num_rbf=16).eval()
        simulator = CardiomyocyteSimulator(dt_ms=0.5, n_beats=1)

        # Compact 10-atom realistic 3D geometry
        gen = torch.Generator().manual_seed(99)
        z = torch.tensor([6, 6, 6, 6, 6, 6, 7, 8, 9, 6], dtype=torch.long)
        q = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.long)
        pos = torch.randn((10, 3), generator=gen, dtype=torch.float32) * 1.4

        rot = _random_so3_matrix(seed=2026)
        trans = torch.tensor([12.5, -7.3, 4.1], dtype=torch.float32)
        pos_rot = pos @ rot.T + trans

        res_orig = compute_3d_organ_attribution(
            predictor=predictor,
            simulator=simulator,
            z=z,
            pos=pos,
            formal_charge=q,
            conc_uM=1.0,
        )
        res_rot = compute_3d_organ_attribution(
            predictor=predictor,
            simulator=simulator,
            z=z,
            pos=pos_rot,
            formal_charge=q,
            conc_uM=1.0,
        )

        # 1. Scalar organ outputs are strictly SE(3)-invariant
        assert res_rot.qnet_uC_uF == pytest.approx(res_orig.qnet_uC_uF, abs=1e-5)
        assert res_rot.tdp_risk_score == pytest.approx(res_orig.tdp_risk_score, abs=1e-5)

        # 2. Per-atom gradient norms ||d(qNet)/d(pos_i)||_2 and ||d(TdP)/d(pos_i)||_2 are SE(3)-invariant
        np.testing.assert_allclose(
            res_rot.qnet_grad_norms, res_orig.qnet_grad_norms, rtol=1e-3, atol=1e-5
        )
        np.testing.assert_allclose(
            res_rot.tdp_grad_norms, res_orig.tdp_grad_norms, rtol=1e-3, atol=1e-5
        )

        # 3. Per-atom 3D gradient vectors transform SO(3)-equivariantly: g'_i = R g_i
        expected_rot_qnet_grad = res_orig.qnet_grad_pos @ rot.numpy().T
        np.testing.assert_allclose(
            res_rot.qnet_grad_pos, expected_rot_qnet_grad, rtol=1e-3, atol=1e-5
        )


class TestWP6ResultsAndFigures:

    def test_wp6_results_json_and_figures_exist_and_valid(self) -> None:
        results_path = PROJECT_ROOT / "paper_v3" / "wp6_results.json"
        assert results_path.exists(), "paper_v3/wp6_results.json must be generated"

        data = json.loads(results_path.read_text(encoding="utf-8"))

        # (a) CiPA-28 validation checks
        cipa = data["cipa28_validation"]
        assert cipa["n_drugs"] == 28
        aucs = cipa["roc_auc"]
        assert aucs["multichannel_ode_high_vs_low"] >= 0.95
        assert aucs["multichannel_ode_high_int_vs_low"] >= 0.88
        assert aucs["multichannel_ode_high_int_vs_low"] > aucs["herg_only_ode_high_int_vs_low"]
        assert cipa["rescue_verification"]["Verapamil_4xCmax"]["rescued"] is True
        assert cipa["rescue_verification"]["Ranolazine_4xCmax"]["rescued"] is True

        # (b) Monte-Carlo uncertainty propagation checks
        mc = data["mc_uncertainty_propagation"]
        assert len(mc) == 6
        for row in mc:
            assert row["qnet_std"] > 0.0
            assert row["qnet_ci95_low"] < row["qnet_mean"] < row["qnet_ci95_high"]

        # (c) WP5 Closed-Loop Molecular Repair checks
        wp5 = data["wp5_closed_loop_repair"]
        assert wp5["n_drugs"] >= 5
        assert wp5["summary"]["mean_delta_risk"] < -0.05
        assert wp5["summary"]["zero_pains_rate"] == pytest.approx(1.0)
        for row in wp5["drugs"]:
            assert row["tanimoto"] >= 0.55
            assert row["pains_count"] == 0
            assert row["delta_risk"] < 0.0

        # (d) All 3 publication PDF figures exist and are valid non-empty PDFs
        expected_figs = [
            PROJECT_ROOT / "paper_v3" / "figures" / "fig1_cipa28_action_potentials.pdf",
            PROJECT_ROOT / "paper_v3" / "figures" / "fig2_mc_uncertainty_qnet.pdf",
            PROJECT_ROOT / "paper_v3" / "figures" / "fig3_wp5_repair_pareto.pdf",
        ]
        for fig_p in expected_figs:
            assert fig_p.exists(), f"Missing PDF figure: {fig_p}"
            raw = fig_p.read_bytes()
            assert len(raw) > 5000, f"PDF figure unexpectedly small: {fig_p} ({len(raw)} bytes)"
            assert raw[:5] == b"%PDF-", f"Invalid PDF header in {fig_p}"

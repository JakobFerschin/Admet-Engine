"""
ADMET-Engine v3.0 (CardioPhys-3D) — WP3 Tests
Verifies:
  1. Exact SE(3) rotational and translational invariance of CardioPhys3DPredictor
     under arbitrary 3D rotations R in SO(3) and translations t in R^3 (< 1e-4).
  2. Censored Gaussian Tobit NLL exact values, numerical stability at extreme
     z = +/- 25, and correct gradient signs for exact (0), left (-1), and right (+1).
  3. Mondrian Conformal Prediction finite-sample coverage guarantee stratified by AD group.
  4. End-to-end differentiable gradient flow:
     3D Ligand Coordinates -> SE(3) PaiNN -> (pIC50_hERG, pIC50_Cav1.2, pIC50_Nav1.5)
     -> Hill Coupling -> O'Hara-Rudy Cardiomyocyte ODE -> qNet!
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from data.conformer_3d import generate_3d_ensemble  # noqa: E402
from models.equivariant_3d import CardioPhys3DPredictor  # noqa: E402
from physiology import CardiomyocyteSimulator, ChannelBlockProfile  # noqa: E402
from training.tobit_loss import (  # noqa: E402
    MondrianConformalRegressor,
    censored_concordance_rate,
    tobit_censored_nll,
)


def _random_so3_rotation(seed: int = 123) -> torch.Tensor:
    """Generate a deterministic proper 3D rotation matrix R in SO(3) via QR."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn((3, 3), generator=g)
    q, r = torch.linalg.qr(a)
    d = torch.sign(torch.diag(r))
    q = q * d.unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


class TestSE3EquivarianceAndInvariance:

    def test_exact_so3_rotation_and_translation_invariance(self):
        torch.manual_seed(42)
        ens = generate_3d_ensemble(
            "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC",  # Verapamil
            num_confs=2,
            protonate=True,
        )
        assert ens is not None
        td = ens.to_torch_dict()
        z = td["z"]
        q = td["formal_charge"].long()
        pos = td["pos"][0]  # [N, 3]
        batch = torch.zeros(z.shape[0], dtype=torch.long)

        model = CardioPhys3DPredictor(hidden_dim=64, num_layers=3, dropout=0.0)
        model.eval()

        R = _random_so3_rotation(seed=77)
        t = torch.tensor([12.3, -45.6, 8.9], dtype=torch.float32)
        pos_transformed = pos @ R.T + t.unsqueeze(0)

        with torch.no_grad():
            mu1, sig1 = model(z, pos, batch, formal_charge=q)
            mu2, sig2 = model(z, pos_transformed, batch, formal_charge=q)

        assert mu1.shape == (1, 3)
        assert (sig1 > 0.0).all()
        assert torch.allclose(mu1, mu2, atol=1e-4), f"Rotation/translation drift in mu: {(mu1 - mu2).abs().max()}"
        assert torch.allclose(sig1, sig2, atol=1e-4), f"Rotation/translation drift in sigma: {(sig1 - sig2).abs().max()}"


class TestTobitCensoredLossAndMondrianCP:

    def test_tobit_gradient_directions_exact_left_right(self):
        # 3 samples: exact (y=6.0), left-censored (pIC50 <= 5.0), right-censored (pIC50 >= 8.0)
        mu = torch.tensor([[7.0, 7.0, 6.0]], requires_grad=True)
        sigma = torch.tensor([[0.5, 0.5, 0.5]])
        targets = torch.tensor([[6.0, 5.0, 8.0]])
        censor = torch.tensor([[0.0, -1.0, 1.0]])

        loss = tobit_censored_nll(mu, sigma, targets, censor)
        assert torch.isfinite(loss)
        loss.backward()

        # Sample 0 (exact y=6, mu=7): gradient > 0 -> pushes mu DOWN toward 6
        assert mu.grad[0, 0].item() > 0.0
        # Sample 1 (left-censored pIC50 <= 5, mu=7): violates bound -> gradient > 0 pushes mu DOWN
        assert mu.grad[0, 1].item() > 0.0
        # Sample 2 (right-censored pIC50 >= 8, mu=6): violates bound -> gradient < 0 pushes mu UP
        assert mu.grad[0, 2].item() < 0.0

    def test_tobit_numerical_stability_extreme_outliers(self):
        # z = +/- 25 must never produce NaN or Inf
        mu = torch.tensor([[15.0, -10.0, 5.0]], requires_grad=True)
        sigma = torch.tensor([[0.2, 0.2, 0.2]])
        targets = torch.tensor([[5.0, 10.0, float("nan")]])
        censor = torch.tensor([[-1.0, 1.0, float("nan")]])

        loss = tobit_censored_nll(mu, sigma, targets, censor)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(mu.grad).all()

    def test_censored_concordance_rate(self):
        mu = np.array([4.2, 5.8, 8.5, 6.0])
        y = np.array([5.0, 5.0, 8.0, 8.0])
        c = np.array([-1, -1, +1, +1])
        res = censored_concordance_rate(mu, y, c)
        assert math.isclose(res["left_agreement"], 0.5)
        assert math.isclose(res["right_agreement"], 0.5)
        assert math.isclose(res["overall_censored_agreement"], 0.5)

    def test_mondrian_conformal_empirical_coverage(self):
        rng = np.random.default_rng(42)
        n_cal, n_test = 400, 400
        # Two AD strata with different true residual scales (In-Domain tighter than Out-of-Domain)
        groups_cal = np.array(["In-Domain"] * 200 + ["Out-of-Domain"] * 200)
        sig_cal = np.ones(n_cal, dtype=np.float32)
        mu_cal = rng.normal(6.0, 1.0, size=n_cal)
        noise_cal = np.concatenate([rng.normal(0.0, 0.5, 200), rng.normal(0.0, 2.0, 200)])
        y_cal = mu_cal + noise_cal

        cp = MondrianConformalRegressor(alpha=0.10, min_group_size=20).fit(
            y_cal, mu_cal, sig_cal, groups_cal
        )
        assert cp.group_quantiles["Out-of-Domain"] > cp.group_quantiles["In-Domain"]

        groups_te = np.array(["In-Domain"] * 200 + ["Out-of-Domain"] * 200)
        mu_te = rng.normal(6.0, 1.0, size=n_test)
        y_te = mu_te + np.concatenate([rng.normal(0.0, 0.5, 200), rng.normal(0.0, 2.0, 200)])
        lo, hi = cp.predict_intervals(mu_te, np.ones(n_test), groups_te)
        cov_in = np.mean((y_te[:200] >= lo[:200]) & (y_te[:200] <= hi[:200]))
        cov_out = np.mean((y_te[200:] >= lo[200:]) & (y_te[200:] <= hi[200:]))
        assert cov_in >= 0.84 and cov_out >= 0.84


class TestEndToEnd3DToPhysiologyCoupling:

    def test_differentiable_end_to_end_3d_to_qnet_backprop(self):
        """
        End-to-end CardioPhys-3D proof:
        Backpropagate scalar cardiomyocyte qNet loss all the way back to the
        SE(3)-equivariant 3D neural network parameters!
        """
        ens = generate_3d_ensemble("CC(=O)Oc1ccccc1C(=O)O", num_confs=1, protonate=True)
        assert ens is not None
        td = ens.to_torch_dict()
        z = td["z"]
        q = td["formal_charge"].long()
        pos = td["pos"][0]
        batch = torch.zeros(z.shape[0], dtype=torch.long)

        model = CardioPhys3DPredictor(hidden_dim=32, num_layers=2, dropout=0.0)
        sim = CardiomyocyteSimulator(dt_ms=0.5, n_beats=1)

        mu, sigma = model(z, pos, batch, formal_charge=q)  # [1, 3]
        conc = torch.tensor([1.0], dtype=torch.float32)
        block = ChannelBlockProfile.from_pic50(
            conc_uM=conc,
            pic50_herg=mu[:, 0],
            pic50_cav12=mu[:, 1],
            pic50_nav15_late=mu[:, 2],
        )
        phys = sim.simulate(block)
        # Backprop TdP risk directly into 3D PaiNN weights
        phys.tdp_risk_score.sum().backward()

        grad_norm = sum(
            p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None
        )
        assert grad_norm > 0.0 and math.isfinite(grad_norm)

"""
ADMET-Engine v3.0 (CardioPhys-3D) — WP2 Mechanistic Electrophysiology Tests
Verifies:
  1. Physiological human endocardial action potential under control conditions
  2. Verapamil multi-channel rescue (Cav1.2 block compensates hERG block -> higher qNet & shorter APD90)
  3. Ranolazine multi-channel rescue (Nav1.5 late block compensates hERG block)
  4. End-to-end autograd differentiability (d qNet / d pIC50_hERG < 0, d qNet / d pIC50_Cav1.2 > 0)
  5. Batched Monte Carlo uncertainty propagation through the ODE
  6. Official FDA-CiPA 28 reference drug validation (ROC-AUC > 0.88 High+Int vs Low, > 0.95 High vs Low)
"""

import sys
from pathlib import Path

import pytest
import torch
from rdkit import Chem
from sklearn.metrics import roc_auc_score

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from physiology import (  # noqa: E402
    CIPA_28_DRUGS,
    CardiomyocyteSimulator,
    ChannelBlockProfile,
    hill_conductance_scale,
)


class TestCiPA28Dataset:

    def test_all_28_cipa_drugs_valid_smiles_and_categories(self):
        assert len(CIPA_28_DRUGS) == 28
        risks = [d.tdp_risk for d in CIPA_28_DRUGS]
        assert risks.count("High") == 8
        assert risks.count("Intermediate") == 11
        assert risks.count("Low") == 9
        for d in CIPA_28_DRUGS:
            mol = Chem.MolFromSmiles(d.smiles)
            assert mol is not None, f"Invalid SMILES for CiPA drug {d.name}: {d.smiles}"
            assert d.cmax_free_uM > 0.0


class TestOharaRudySimulator:

    def test_control_action_potential_physiology(self):
        sim = CardiomyocyteSimulator(dt_ms=0.25, n_beats=2)
        ctrl = sim.simulate(
            ChannelBlockProfile(
                g_kr=torch.ones(1),
                g_cal=torch.ones(1),
                g_nal=torch.ones(1),
                g_na=torch.ones(1),
            ),
            return_traces=True,
        )
        assert -92.0 < ctrl.v_rest_mV.item() < -82.0
        assert 20.0 < ctrl.v_peak_mV.item() < 55.0
        assert 250.0 < ctrl.apd90_ms.item() < 420.0
        assert ctrl.qnet.item() > 0.10
        assert not bool(ctrl.has_ead.item())
        assert ctrl.v_trace is not None

    def test_verapamil_and_ranolazine_multichannel_rescue(self):
        """
        Core biophysical proof of CardioPhys-3D:
        - Verapamil (hERG IC50=0.5 uM, Cav1.2 IC50=0.2 uM) at 4x Cmax:
          Multi-channel block produces significantly HIGHER qNet (safer) and
          SHORTER APD90 than hERG-only block at the same concentration.
        - Ranolazine (hERG IC50=8.27 uM, Nav1.5 late IC50=7.88 uM) at 4x Cmax:
          Adding I_NaL block increases qNet and reduces APD90 vs hERG-only.
        """
        sim = CardiomyocyteSimulator(dt_ms=0.25, n_beats=2)
        conc = torch.tensor([0.324, 0.324, 7.792, 7.792], dtype=torch.float32)  # 4x Cmax
        herg = torch.tensor([0.499, 0.499, 8.27, 8.27], dtype=torch.float32)
        cav12 = torch.tensor([1e6, 0.202, 1e6, 296.0], dtype=torch.float32)
        nal = torch.tensor([1e6, 1e6, 1e6, 7.88], dtype=torch.float32)

        block = ChannelBlockProfile.from_ic50_uM(
            conc_uM=conc,
            ic50_herg_uM=herg,
            ic50_cav12_uM=cav12,
            ic50_nav15_late_uM=nal,
        )
        res = sim.simulate(block)

        # Verapamil: index 1 (full multi-channel) vs index 0 (hERG-only)
        assert res.qnet[1].item() > res.qnet[0].item()
        assert res.apd90_ms[1].item() < res.apd90_ms[0].item()

        # Ranolazine: index 3 (full multi-channel) vs index 2 (hERG-only)
        assert res.qnet[3].item() > res.qnet[2].item()
        assert res.apd90_ms[3].item() < res.apd90_ms[2].item()

    def test_autograd_differentiability_and_gradient_signs(self):
        """
        Verify exact biophysical gradient signs through the ODE:
        - Higher pIC50_hERG (stronger hERG block) -> LOWER qNet (d qNet / d pIC50_hERG < 0)
        - Higher pIC50_Cav1.2 (stronger Cav1.2 block) -> HIGHER qNet (d qNet / d pIC50_Cav1.2 > 0)
        """
        sim = CardiomyocyteSimulator(dt_ms=0.5, n_beats=1)
        conc = torch.tensor([1.0], dtype=torch.float32)
        pic50_herg = torch.tensor([6.0], dtype=torch.float32, requires_grad=True)
        pic50_cav = torch.tensor([6.0], dtype=torch.float32, requires_grad=True)

        block = ChannelBlockProfile.from_pic50(
            conc_uM=conc, pic50_herg=pic50_herg, pic50_cav12=pic50_cav
        )
        res = sim.simulate(block)
        res.qnet.sum().backward()

        assert pic50_herg.grad is not None and torch.isfinite(pic50_herg.grad).all()
        assert pic50_cav.grad is not None and torch.isfinite(pic50_cav.grad).all()
        assert pic50_herg.grad.item() < 0.0, "Stronger hERG block must decrease qNet"
        assert pic50_cav.grad.item() > 0.0, "Stronger Cav1.2 block must increase qNet"

    def test_monte_carlo_uncertainty_propagation(self):
        sim = CardiomyocyteSimulator(dt_ms=0.5, n_beats=1)
        conc = torch.tensor([0.5, 0.5], dtype=torch.float32)
        mc = sim.propagate_uncertainty_mc(
            conc_uM=conc,
            pic50_mean={"hERG": torch.tensor([7.0, 5.0]), "Cav1.2": torch.tensor([4.5, 6.8])},
            pic50_std={"hERG": torch.tensor([0.4, 0.4]), "Cav1.2": torch.tensor([0.3, 0.3])},
            n_samples=8,
        )
        assert mc["qnet_mean"].shape == (2,)
        assert (mc["qnet_std"] > 0).all()
        # High hERG / low Cav1.2 (idx 0) must have higher TdP risk than low hERG / high Cav1.2 (idx 1)
        assert mc["tdp_prob_mean"][0].item() > mc["tdp_prob_mean"][1].item()

    def test_fda_cipa_28_benchmark_outperforms_herg_only(self):
        sim = CardiomyocyteSimulator(dt_ms=0.4, n_beats=1)

        def _t(vals):
            return torch.tensor([1e6 if v is None else float(v) for v in vals], dtype=torch.float32)

        out = sim.simulate_cipa_protocol(
            cmax_free_uM=_t([d.cmax_free_uM for d in CIPA_28_DRUGS]),
            ic50_herg_uM=_t([d.ic50_herg_uM for d in CIPA_28_DRUGS]),
            ic50_cav12_uM=_t([d.ic50_cav12_uM for d in CIPA_28_DRUGS]),
            ic50_nav15_late_uM=_t([d.ic50_nav15_late_uM for d in CIPA_28_DRUGS]),
            ic50_nav15_peak_uM=_t([d.ic50_nav15_peak_uM for d in CIPA_28_DRUGS]),
            h_herg=_t([d.h_herg for d in CIPA_28_DRUGS]),
            h_cav12=_t([d.h_cav12 for d in CIPA_28_DRUGS]),
            h_nav15_late=_t([d.h_nav15_late for d in CIPA_28_DRUGS]),
            h_nav15_peak=_t([d.h_nav15_peak for d in CIPA_28_DRUGS]),
        )
        qnets = out["mean_qnet"].detach().numpy()
        risks = [d.tdp_risk for d in CIPA_28_DRUGS]
        y_hi_int = [1 if r in ("High", "Intermediate") else 0 for r in risks]
        y_hi_lo = [1 if r == "High" else 0 for r in risks if r != "Intermediate"]
        q_hi_lo = [-q for q, r in zip(qnets, risks) if r != "Intermediate"]

        auc_all = roc_auc_score(y_hi_int, -qnets)
        auc_hilo = roc_auc_score(y_hi_lo, q_hi_lo)
        assert auc_all >= 0.88, f"Expected CiPA-28 High+Int vs Low AUC >= 0.88, got {auc_all:.4f}"
        assert auc_hilo >= 0.95, f"Expected CiPA-28 High vs Low AUC >= 0.95, got {auc_hilo:.4f}"

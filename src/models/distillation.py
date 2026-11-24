"""
ADMET-Engine v3.0 (CardioPhys-3D) — WP4: Knowledge Distillation & End-to-End Coupling
=====================================================================================
Compresses heavy 3D SE(3)-equivariant / multi-teacher potency predictors into a
sub-2ms edge student network (`CardioPhysStudentNet`) via closed-form Gaussian
KL-divergence distillation + censored Tobit NLL, and couples predicted
(hERG, Cav1.2, Nav1.5) potency distributions directly into the WP2
differentiable O'Hara-Rudy cardiomyocyte ODE simulator.

Loss function:
  L_distill = (1 - lam) * L_Tobit(mu_S, sigma_S; y, c)
            + lam * sum_c KL( N(mu_T,c, sigma_T,c^2) || N(mu_S,c, sigma_S,c^2) )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch import Tensor

from cheminformatics import compute_morgan_numpy, compute_rdkit_ml_descriptors
from data_pipeline import standardize_molecule
from models.equivariant_3d import CHANNELS_V3
from physiology.ohara_rudy import CardiomyocyteSimulator, ChannelBlockProfile
from training.tobit_loss import tobit_censored_nll


# ---------------------------------------------------------------------------
# Closed-Form Gaussian KL Distillation Loss
# ---------------------------------------------------------------------------

def gaussian_kl_divergence(
    mu_teacher: Tensor,
    sigma_teacher: Tensor,
    mu_student: Tensor,
    sigma_student: Tensor,
    sigma_floor: float = 0.05,
) -> Tensor:
    """
    Analytical KL divergence KL( N(mu_T, sig_T^2) || N(mu_S, sig_S^2) )
    averaged over batch and channels. Transfers both mean potency AND
    calibrated uncertainty from the 3D/foundation teacher to the fast student.
    """
    sig_t = torch.clamp(sigma_teacher.detach(), min=sigma_floor)
    sig_s = torch.clamp(sigma_student, min=sigma_floor)
    mu_t = mu_teacher.detach()

    kl = (
        torch.log(sig_s / sig_t)
        + (sig_t ** 2 + (mu_t - mu_student) ** 2) / (2.0 * (sig_s ** 2))
        - 0.5
    )
    return kl.mean()


def distillation_tobit_loss(
    mu_student: Tensor,
    sigma_student: Tensor,
    mu_teacher: Tensor,
    sigma_teacher: Tensor,
    targets: Optional[Tensor] = None,
    censor: Optional[Tensor] = None,
    distill_weight: float = 0.5,
) -> Tensor:
    """
    Hybrid Knowledge Distillation objective combining censored Tobit NLL on
    ground-truth ChEMBL labels with Gaussian KL matching against the teacher.
    Works seamlessly on unlabeled virtual libraries (targets=None -> pure KL).
    """
    kl = gaussian_kl_divergence(mu_teacher, sigma_teacher, mu_student, sigma_student)
    if targets is None or censor is None:
        return kl
    tobit = tobit_censored_nll(mu_student, sigma_student, targets, censor)
    return (1.0 - distill_weight) * tobit + distill_weight * kl


# ---------------------------------------------------------------------------
# Fast Edge Student Network (< 2 ms CPU inference)
# ---------------------------------------------------------------------------

class CardioPhysStudentNet(nn.Module):
    """
    Lightweight residual student MLP operating on 2048-bit ECFP4 + 20 RDKit
    physicochemical descriptors (2,068 dims -> 3 cardiac ion channels).
    Designed as an ultra-fast (<2 ms) distilled oracle for the WP5 Closed-Loop
    Molecular Repair Agent.
    """

    def __init__(
        self,
        in_dim: int = 2068,
        hidden_dim: int = 256,
        num_channels: int = 3,
        dropout: float = 0.20,
        min_sigma: float = 0.10,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.min_sigma = min_sigma

        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.out_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * num_channels),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.in_proj(x)
        h = h + self.res_block(h)
        raw = self.out_head(h).view(-1, self.num_channels, 2)
        mu = raw[:, :, 0] + 5.5
        sigma = F.softplus(raw[:, :, 1]) + self.min_sigma
        return mu, sigma


# ---------------------------------------------------------------------------
# End-to-End SMILES -> Multi-Channel pIC50 -> ORd-CiPA ODE Pipeline
# ---------------------------------------------------------------------------

@dataclass
class CardioPhysPrediction:
    smiles: str
    canonical_smiles: str
    pic50_mean: dict[str, float]
    pic50_std: dict[str, float]
    qnet_uC_uF: float
    apd90_ms: float
    tdp_risk_score: float
    has_ead: bool


class CardioPhysEndToEndPipeline:
    """
    Unified WP4 inference pipeline connecting the distilled student potency
    model directly to the WP2 differentiable O'Hara-Rudy cardiomyocyte ODE.
    """

    def __init__(
        self,
        student: CardioPhysStudentNet,
        simulator: Optional[CardiomyocyteSimulator] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.student = student.to(self.device).eval()
        self.simulator = (simulator or CardiomyocyteSimulator(dt_ms=0.3, n_beats=1)).to(self.device)

    def featurize_mols(self, mols: list[Chem.Mol]) -> Tensor:
        n = len(mols)
        mat = np.zeros((n, 2068), dtype=np.float32)
        for i, mol in enumerate(mols):
            mat[i, :2048] = compute_morgan_numpy(mol)
            mat[i, 2048:] = compute_rdkit_ml_descriptors(mol)
        return torch.from_numpy(mat).to(self.device)

    @torch.no_grad()
    def predict_smiles(
        self,
        smiles_list: list[str],
        conc_uM: float | list[float] = 1.0,
    ) -> list[Optional[CardioPhysPrediction]]:
        """Run end-to-end SMILES -> 3-channel pIC50 -> ORd-CiPA ODE -> qNet & P(TdP)."""
        valid_mols: list[Chem.Mol] = []
        valid_idx: list[int] = []
        can_smiles: list[str] = []

        for idx, smi in enumerate(smiles_list):
            try:
                mol = standardize_molecule(smi)
                if mol is not None and mol.GetNumAtoms() > 0:
                    valid_mols.append(mol)
                    valid_idx.append(idx)
                    can_smiles.append(Chem.MolToSmiles(mol, canonical=True))
            except Exception:
                continue

        outputs: list[Optional[CardioPhysPrediction]] = [None] * len(smiles_list)
        if not valid_mols:
            return outputs

        x = self.featurize_mols(valid_mols)
        mu, sigma = self.student(x)  # [B_v, 3]

        if isinstance(conc_uM, (int, float)):
            conc_t = torch.full((len(valid_mols),), float(conc_uM), device=self.device)
        else:
            conc_t = torch.tensor([conc_uM[i] for i in valid_idx], dtype=torch.float32, device=self.device)

        block = ChannelBlockProfile.from_pic50(
            conc_uM=conc_t,
            pic50_herg=mu[:, 0],
            pic50_cav12=mu[:, 1],
            pic50_nav15_late=mu[:, 2],
        )
        phys = self.simulator.simulate(block, return_traces=False)

        mu_np = mu.cpu().numpy()
        sig_np = sigma.cpu().numpy()
        qnet_np = phys.qnet.cpu().numpy()
        apd_np = phys.apd90_ms.cpu().numpy()
        tdp_np = phys.tdp_risk_score.cpu().numpy()
        ead_np = phys.has_ead.cpu().numpy()

        for j, orig_i in enumerate(valid_idx):
            outputs[orig_i] = CardioPhysPrediction(
                smiles=smiles_list[orig_i],
                canonical_smiles=can_smiles[j],
                pic50_mean={ch: round(float(mu_np[j, c]), 4) for c, ch in enumerate(CHANNELS_V3)},
                pic50_std={ch: round(float(sig_np[j, c]), 4) for c, ch in enumerate(CHANNELS_V3)},
                qnet_uC_uF=round(float(qnet_np[j]), 5),
                apd90_ms=round(float(apd_np[j]), 1),
                tdp_risk_score=round(float(tdp_np[j]), 4),
                has_ead=bool(ead_np[j]),
            )

        return outputs

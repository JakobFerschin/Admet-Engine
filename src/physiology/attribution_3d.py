"""
ADMET-Engine v3.0 (CardioPhys-3D) — End-to-End 3D-to-Organ Differentiable Attribution
=====================================================================================
Propagates exact PyTorch autograd sensitivities backward from organ-level human
ventricular electrophysiology biomarkers (`qNet`, `tdp_risk_score`) through the
differentiable O'Hara-Rudy (ORd-CiPA) cardiomyocyte ODE and Hill-equation
multi-channel block coupling directly onto 3D atomic coordinates `pos_i` of the
pH-7.4-protonated ligand conformer:

    3D Conformer (z, pos, q_pH7.4)
        --[CardioPhys3DPredictor (SE(3)-Equivariant PaiNN)]-->
    Multi-Channel Potencies mu = (pIC50_hERG, pIC50_Cav1.2, pIC50_Nav1.5)
        --[ChannelBlockProfile (Differentiable Hill Equation)]-->
    Fractional Conductances (g_Kr, g_CaL, g_NaL, g_Na)
        --[CardiomyocyteSimulator (Rush-Larsen ORd-CiPA ODE)]-->
    Organ Biomarkers (qNet, P(TdP))

Exact Chain-Rule Decomposition:
    d(qNet) / d(pos_i) = sum_{c in {hERG, Cav1.2, Nav1.5}}
                           [d(qNet) / d(mu_c)] * [d(mu_c) / d(pos_i)]

Mathematical Guarantee:
    Because `CardioPhys3DPredictor` is strictly SE(3)-invariant, the end-to-end
    organ-to-atom 3D gradient vectors `d(qNet)/d(pos_i)` and `d(TdP)/d(pos_i)`
    are strictly SO(3)-equivariant (`g'_i = R g_i`), and their per-atom Euclidean
    sensitivity norms `||g_i||_2` are strictly SE(3)-invariant (invariant to any
    3D rigid rotation R in SO(3) and translation t in R^3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from rdkit import Chem
from torch import Tensor

from data.conformer_3d import generate_3d_ensemble
from models.equivariant_3d import CHANNELS_V3, CardioPhys3DPredictor
from physiology.ohara_rudy import CardiomyocyteSimulator, ChannelBlockProfile


def _atomic_num_to_symbol(z: int) -> str:
    pt = Chem.GetPeriodicTable()
    try:
        return pt.GetElementSymbol(int(z))
    except Exception:
        return f"Z{int(z)}"


@dataclass
class AtomAttribution:
    """Per-atom 3D-to-organ proarrhythmic sensitivity summary."""

    atom_index: int
    atomic_num: int
    element: str
    formal_charge: int
    qnet_grad_norm: float
    tdp_grad_norm: float
    qnet_grad_vec: tuple[float, float, float]
    tdp_grad_vec: tuple[float, float, float]
    channel_qnet_grad_norms: dict[str, float]
    dominant_channel: str


@dataclass
class OrganAttributionResult:
    """Complete end-to-end 3D-to-organ differentiable attribution report."""

    smiles: Optional[str]
    smiles_ph74: Optional[str]
    conc_uM: float
    pic50_mean: dict[str, float]
    pic50_std: dict[str, float]
    qnet_uC_uF: float
    apd90_ms: float
    tdp_risk_score: float
    has_ead: bool
    qnet_grad_pos: np.ndarray                       # [N, 3] d(qNet)/d(pos_i)
    tdp_grad_pos: np.ndarray                        # [N, 3] d(tdp_risk_score)/d(pos_i)
    qnet_grad_norms: np.ndarray                     # [N] ||d(qNet)/d(pos_i)||_2
    tdp_grad_norms: np.ndarray                      # [N] ||d(tdp_risk_score)/d(pos_i)||_2
    channel_qnet_jacobian: dict[str, float]         # d(qNet)/d(pIC50_c)
    channel_tdp_jacobian: dict[str, float]          # d(tdp_risk_score)/d(pIC50_c)
    channel_qnet_contribution_fraction: dict[str, float]
    channel_atom_qnet_norms: dict[str, np.ndarray]  # c -> [N] ||(dqNet/dmu_c)*(dmu_c/dpos_i)||_2
    ranked_hotspots: list[AtomAttribution]          # descending by tdp_grad_norm


def compute_3d_organ_attribution(
    predictor: CardioPhys3DPredictor,
    simulator: CardiomyocyteSimulator,
    z: Tensor,
    pos: Tensor,
    formal_charge: Optional[Tensor] = None,
    conc_uM: float = 1.0,
    smiles: Optional[str] = None,
    smiles_ph74: Optional[str] = None,
    top_k: Optional[int] = None,
) -> OrganAttributionResult:
    """
    Run end-to-end differentiable 3D-to-organ attribution for a single 3D conformer.

    Args:
        predictor:     SE(3)-equivariant `CardioPhys3DPredictor`.
        simulator:     Differentiable `CardiomyocyteSimulator` (ORd-CiPA ODE).
        z:             Atomic numbers `[N]` (int64 tensor).
        pos:           3D atomic coordinates `[N, 3]` (float32 tensor, Angstroms).
        formal_charge: Optional pH 7.4 formal charges `[N]` (int8/int64 tensor).
        conc_uM:       Free therapeutic plasma concentration in uM.
        smiles:        Optional original SMILES string for provenance.
        smiles_ph74:   Optional pH 7.4 canonical SMILES string.
        top_k:         Optional cap on returned `ranked_hotspots` (None = all atoms).

    Returns:
        `OrganAttributionResult` with exact per-atom 3D gradient vectors & norms,
        per-channel organ Jacobians, channel contribution fractions, and ranked hotspots.
    """
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Expected pos of shape [N, 3], got {tuple(pos.shape)}")
    n_atoms = pos.shape[0]
    if z.shape[0] != n_atoms:
        raise ValueError(f"z length ({z.shape[0]}) must match pos length ({n_atoms})")

    device = next(predictor.parameters()).device
    dtype = pos.dtype if pos.is_floating_point() else torch.float32

    was_training = predictor.training
    predictor.eval()

    try:
        z_t = z.to(device=device, dtype=torch.long)
        q_t = (
            formal_charge.to(device=device, dtype=torch.long)
            if formal_charge is not None
            else torch.zeros(n_atoms, device=device, dtype=torch.long)
        )
        batch_t = torch.zeros(n_atoms, device=device, dtype=torch.long)
        pos_req = pos.detach().to(device=device, dtype=dtype).clone().requires_grad_(True)
        conc_t = torch.tensor([float(conc_uM)], device=device, dtype=dtype)

        # 1. SE(3)-equivariant 3D ligand -> multi-channel pIC50 distributions
        mu, sigma = predictor(z_t, pos_req, batch_t, formal_charge=q_t)  # [1, 3]

        # 2. Differentiable Hill equation -> ORd-CiPA ventricular action potential ODE
        block = ChannelBlockProfile.from_pic50(
            conc_uM=conc_t,
            pic50_herg=mu[:, 0],
            pic50_cav12=mu[:, 1],
            pic50_nav15_late=mu[:, 2],
        )
        sim_res = simulator.simulate(block, return_traces=False)
        qnet_scalar = sim_res.qnet[0]
        tdp_scalar = sim_res.tdp_risk_score[0]

        # 3. Exact end-to-end 3D gradients: d(qNet)/d(pos) and d(TdP)/d(pos)
        grad_qnet_pos = torch.autograd.grad(
            qnet_scalar, pos_req, retain_graph=True, create_graph=False
        )[0]
        grad_tdp_pos = torch.autograd.grad(
            tdp_scalar, pos_req, retain_graph=True, create_graph=False
        )[0]

        # 4. Exact organ-to-channel Jacobians: d(qNet)/d(mu_c) and d(TdP)/d(mu_c)
        dqnet_dmu = torch.autograd.grad(
            qnet_scalar, mu, retain_graph=True, create_graph=False
        )[0][0]  # [3]
        dtdp_dmu = torch.autograd.grad(
            tdp_scalar, mu, retain_graph=True, create_graph=False
        )[0][0]  # [3]

        # 5. Per-channel chain-rule decomposition: (d(qNet)/d(mu_c)) * (d(mu_c)/d(pos_i))
        channel_atom_qnet_norms: dict[str, np.ndarray] = {}
        channel_total_l2: dict[str, float] = {}
        num_channels = min(len(CHANNELS_V3), mu.shape[1])

        for c in range(num_channels):
            ch_name = CHANNELS_V3[c]
            dmu_c_dpos = torch.autograd.grad(
                mu[0, c],
                pos_req,
                retain_graph=(c < num_channels - 1),
                create_graph=False,
            )[0]  # [N, 3]
            contrib_vec = dqnet_dmu[c] * dmu_c_dpos  # [N, 3]
            contrib_norms = torch.linalg.norm(contrib_vec, dim=-1).detach().cpu().numpy()
            channel_atom_qnet_norms[ch_name] = contrib_norms.astype(np.float64)
            channel_total_l2[ch_name] = float(np.sum(contrib_norms))

        total_channel_l2 = sum(channel_total_l2.values())
        if total_channel_l2 > 1e-12:
            channel_contrib_frac = {
                ch: float(val / total_channel_l2) for ch, val in channel_total_l2.items()
            }
        else:
            channel_contrib_frac = {
                ch: 1.0 / max(num_channels, 1) for ch in channel_total_l2
            }

        qnet_grad_np = grad_qnet_pos.detach().cpu().numpy().astype(np.float64)
        tdp_grad_np = grad_tdp_pos.detach().cpu().numpy().astype(np.float64)
        qnet_norms_np = np.linalg.norm(qnet_grad_np, axis=-1)
        tdp_norms_np = np.linalg.norm(tdp_grad_np, axis=-1)

        z_np = z_t.detach().cpu().numpy()
        q_np = q_t.detach().cpu().numpy()

        hotspots: list[AtomAttribution] = []
        for i in range(n_atoms):
            ch_norms_i = {
                ch: float(channel_atom_qnet_norms[ch][i])
                for ch in channel_atom_qnet_norms
            }
            dom_ch = max(ch_norms_i, key=ch_norms_i.get) if ch_norms_i else "hERG"
            hotspots.append(
                AtomAttribution(
                    atom_index=i,
                    atomic_num=int(z_np[i]),
                    element=_atomic_num_to_symbol(int(z_np[i])),
                    formal_charge=int(q_np[i]),
                    qnet_grad_norm=float(qnet_norms_np[i]),
                    tdp_grad_norm=float(tdp_norms_np[i]),
                    qnet_grad_vec=(
                        float(qnet_grad_np[i, 0]),
                        float(qnet_grad_np[i, 1]),
                        float(qnet_grad_np[i, 2]),
                    ),
                    tdp_grad_vec=(
                        float(tdp_grad_np[i, 0]),
                        float(tdp_grad_np[i, 1]),
                        float(tdp_grad_np[i, 2]),
                    ),
                    channel_qnet_grad_norms=ch_norms_i,
                    dominant_channel=dom_ch,
                )
            )

        hotspots.sort(key=lambda a: a.tdp_grad_norm, reverse=True)
        if top_k is not None:
            hotspots = hotspots[:top_k]

        mu_np = mu[0].detach().cpu().numpy()
        sigma_np = sigma[0].detach().cpu().numpy()
        dqnet_np = dqnet_dmu.detach().cpu().numpy()
        dtdp_np = dtdp_dmu.detach().cpu().numpy()

        return OrganAttributionResult(
            smiles=smiles,
            smiles_ph74=smiles_ph74,
            conc_uM=float(conc_uM),
            pic50_mean={CHANNELS_V3[c]: float(mu_np[c]) for c in range(num_channels)},
            pic50_std={CHANNELS_V3[c]: float(sigma_np[c]) for c in range(num_channels)},
            qnet_uC_uF=float(qnet_scalar.detach().item()),
            apd90_ms=float(sim_res.apd90_ms[0].detach().item()),
            tdp_risk_score=float(tdp_scalar.detach().item()),
            has_ead=bool(sim_res.has_ead[0].detach().item()),
            qnet_grad_pos=qnet_grad_np,
            tdp_grad_pos=tdp_grad_np,
            qnet_grad_norms=qnet_norms_np,
            tdp_grad_norms=tdp_norms_np,
            channel_qnet_jacobian={
                CHANNELS_V3[c]: float(dqnet_np[c]) for c in range(num_channels)
            },
            channel_tdp_jacobian={
                CHANNELS_V3[c]: float(dtdp_np[c]) for c in range(num_channels)
            },
            channel_qnet_contribution_fraction=channel_contrib_frac,
            channel_atom_qnet_norms=channel_atom_qnet_norms,
            ranked_hotspots=hotspots,
        )
    finally:
        if was_training:
            predictor.train()


def attribute_smiles_3d_to_organ(
    smiles: str,
    predictor: CardioPhys3DPredictor,
    simulator: Optional[CardiomyocyteSimulator] = None,
    conc_uM: float = 1.0,
    num_confs: int = 5,
    top_k: Optional[int] = 10,
    seed: int = 42,
) -> OrganAttributionResult:
    """
    End-to-end helper: SMILES -> lowest-energy pH 7.4 3D conformer ->
    CardioPhys3DPredictor -> ORd-CiPA ODE backward pass -> ranked proarrhythmic
    3D hotspot atoms.
    """
    ens = generate_3d_ensemble(
        smiles=smiles,
        num_confs=num_confs,
        protonate=True,
        include_hydrogens=False,
        seed=seed,
    )
    if ens is None or ens.positions.shape[0] == 0:
        raise ValueError(f"Failed to generate pH 7.4 3D conformer for SMILES: {smiles}")

    sim = simulator or CardiomyocyteSimulator(dt_ms=0.5, n_beats=1)
    # Conformer index 0 is the lowest-energy MMFF94s/UFF conformer
    z_t = torch.from_numpy(ens.atomic_numbers)
    pos_t = torch.from_numpy(ens.positions[0])
    q_t = torch.from_numpy(ens.formal_charges)

    return compute_3d_organ_attribution(
        predictor=predictor,
        simulator=sim,
        z=z_t,
        pos=pos_t,
        formal_charge=q_t,
        conc_uM=conc_uM,
        smiles=smiles,
        smiles_ph74=ens.canonical_smiles_ph74,
        top_k=top_k,
    )

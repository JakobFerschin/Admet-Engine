"""
ADMET-Engine v3.0 (CardioPhys-3D) — WP1: pH 7.4 Microspecies & 3D Conformer Engine
==================================================================================
Generates physiologically protonated (pH 7.4) 3D conformer ensembles for
SE(3)-equivariant protein-ligand modeling (WP3):
  1. Physiological pH 7.4 ionization state assignment:
       - Uses `dimorphite_dl` if installed, else applies deterministic RDKit
         SMARTS protonation/deprotonation rules grounded in medicinal chemistry
         pKa ranges (protonates basic aliphatic amines, piperidines, piperazines,
         amidines, and guanidines essential for hERG Tyr652/Phe656 cation-pi
         binding; deprotonates carboxylic/sulfonic acids and tetrazoles).
  2. ETKDGv3 3D conformer embedding with RMSD pruning + MMFF94(s) force-field
     minimization (with UFF fallback for non-MMFF elements).
  3. Boltzmann statistical weights at human body temperature (T = 310.15 K):
         w_k = exp(-(E_k - E_min) / (R * T)) / Z
  4. Compact PyTorch geometric tensor serialization for fast Colab GPU loading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# Gas constant in kcal / (mol * K) at 37 deg C (310.15 K)
R_KCAL_MOL_K = 1.98720425864083e-3
PHYSIOLOGICAL_TEMP_K = 310.15
RT_PHYSIOLOGICAL = R_KCAL_MOL_K * PHYSIOLOGICAL_TEMP_K  # ~0.6163 kcal/mol

# ---------------------------------------------------------------------------
# Deterministic pH 7.4 Physiological Protonation Rules (RDKit SMARTS)
# ---------------------------------------------------------------------------
# Ordered reactions capturing dominant microspecies at pH 7.4:
#   - Basic aliphatic amines (pKa ~9-11) -> protonated [NH+]/[NH2+]/[NH3+]
#   - Guanidines & amidines (pKa ~11-13) -> protonated
#   - Carboxylic acids (pKa ~3-5) -> carboxylate [O-]
#   - Sulfonic & phosphonic acids (pKa < 3) -> deprotonated [O-]
#   - 5-substituted tetrazoles (pKa ~4.9) -> tetrazolate [n-]
# Explicitly excludes amides, sulfonamides, anilines, nitrogens adjacent to C=O/S=O.

_PH74_REACTIONS_SMARTS: list[tuple[str, str]] = [
    # Carboxylic acid -> carboxylate
    ("carboxylic_acid", "[C:1](=[O:2])[OH:3]>>[C:1](=[O:2])[O-:3]"),
    # Sulfonic acid -> sulfonate
    ("sulfonic_acid", "[S:1](=[O:2])(=[O:3])[OH:4]>>[S:1](=[O:2])(=[O:3])[O-:4]"),
    # Tetrazole -> tetrazolate
    ("tetrazole", "[c:1]1[nH:2][n:3][n:4][n:5]1>>[c:1]1[n-:2][n:3][n:4][n:5]1"),
    # Guanidine (=NH) -> guanidinium
    ("guanidine", "[N:1]=[C:2]([N:3])[N:4]>>[NH+:1]=[C:2]([N:3])[N:4]"),
    # Amidine (=NH) -> amidinium
    ("amidine", "[NH:1]=[C:2]([#6:3])[N:4]>>[NH2+:1]=[C:2]([#6:3])[N:4]"),
    # Tertiary aliphatic amine (not amide, sulfonamide, aniline, or vinylogous) -> [NH+]
    (
        "tertiary_amine",
        "[NX3;H0;+0;!$(NC=O);!$(NS=O);!$(N-a);!$(N=*);!$(NC=N):1]>>[NH+:1]",
    ),
    # Secondary aliphatic amine -> [NH2+]
    (
        "secondary_amine",
        "[NX3;H1;+0;!$(NC=O);!$(NS=O);!$(N-a);!$(N=*);!$(NC=N):1]>>[NH2+:1]",
    ),
    # Primary aliphatic amine -> [NH3+]
    (
        "primary_amine",
        "[NX3;H2;+0;!$(NC=O);!$(NS=O);!$(N-a);!$(N=*);!$(NC=N):1]>>[NH3+:1]",
    ),
]

_COMPILED_PH74_RXNS = [
    (name, AllChem.ReactionFromSmarts(smarts))
    for name, smarts in _PH74_REACTIONS_SMARTS
]


def protonate_at_ph74(mol: Chem.Mol) -> Chem.Mol:
    """
    Generate the dominant physiological microspecies at pH 7.4.
    Uses `dimorphite_dl` if available in the environment, otherwise applies
    the deterministic RDKit SMARTS pH 7.4 transformation suite.
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return mol

    try:
        from dimorphite_dl import protonate_smiles  # type: ignore

        smi = Chem.MolToSmiles(mol, canonical=True)
        variants = protonate_smiles(smi, ph_min=7.2, ph_max=7.6, max_variants=1)
        if variants:
            out = Chem.MolFromSmiles(variants[0])
            if out is not None:
                return out
    except Exception:
        pass

    curr = Chem.Mol(mol)
    for _, rxn in _COMPILED_PH74_RXNS:
        # Apply iteratively until no more sites match (capped at 6 ionizable sites)
        for _ in range(6):
            products = rxn.RunReactants((curr,))
            if not products:
                break
            cand = products[0][0]
            try:
                Chem.SanitizeMol(cand)
                curr = cand
            except Exception:
                break
    return curr


# ---------------------------------------------------------------------------
# 3D Conformer Ensemble + MMFF94(s) Minimization + Boltzmann Weighting
# ---------------------------------------------------------------------------

@dataclass
class ConformerEnsemble3D:
    """Boltzmann-weighted 3D conformer ensemble ready for SE(3)-equivariant GNNs."""
    canonical_smiles_ph74: str
    atomic_numbers: np.ndarray      # [N_atoms] int64
    formal_charges: np.ndarray      # [N_atoms] int8
    positions: np.ndarray           # [K, N_atoms, 3] float32 (Angstroms)
    energies_kcal: np.ndarray       # [K] float32 (MMFF94/UFF energies)
    boltzmann_weights: np.ndarray   # [K] float32 (sum to 1.0 at 310.15 K)
    net_charge_ph74: int

    def to_torch_dict(self) -> dict[str, Any]:
        """Serialize into lightweight PyTorch tensors for .pt dataset caching."""
        return {
            "smiles_ph74": self.canonical_smiles_ph74,
            "z": torch.from_numpy(self.atomic_numbers),
            "formal_charge": torch.from_numpy(self.formal_charges),
            "pos": torch.from_numpy(self.positions),
            "energies_kcal": torch.from_numpy(self.energies_kcal),
            "boltzmann_weights": torch.from_numpy(self.boltzmann_weights),
            "net_charge_ph74": self.net_charge_ph74,
        }


def compute_boltzmann_weights(
    energies_kcal: np.ndarray,
    temperature_k: float = PHYSIOLOGICAL_TEMP_K,
) -> np.ndarray:
    """Compute numerically stable Boltzmann probabilities at `temperature_k`."""
    if len(energies_kcal) == 0:
        return np.array([], dtype=np.float32)
    rt = R_KCAL_MOL_K * temperature_k
    rel_e = energies_kcal - np.min(energies_kcal)
    logits = np.clip(-rel_e / rt, -60.0, 0.0)
    weights = np.exp(logits)
    weights /= np.sum(weights)
    return weights.astype(np.float32)


def generate_3d_ensemble(
    smiles: str,
    num_confs: int = 5,
    prune_rms_thresh: float = 0.5,
    max_iters: int = 200,
    protonate: bool = True,
    include_hydrogens: bool = False,
    seed: int = 42,
) -> Optional[ConformerEnsemble3D]:
    """
    Generate a pH-7.4-protonated, MMFF94(s)-minimized, Boltzmann-weighted
    3D conformer ensemble from a SMILES string.

    Args:
        smiles:            Input SMILES string.
        num_confs:         Target number of 3D conformers (default 5; use 10 on Colab).
        prune_rms_thresh:  Heavy-atom RMSD pruning threshold in Angstroms.
        max_iters:         Maximum MMFF94/UFF minimization steps per conformer.
        protonate:         If True, assign physiological pH 7.4 microspecies first.
        include_hydrogens: If False, strip hydrogens after 3D minimization to keep
                           heavy-atom graphs compact for SE(3) message passing.
        seed:              Deterministic ETKDGv3 RNG seed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    if protonate:
        mol = protonate_at_ph74(mol)

    smi_ph74 = Chem.MolToSmiles(mol, canonical=True)
    net_charge = int(Chem.GetFormalCharge(mol))

    # Add explicit hydrogens for accurate 3D geometry & MMFF94 energetics
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = prune_rms_thresh
    params.useSmallRingTorsions = True
    params.numThreads = 0  # use available cores when called sequentially

    conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=num_confs, params=params))
    if not conf_ids:
        # Fallback with random coordinates for macrocycles / strained cages
        params.useRandomCoords = True
        conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=max(1, num_confs // 2), params=params))
        if not conf_ids:
            return None

    # Force-field energy minimization: prefer MMFF94s, fall back to UFF
    energies: list[float] = []
    if AllChem.MMFFHasAllMoleculeParams(mol_h):
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
        for cid in conf_ids:
            ff = AllChem.MMFFGetMoleculeForceField(mol_h, mmff_props, confId=cid)
            if ff is not None:
                ff.Minimize(maxIts=max_iters)
                energies.append(float(ff.CalcEnergy()))
            else:
                energies.append(0.0)
    else:
        for cid in conf_ids:
            ff = AllChem.UFFGetMoleculeForceField(mol_h, confId=cid)
            if ff is not None:
                ff.Minimize(maxIts=max_iters)
                energies.append(float(ff.CalcEnergy()))
            else:
                energies.append(0.0)

    out_mol = mol_h if include_hydrogens else Chem.RemoveHs(mol_h)
    order = np.argsort(energies)
    sorted_cids = [conf_ids[i] for i in order]
    sorted_energies = np.array([energies[i] for i in order], dtype=np.float32)

    # Extract 3D coordinate array [K, N, 3] centered at origin
    n_atoms = out_mol.GetNumAtoms()
    coords = np.zeros((len(sorted_cids), n_atoms, 3), dtype=np.float32)
    for k, cid in enumerate(sorted_cids):
        conf = out_mol.GetConformer(cid)
        pos_k = np.array(conf.GetPositions(), dtype=np.float32)
        coords[k] = pos_k - pos_k.mean(axis=0, keepdims=True)

    atomic_nums = np.array([a.GetAtomicNum() for a in out_mol.GetAtoms()], dtype=np.int64)
    formal_charges = np.array([a.GetFormalCharge() for a in out_mol.GetAtoms()], dtype=np.int8)
    boltz_w = compute_boltzmann_weights(sorted_energies)

    return ConformerEnsemble3D(
        canonical_smiles_ph74=smi_ph74,
        atomic_numbers=atomic_nums,
        formal_charges=formal_charges,
        positions=coords,
        energies_kcal=sorted_energies,
        boltzmann_weights=boltz_w,
        net_charge_ph74=net_charge,
    )

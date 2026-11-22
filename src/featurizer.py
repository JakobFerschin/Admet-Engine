"""
ADMET-Engine — Molecular Featurizer
Phase 4, Month 7–9: Backend & API Infrastructure

Converts RDKit Mol objects into the PyTorch Geometric graph tensors
required by the DMPNN model.

Atom features (9 dims, Section 4.2):
  [atomic_num, degree, formal_charge, hybridization,
   chirality, aromaticity, num_hydrogens, mass, is_in_ring]

Bond features (4 dims, Section 4.2):
  [bond_type, conjugated, in_ring, stereo]

Directed edges: both (u→v) and (v→u) are included for every bond,
as required by DMPNN's directed message-passing scheme.
"""

from typing import Optional, Tuple

import torch
from rdkit import Chem
from torch import Tensor
from torch_geometric.data import Data

from data_pipeline import standardize_molecule

# ---------------------------------------------------------------------------
# Categorical encoding tables
# ---------------------------------------------------------------------------

_HYBRIDIZATION_MAP = {
    Chem.rdchem.HybridizationType.SP:    1,
    Chem.rdchem.HybridizationType.SP2:   2,
    Chem.rdchem.HybridizationType.SP3:   3,
    Chem.rdchem.HybridizationType.SP3D:  4,
    Chem.rdchem.HybridizationType.SP3D2: 5,
}
_HYBRIDIZATION_DEFAULT = 0

_CHIRALITY_MAP = {
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:  1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    Chem.rdchem.ChiralType.CHI_OTHER:           3,
}
_CHIRALITY_DEFAULT = 0

_BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE:   1,
    Chem.rdchem.BondType.DOUBLE:   2,
    Chem.rdchem.BondType.TRIPLE:   3,
    Chem.rdchem.BondType.AROMATIC: 4,
}
_BOND_TYPE_DEFAULT = 0

_BOND_STEREO_MAP = {
    Chem.rdchem.BondStereo.STEREOANY:  1,
    Chem.rdchem.BondStereo.STEREOZ:    2,
    Chem.rdchem.BondStereo.STEREOE:    3,
}
_BOND_STEREO_DEFAULT = 0

NODE_DIM = 9
EDGE_DIM = 4

# ---------------------------------------------------------------------------
# Chemprop-grade One-Hot Featurizer (Yang et al., JCIM 2019)
# ---------------------------------------------------------------------------

_ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "B", "Si", "Se"]
_DEGREES = [0, 1, 2, 3, 4, 5]
_FORMAL_CHARGES = [-2, -1, 0, 1, 2]
_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
_CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
_NUM_HS = [0, 1, 2, 3, 4]
_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
_BOND_STEREOS = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]


def _one_hot_with_other(value, choices: list) -> list[float]:
    vec = [0.0] * (len(choices) + 1)
    try:
        vec[choices.index(value)] = 1.0
    except ValueError:
        vec[-1] = 1.0
    return vec


def _one_hot_exact(value, choices: list) -> list[float]:
    vec = [0.0] * len(choices)
    try:
        vec[choices.index(value)] = 1.0
    except ValueError:
        pass
    return vec


def _rich_atom_features(atom: Chem.Atom) -> list[float]:
    """
    Extract 45-dim Chemprop-style one-hot + scaled continuous atom feature vector.
    Eliminates ordinal distortion of categorical chemical properties and scales
    atomic mass to O(1) to match binary indicators.
    """
    return (
        _one_hot_with_other(atom.GetSymbol(), _ATOM_SYMBOLS)               # 13
        + _one_hot_with_other(atom.GetTotalDegree(), _DEGREES)             # 7
        + _one_hot_with_other(atom.GetFormalCharge(), _FORMAL_CHARGES)     # 6
        + _one_hot_with_other(atom.GetHybridization(), _HYBRIDIZATIONS)    # 6
        + _one_hot_exact(atom.GetChiralTag(), _CHIRAL_TAGS)                # 4
        + _one_hot_with_other(min(atom.GetTotalNumHs(), 4), _NUM_HS)       # 6 -> wait: 5+1=6
        + [
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(atom.GetMass()) * 0.01,
        ]                                                                  # 3
    )  # Total = 13 + 7 + 6 + 6 + 4 + 6 + 3 = 45


def _rich_bond_features(bond: Chem.Bond) -> list[float]:
    """Extract 10-dim Chemprop-style one-hot bond feature vector."""
    return (
        _one_hot_exact(bond.GetBondType(), _BOND_TYPES)                    # 4
        + [
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
        ]                                                                  # 2
        + _one_hot_exact(bond.GetStereo(), _BOND_STEREOS)                  # 4
    )  # Total = 10


RICH_NODE_DIM = 45
RICH_EDGE_DIM = 10


# ---------------------------------------------------------------------------
# Feature extraction (Legacy 9/4 dims preserved for backward compatibility)
# ---------------------------------------------------------------------------

def _atom_features(atom: Chem.Atom) -> list[float]:
    """Extract 9-dim feature vector for a single atom."""
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetTotalDegree()),
        float(atom.GetFormalCharge()),
        float(_HYBRIDIZATION_MAP.get(atom.GetHybridization(), _HYBRIDIZATION_DEFAULT)),
        float(_CHIRALITY_MAP.get(atom.GetChiralTag(), _CHIRALITY_DEFAULT)),
        float(atom.GetIsAromatic()),
        float(atom.GetTotalNumHs()),
        float(atom.GetMass()),
        float(atom.IsInRing()),
    ]


def _bond_features(bond: Chem.Bond) -> list[float]:
    """Extract 4-dim feature vector for a single bond."""
    return [
        float(_BOND_TYPE_MAP.get(bond.GetBondType(), _BOND_TYPE_DEFAULT)),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        float(_BOND_STEREO_MAP.get(bond.GetStereo(), _BOND_STEREO_DEFAULT)),
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mol_to_data(mol: Chem.Mol, rich: bool = False) -> Optional[Data]:
    """Convert an already-standardized RDKit Mol into a PyG Data object."""
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    atom_fn = _rich_atom_features if rich else _atom_features
    bond_fn = _rich_bond_features if rich else _bond_features
    edge_dim = RICH_EDGE_DIM if rich else EDGE_DIM

    x = torch.tensor([atom_fn(a) for a in mol.GetAtoms()], dtype=torch.float)

    row_list: list[int] = []
    col_list: list[int] = []
    edge_attr_list: list[list[float]] = []

    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bf = bond_fn(bond)

        row_list.append(u)
        col_list.append(v)
        edge_attr_list.append(bf)

        row_list.append(v)
        col_list.append(u)
        edge_attr_list.append(bf)

    if not row_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, edge_dim), dtype=torch.float)
    else:
        edge_index = torch.tensor([row_list, col_list], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def smiles_to_data(smiles: str, rich: bool = False) -> Optional[Data]:
    """
    Convert a single SMILES string into a PyG Data object suitable
    for the DMPNN model.

    Steps:
      1. Standardize via the RDKit pipeline (salt removal, canonicalize, …).
      2. Extract atom features (9-dim legacy or 45-dim one-hot) and bond features.
      3. Build directed edge index (both directions for every bond).

    Args:
        smiles: Input SMILES string.
        rich:   If True, use 45-dim one-hot atom and 10-dim one-hot bond features.

    Returns:
        Data(x, edge_index, edge_attr) or None if the SMILES is invalid.
    """
    try:
        mol = standardize_molecule(smiles)
    except (ValueError, RuntimeError):
        return None

    return mol_to_data(mol, rich=rich)


def smiles_batch_to_data_list(
    smiles_list: list[str],
    rich: bool = False,
) -> Tuple[list[Data], list[dict]]:
    """
    Convert a batch of SMILES strings into a list of PyG Data objects.

    Args:
        smiles_list: List of SMILES strings.
        rich:        If True, use Chemprop-grade one-hot features.

    Returns:
        (valid_data_list, error_info_list) where:
          - valid_data_list[i] maps to the i-th *valid* SMILES
          - error_info_list[j] contains {"index": original_idx, "smiles": s, "error": msg}
            for every *invalid* SMILES.
    """
    valid: list[Data] = []
    errors: list[dict] = []

    for idx, smi in enumerate(smiles_list):
        data = smiles_to_data(smi, rich=rich)
        if data is None:
            errors.append({
                "index": idx,
                "smiles": smi,
                "error": f"Invalid or unparsable SMILES: '{smi}'",
            })
        else:
            data.smiles = smi  # attach metadata for downstream tracking
            data.orig_index = idx
            valid.append(data)

    return valid, errors

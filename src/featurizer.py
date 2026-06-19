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
# Feature extraction
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

def smiles_to_data(smiles: str) -> Optional[Data]:
    """
    Convert a single SMILES string into a PyG Data object suitable
    for the DMPNN model.

    Steps:
      1. Standardize via the RDKit pipeline (salt removal, canonicalize, …).
      2. Extract 9-dim atom features and 4-dim bond features.
      3. Build directed edge index (both directions for every bond).

    Args:
        smiles: Input SMILES string.

    Returns:
        Data(x, edge_index, edge_attr) or None if the SMILES is invalid.
    """
    try:
        mol = standardize_molecule(smiles)
    except (ValueError, RuntimeError):
        return None

    if mol is None or mol.GetNumAtoms() == 0:
        return None

    # --- Atom features -----------------------------------------------------
    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    # --- Edge index + bond features ----------------------------------------
    row_list: list[int] = []
    col_list: list[int] = []
    edge_attr_list: list[list[float]] = []

    num_atoms = mol.GetNumAtoms()
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bf = _bond_features(bond)

        # Forward direction
        row_list.append(u)
        col_list.append(v)
        edge_attr_list.append(bf)

        # Reverse direction (required by DMPNN)
        row_list.append(v)
        col_list.append(u)
        edge_attr_list.append(bf)

    if not row_list:
        # Single-atom molecule (e.g. noble gas) → no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor([row_list, col_list], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def smiles_batch_to_data_list(smiles_list: list[str]) -> Tuple[list[Data], list[dict]]:
    """
    Convert a batch of SMILES strings into a list of PyG Data objects.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        (valid_data_list, error_info_list) where:
          - valid_data_list[i] maps to the i-th *valid* SMILES
          - error_info_list[j] contains {"index": original_idx, "smiles": s, "error": msg}
            for every *invalid* SMILES.
    """
    valid: list[Data] = []
    errors: list[dict] = []

    for idx, smi in enumerate(smiles_list):
        data = smiles_to_data(smi)
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

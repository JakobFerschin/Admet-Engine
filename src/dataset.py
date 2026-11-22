"""
ADMET-Engine — PyG Dataset & DataLoader (ChEMBL v37)
Supports both Bemis-Murcko scaffold splitting (MoleculeNet / Chemprop standard)
and deterministic random splitting.

Reads chembl_admet_training.csv, converts SMILES to graph tensors
via the featurizer, builds binary label vectors [hERG, CYP3A4],
and provides train/val/test splits.

Labels: pre-computed binary (1=active/toxic, 0=inactive/safe).
Missing labels: NaN (masked during loss computation).
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from cheminformatics import get_murcko_scaffold
from data_pipeline import standardize_molecule
from featurizer import mol_to_data

logger = logging.getLogger("admet.dataset")

ENDPOINT_LABELS: list[str] = ["hERG", "CYP3A4"]
NUM_ENDPOINTS: int = len(ENDPOINT_LABELS)


def _load_molecules(
    csv_path: str | Path,
    rich: bool = False,
) -> list[Data]:
    """Load CSV, standardize + featurise every molecule, and build label vectors."""
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)

    has_binary_cols = any(col in df.columns for col in ("hERG_label", "CYP3A4_label"))
    smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else "standardized_smiles"

    mol_data: list[Data] = []
    failed = 0

    if not has_binary_cols:
        # Long-format ETL CSV (multiple assays per molecule) -> aggregate per unique molecule
        grouped: dict[str, tuple[Chem.Mol, np.ndarray]] = {}
        for _, row in df.iterrows():
            smi = str(row[smiles_col])
            try:
                mol = standardize_molecule(smi)
            except (ValueError, RuntimeError):
                failed += 1
                continue
            if mol is None or mol.GetNumAtoms() == 0:
                failed += 1
                continue

            can_smi = Chem.MolToSmiles(mol, canonical=True)
            if can_smi not in grouped:
                grouped[can_smi] = (mol, np.full(NUM_ENDPOINTS, np.nan, dtype=np.float32))

            pchembl = row.get("pchembl_value")
            target = str(row.get("target_name", "")).upper()
            if not pd.isna(pchembl):
                label = 1.0 if float(pchembl) >= 5.0 else 0.0
                if "HERG" in target or "KCNH2" in target:
                    grouped[can_smi][1][0] = label
                elif "CYP3A4" in target:
                    grouped[can_smi][1][1] = label
                else:
                    grouped[can_smi][1][0] = label

        for can_smi, (mol, y) in grouped.items():
            data = mol_to_data(mol, rich=rich)
            if data is None:
                failed += 1
                continue
            data.y = torch.tensor(np.array([y], dtype=np.float32), dtype=torch.float32)
            data.smiles = can_smi
            data.scaffold = get_murcko_scaffold(mol)
            mol_data.append(data)

        logger.info("Featurised %d unique molecules (%d failed)", len(mol_data), failed)
        return mol_data

    for _, row in df.iterrows():
        smi = str(row[smiles_col])
        try:
            mol = standardize_molecule(smi)
        except (ValueError, RuntimeError):
            failed += 1
            continue

        data = mol_to_data(mol, rich=rich)
        if data is None:
            failed += 1
            continue

        y = np.full(NUM_ENDPOINTS, np.nan, dtype=np.float32)
        for i, col in enumerate(["hERG_label", "CYP3A4_label"]):
            val = row.get(col)
            if val == "" or pd.isna(val):
                y[i] = np.nan
            else:
                y[i] = float(val)

        data.y = torch.tensor(np.array([y], dtype=np.float32), dtype=torch.float32)
        data.smiles = Chem.MolToSmiles(mol, canonical=True)
        data.scaffold = get_murcko_scaffold(mol)
        mol_data.append(data)

    logger.info("Featurised %d molecules (%d failed)", len(mol_data), failed)
    return mol_data


def _scaffold_split(
    all_data: list[Data],
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> tuple[list[Data], list[Data], list[Data], dict[str, int]]:
    """
    Deterministic Bemis-Murcko scaffold split (Wu et al., MoleculeNet 2018).
    Ensures no core scaffold is shared across train, validation, and test sets,
    preventing structural analog leakage and measuring true chemical-space
    generalization.
    """
    scaffold_groups: dict[str, list[Data]] = defaultdict(list)
    for d in all_data:
        scaffold_groups[getattr(d, "scaffold", "Acyclic")].append(d)

    # Sort scaffold clusters deterministically: largest first, then lexicographically
    sorted_groups = sorted(
        scaffold_groups.values(),
        key=lambda grp: (-len(grp), grp[0].scaffold),
    )

    n = len(all_data)
    train_cutoff = int(n * split_ratios[0])
    val_cutoff = int(n * (split_ratios[0] + split_ratios[1]))

    train_data: list[Data] = []
    val_data: list[Data] = []
    test_data: list[Data] = []
    n_train_scaf = n_val_scaf = n_test_scaf = 0

    for grp in sorted_groups:
        if len(train_data) + len(grp) <= train_cutoff or not train_data:
            train_data.extend(grp)
            n_train_scaf += 1
        elif len(train_data) + len(val_data) + len(grp) <= val_cutoff or not val_data:
            val_data.extend(grp)
            n_val_scaf += 1
        else:
            test_data.extend(grp)
            n_test_scaf += 1

    # Guarantee non-empty test split on tiny mock datasets
    if not test_data and len(val_data) > 1:
        test_data.append(val_data.pop())
        n_test_scaf = 1
    elif not test_data and len(train_data) > 2:
        test_data.append(train_data.pop())
        n_test_scaf = 1

    scaf_stats = {
        "n_scaffolds_total": len(sorted_groups),
        "n_scaffolds_train": n_train_scaf,
        "n_scaffolds_val": n_val_scaf,
        "n_scaffolds_test": n_test_scaf,
    }
    return train_data, val_data, test_data, scaf_stats


def create_dataloaders(
    csv_path: str | Path,
    batch_size: int = 32,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    num_workers: int = 0,
    split_type: Literal["scaffold", "random"] = "scaffold",
    rich: bool = False,
) -> tuple:
    """
    Create train/val/test DataLoaders from a ChEMBL ADMET CSV.

    Returns:
        (train_loader, val_loader, test_loader, dataset_info_dict)
    """
    all_data = _load_molecules(csv_path, rich=rich)
    n = len(all_data)

    if n == 0:
        raise RuntimeError(f"No valid molecules found in {csv_path}")

    scaf_stats: dict[str, int] = {}
    if split_type == "scaffold" and n >= 10:
        train_data, val_data, test_data, scaf_stats = _scaffold_split(all_data, split_ratios)
    else:
        n_train = max(1, int(n * split_ratios[0]))
        n_val = max(1, int(n * split_ratios[1]))
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_val = max(1, n - n_train - n_test)
            n_train = max(1, n - n_val - n_test)

        rng = random.Random(seed)
        shuffled = list(all_data)
        rng.shuffle(shuffled)

        train_data = shuffled[:n_train]
        val_data = shuffled[n_train:n_train + n_val]
        test_data = shuffled[n_train + n_val:]

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    info = {
        "n_total": n,
        "n_train": len(train_data),
        "n_val": len(val_data),
        "n_test": len(test_data),
        "num_endpoints": NUM_ENDPOINTS,
        "split_type": split_type if n >= 10 else "random",
        **scaf_stats,
    }
    logger.info(
        "Split (%s): %d/%d/%d (train/val/test)",
        info["split_type"], info["n_train"], info["n_val"], info["n_test"],
    )

    return train_loader, val_loader, test_loader, info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    csv = Path(__file__).resolve().parent.parent / "data" / "processed" / "chembl_admet_training.csv"
    tl, vl, tst, info = create_dataloaders(csv, batch_size=64, split_type="scaffold")
    print(f"Info: {info}")
    for batch in tl:
        print(
            f"Train batch: x={batch.x.shape}, y={batch.y.shape}, "
            f"non-NaN={(~batch.y.isnan()).sum().item()}/{batch.y.numel()}"
        )
        break

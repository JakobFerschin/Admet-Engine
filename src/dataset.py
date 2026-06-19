"""
ADMET-Engine — PyG Dataset & DataLoader (ChEMBL v37)
Phase 7.2: Real ChEMBL training data — 2 targets (hERG + CYP3A4)

Reads chembl_admet_training.csv, converts SMILES to graph tensors
via the featurizer, builds binary label vectors [hERG, CYP3A4],
and provides train/val/test splits.

Labels: pre-computed binary (1=active/toxic, 0=inactive/safe).
Missing labels: NaN (masked during loss computation).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from featurizer import smiles_to_data

logger = logging.getLogger("admet.dataset")

ENDPOINT_LABELS: list[str] = ["hERG", "CYP3A4"]
NUM_ENDPOINTS: int = len(ENDPOINT_LABELS)


def _load_molecules(csv_path: str | Path) -> list[Data]:
    """Load CSV, featurise every molecule, build 2-target label vectors."""
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)

    mol_data: list[Data] = []
    failed = 0

    for _, row in df.iterrows():
        smi = str(row["canonical_smiles"])
        data = smiles_to_data(smi)
        if data is None:
            failed += 1
            continue

        # Build [hERG_label, CYP3A4_label] from CSV columns
        y = np.full(NUM_ENDPOINTS, np.nan, dtype=np.float32)
        for i, col in enumerate(["hERG_label", "CYP3A4_label"]):
            val = row.get(col)
            if val == "" or pd.isna(val):
                y[i] = np.nan
            else:
                y[i] = float(val)

        # 2D tensor [1, 2] for PyG DataLoader batching
        data.y = torch.tensor([y], dtype=torch.float32)
        data.smiles = smi
        mol_data.append(data)

    logger.info("Featurised %d molecules (%d failed)", len(mol_data), failed)
    return mol_data


def create_dataloaders(
    csv_path: str | Path,
    batch_size: int = 32,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    num_workers: int = 0,
) -> tuple:
    """
    Create train/val/test DataLoaders from chembl_admet_training.csv.

    Returns:
        (train_loader, val_loader, test_loader, dataset_info_dict)
    """
    all_data = _load_molecules(csv_path)
    n = len(all_data)

    if n == 0:
        raise RuntimeError(f"No valid molecules found in {csv_path}")

    n_train = max(1, int(n * split_ratios[0]))
    n_val   = max(1, int(n * split_ratios[1]))
    n_test  = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_val = max(1, n - n_train - n_test)

    rng = random.Random(seed)
    shuffled = list(all_data)
    rng.shuffle(shuffled)

    train_data = shuffled[:n_train]
    val_data   = shuffled[n_train:n_train + n_val]
    test_data  = shuffled[n_train + n_val:]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    info = {
        "n_total": n, "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "num_endpoints": NUM_ENDPOINTS,
    }
    logger.info("Split: %d/%d/%d (train/val/test)", n_train, n_val, n_test)

    return train_loader, val_loader, test_loader, info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    csv = Path(__file__).resolve().parent.parent / "data" / "processed" / "chembl_admet_training.csv"
    tl, vl, tst, info = create_dataloaders(csv, batch_size=64)
    print(f"Info: {info}")
    for batch in tl:
        print(f"Train batch: x={batch.x.shape}, y={batch.y.shape}, "
              f"non-NaN={(~batch.y.isnan()).sum().item()}/{batch.y.numel()}")
        break

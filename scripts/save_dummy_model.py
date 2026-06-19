#!/usr/bin/env python
"""
ADMET-Engine — Dummy Model Saver
Phase 4, Month 7–9: Backend & API Infrastructure

Initialises the ADMETPredictor with random weights and saves the
state_dict to models/dmpnn_v1.pt so the FastAPI server has something
to load before real training data is available.
"""

import sys
from pathlib import Path

# Ensure src/ is importable
_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))

import torch
from models.dmpnn import ADMETPredictor


def main():
    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    save_path = model_dir / "dmpnn_v1.pt"

    print(f"Initialising ADMETPredictor (hidden_dim=300, num_tasks=12) …")
    model = ADMETPredictor(
        node_dim=9,
        edge_dim=4,
        hidden_dim=300,
        num_tasks=12,
        T=3,
        set2set_steps=3,
        dropout=0.1,
    )
    model.eval()

    torch.save(model.state_dict(), str(save_path))
    print(f"[OK] Dummy model saved -> {save_path}  ({save_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

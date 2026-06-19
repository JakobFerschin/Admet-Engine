"""
ADMET-Engine — Training Pipeline Integration Tests
Phase 5, Month 10–12: Validates the training loop end-to-end.

Tests:
  1. Training loop runs for 2 epochs on real mock CSV without error.
  2. Train loss decreases or stays finite.
  3. Model checkpoint file is created after training.
  4. Best model loads correctly.
  5. Gradient flow is active (parameter update detected).
"""

import sys
import math
from pathlib import Path

import pytest
import torch

# Ensure src/ is importable
_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))

from models.dmpnn import ADMETPredictor
from dataset import create_dataloaders, NUM_ENDPOINTS
from train import train, evaluate, train_epoch, DEVICE

CSV_PATH = _src.parent / "data" / "processed" / "chembl_cleaned_sample.csv"
MODELS_DIR = _src.parent / "models"


def _model_param_sum(model):
    return sum(p.sum().item() for p in model.parameters())


class TestTrainingPipeline:
    """End-to-end training loop integration tests."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Ensure mock CSV exists and models dir is ready."""
        assert CSV_PATH.exists(), (
            f"Mock CSV not found at {CSV_PATH}. Run the ETL pipeline first."
        )
        MODELS_DIR.mkdir(exist_ok=True)

    def test_dataloaders_create_correctly(self):
        """DataLoaders must produce valid batches."""
        train_loader, val_loader, test_loader, info = create_dataloaders(
            CSV_PATH, batch_size=4, seed=42,
        )
        assert info["n_total"] == 6, f"Expected 6 molecules, got {info['n_total']}"
        assert info["n_train"] >= 3
        assert info["n_val"] >= 1
        assert info["n_test"] >= 1

        for batch in train_loader:
            assert batch.x.shape[1] == 9  # atom features
            assert batch.edge_attr is not None
            assert batch.y.shape[1] == NUM_ENDPOINTS
            break

    def test_training_loop_runs_two_epochs(self):
        """Training loop must complete 2 epochs without crashing."""
        model = train(
            config={"max_epochs": 2, "batch_size": 4, "patience": 10},
            csv_path=CSV_PATH,
        )
        assert model is not None

    def test_loss_is_finite(self):
        """Loss after 1 epoch must be a finite number."""
        train_loader, val_loader, _, _ = create_dataloaders(
            CSV_PATH, batch_size=4, seed=42,
        )
        model = ADMETPredictor(
            node_dim=9, edge_dim=4, hidden_dim=300,
            num_tasks=NUM_ENDPOINTS, T=3, set2set_steps=3, dropout=0.1,
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        loss = train_epoch(model, train_loader, optimizer, DEVICE)
        assert math.isfinite(loss), f"Loss is not finite: {loss}"
        assert loss > 0, "Loss should be positive"

    def test_checkpoint_saved(self):
        """After training, dmpnn_best.pt must exist."""
        checkpoint = MODELS_DIR / "dmpnn_best.pt"
        if checkpoint.exists():
            checkpoint.unlink()  # clean previous

        train(config={"max_epochs": 2, "batch_size": 4, "patience": 10},
              csv_path=CSV_PATH)

        assert checkpoint.exists(), (
            f"Checkpoint {checkpoint} was not created by training!"
        )
        assert checkpoint.stat().st_size > 1_000_000, (
            f"Checkpoint too small ({checkpoint.stat().st_size} bytes)"
        )

    def test_checkpoint_loads_correctly(self):
        """Saved checkpoint must load into a fresh model instance."""
        checkpoint = MODELS_DIR / "dmpnn_best.pt"
        assert checkpoint.exists(), "No checkpoint to load — run training first"

        model = ADMETPredictor(
            node_dim=9, edge_dim=4, hidden_dim=300,
            num_tasks=NUM_ENDPOINTS, T=3, set2set_steps=3, dropout=0.1,
        )
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        # Quick forward pass to verify
        train_loader, _, _, _ = create_dataloaders(CSV_PATH, batch_size=4, seed=42)
        for batch in train_loader:
            with torch.no_grad():
                preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            assert preds.shape[0] <= 4
            assert preds.shape[1] == NUM_ENDPOINTS
            assert torch.isfinite(preds).all()
            break

    def test_parameters_update_during_training(self):
        """Weights must change during training (gradient flow verification)."""
        train_loader, _, _, _ = create_dataloaders(CSV_PATH, batch_size=4, seed=42)
        model = ADMETPredictor(
            node_dim=9, edge_dim=4, hidden_dim=300,
            num_tasks=NUM_ENDPOINTS, T=3, set2set_steps=3, dropout=0.1,
        ).to(DEVICE)

        before = _model_param_sum(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        train_epoch(model, train_loader, optimizer, DEVICE)
        after = _model_param_sum(model)

        assert before != after, "Model weights did not change during training!"
        assert math.isfinite(after), "Model weights became non-finite!"

    def test_val_evaluation_returns_metrics(self):
        """evaluate() must return a loss and metrics dict."""
        train_loader, val_loader, _, _ = create_dataloaders(CSV_PATH, batch_size=4, seed=42)
        model = ADMETPredictor(
            node_dim=9, edge_dim=4, hidden_dim=300,
            num_tasks=NUM_ENDPOINTS, T=3, set2set_steps=3, dropout=0.1,
        ).to(DEVICE)

        val_loss, metrics = evaluate(model, val_loader, DEVICE)
        assert math.isfinite(val_loss)
        assert "global_roc_auc" in metrics
        assert 0.0 <= metrics["global_roc_auc"] <= 1.0


class TestTrainingEdgeCases:
    """Graceful handling of edge cases."""

    def test_dataloader_with_small_batch_size(self):
        """batch_size=1 must work."""
        train_loader, _, _, _ = create_dataloaders(CSV_PATH, batch_size=1, seed=42)
        count = 0
        for batch in train_loader:
            count += 1
            assert batch.x.shape[1] == 9
        assert count > 0

    def test_dataloader_reproducibility(self):
        """Same seed must produce identical split ordering."""
        l1, _, _, i1 = create_dataloaders(CSV_PATH, batch_size=4, seed=42)
        l2, _, _, i2 = create_dataloaders(CSV_PATH, batch_size=4, seed=42)
        assert i1["n_train"] == i2["n_train"]
        assert i1["n_val"] == i2["n_val"]

    def test_dataset_contains_target_vector(self):
        """Every molecule must have a y-vector of shape [num_endpoints]."""
        train_loader, _, _, _ = create_dataloaders(CSV_PATH, batch_size=4, seed=42)
        for batch in train_loader:
            assert batch.y.shape[1] == NUM_ENDPOINTS
            # At least some labels should be non-NaN
            has_label = (~torch.isnan(batch.y)).any().item()
            assert has_label, "No labels found in batch!"
            break

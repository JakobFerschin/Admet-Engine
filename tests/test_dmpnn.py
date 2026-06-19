"""
ADMET-Engine — DMPNN Architecture Tests
Phase 2, Month 4–6: Unit & integration tests for the model core.

Verifies:
  1. DMPNNEncoder atom embedding shape  [total_nodes, hidden_dim=300]
  2. ADMETPredictor output shape         [batch_size, num_tasks=12]
  3. Gradient flow (backprop through full model)
  4. Batched multi-graph behaviour
  5. Reproducibility (deterministic seed)
  6. Dropout behaviour (train vs eval mode)
  7. Parameter count sanity check
"""

import sys
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data, Batch

# Ensure src/ and src/models/ are importable
_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))

from models.dmpnn import DMPNNEncoder, ADMETPredictor  # noqa: E402


# ============================================================================
# Mock data generators
# ============================================================================

def _make_mock_graph(
    num_nodes: int = 10,
    num_bonds: int = 13,  # undirected bonds; 2× directed edges
    node_dim: int = 9,
    edge_dim: int = 4,
    seed: int = 42,
) -> Data:
    """
    Build a single mock molecular graph with random features.

    Creates a connected path-graph backbone with extra random edges
    to simulate a realistic molecular topology.  Both directions are
    included for every bond (required by DMPNN).
    """
    g = torch.Generator()
    g.manual_seed(seed)

    # Create atom features [num_nodes, node_dim]
    x = torch.randn(num_nodes, node_dim, generator=g)

    # Build a connected backbone (path 0→1→2→…→N−1)
    edges = []
    for i in range(num_nodes - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))  # reverse

    # Add extra random undirected edges to hit num_bonds
    n_needed = num_bonds - (num_nodes - 1)
    existing = {(min(a, b), max(a, b)) for a, b in edges}
    for _ in range(n_needed):
        for __ in range(100):  # try to avoid self-loops and duplicates
            u = int(torch.randint(0, num_nodes, (1,), generator=g).item())
            v = int(torch.randint(0, num_nodes, (1,), generator=g).item())
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            if key not in existing:
                existing.add(key)
                edges.append((u, v))
                edges.append((v, u))
                break

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    num_directed = edge_index.size(1)

    # Create bond features [num_directed_edges, edge_dim]
    edge_attr = torch.randn(num_directed, edge_dim, generator=g)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def make_batch(
    num_molecules: int = 2,
    atoms_per_mol: int = 10,
    bonds_per_mol: int = 13,
    node_dim: int = 9,
    edge_dim: int = 4,
    seed: int = 42,
) -> Batch:
    """Create a PyG Batch of mock molecular graphs."""
    graphs = [
        _make_mock_graph(
            num_nodes=atoms_per_mol,
            num_bonds=bonds_per_mol,
            node_dim=node_dim,
            edge_dim=edge_dim,
            seed=seed + i,
        )
        for i in range(num_molecules)
    ]
    return Batch.from_data_list(graphs)


# ============================================================================
# Shared fixtures
# ============================================================================

@pytest.fixture(scope="module")
def encoder() -> DMPNNEncoder:
    """A fresh DMPNNEncoder with document-specified dimensions."""
    return DMPNNEncoder(node_dim=9, edge_dim=4, hidden_dim=300, T=3, dropout=0.0)


@pytest.fixture(scope="module")
def predictor() -> ADMETPredictor:
    """A fresh ADMETPredictor with 12 tasks."""
    return ADMETPredictor(
        node_dim=9,
        edge_dim=4,
        hidden_dim=300,
        num_tasks=12,
        T=3,
        set2set_steps=3,
        dropout=0.1,
    )


@pytest.fixture
def batch2() -> Batch:
    """Batch of 2 molecules, ~10 atoms / ~13 bonds each."""
    return make_batch(num_molecules=2)


@pytest.fixture
def batch1() -> Batch:
    """Single molecule batch."""
    return make_batch(num_molecules=1)


# ============================================================================
# DMPNNEncoder tests
# ============================================================================

class TestDMPNNEncoder:
    """Unit tests for the Directed MPNN encoder."""

    def test_output_shape(self, encoder, batch2):
        """Atom embeddings must be [total_nodes, hidden_dim=300]."""
        out = encoder(batch2.x, batch2.edge_index, batch2.edge_attr)
        assert out.shape == (batch2.x.size(0), encoder.hidden_dim), (
            f"Expected [num_nodes={batch2.x.size(0)}, hidden={encoder.hidden_dim}], "
            f"got {list(out.shape)}"
        )

    def test_output_shape_single_molecule(self, encoder, batch1):
        """Single molecule output should also be [N, 300]."""
        out = encoder(batch1.x, batch1.edge_index, batch1.edge_attr)
        assert out.shape[1] == 300
        assert out.shape[0] == batch1.x.size(0)

    def test_encoder_is_deterministic(self, encoder, batch2):
        """Same input twice → identical output."""
        torch.manual_seed(0)
        out1 = encoder(batch2.x, batch2.edge_index, batch2.edge_attr)
        torch.manual_seed(0)
        out2 = encoder(batch2.x, batch2.edge_index, batch2.edge_attr)
        assert torch.allclose(out1, out2, atol=1e-6), "Encoder is not deterministic!"

    def test_gradient_flow_encoder(self, encoder, batch2):
        """Gradients must reach the encoder's parameters."""
        encoder.train()
        out = encoder(batch2.x, batch2.edge_index, batch2.edge_attr)
        loss = out.sum()
        loss.backward()

        for name, param in encoder.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient!"
            )
            assert param.grad.abs().sum() > 0, (
                f"Parameter '{name}' gradient is all zeros!"
            )

    def test_parameter_count(self, encoder):
        """Sanity check: encoder should have ~300K parameters."""
        total = sum(p.numel() for p in encoder.parameters())
        # W_i: (9+4)×300 = 3900
        # W_m: 300×300 = 90000
        # W_a: (9+300)×300 = 92700
        # Total = 186600 (no biases in W_i, W_m)
        assert 150_000 < total < 250_000, (
            f"Encoder param count ({total:,}) outside expected range!"
        )

    def test_different_sized_graphs(self, encoder):
        """Encoder must handle graphs of different sizes."""
        for n_atoms, n_bonds in [(5, 6), (20, 25), (50, 65)]:
            g = _make_mock_graph(num_nodes=n_atoms, num_bonds=n_bonds)
            out = encoder(g.x, g.edge_index, g.edge_attr)
            assert out.shape[0] == n_atoms
            assert out.shape[1] == 300


# ============================================================================
# ADMETPredictor integration tests
# ============================================================================

class TestADMETPredictor:
    """Integration tests for the full multitask prediction model."""

    def test_output_shape(self, predictor, batch2):
        """Output must be [batch_size=2, num_tasks=12]."""
        preds = predictor(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
        assert preds.shape == (2, 12), (
            f"Expected [2, 12], got {list(preds.shape)}"
        )

    def test_output_shape_single_molecule(self, predictor, batch1):
        """Single molecule → [1, num_tasks]."""
        preds = predictor(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        assert preds.shape == (1, 12)

    def test_output_shape_variable_batch(self, predictor):
        """Batch size of 5 → [5, 12]."""
        batch5 = make_batch(num_molecules=5, atoms_per_mol=8, bonds_per_mol=10)
        preds = predictor(batch5.x, batch5.edge_index, batch5.edge_attr, batch5.batch)
        assert preds.shape == (5, 12)

    def test_backpropagation(self, predictor, batch2):
        """
        Full backpropagation test:
        1. Forward pass → predictions
        2. Compute MSELoss against random targets
        3. loss.backward() → gradients flow
        4. Check key parameters have non-None gradients
        """
        predictor.train()
        preds = predictor(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)

        # Random targets for the 12 tasks (2 molecules × 12 tasks)
        targets = torch.randn(2, 12)
        loss = F.mse_loss(preds, targets)
        loss.backward()

        # Check that key encoder parameters received gradients
        assert predictor.encoder.W_i.weight.grad is not None, (
            "encoder.W_i.weight.grad is None — backprop failed!"
        )
        assert predictor.encoder.W_m.weight.grad is not None, (
            "encoder.W_m.weight.grad is None — backprop failed!"
        )
        assert predictor.encoder.W_a.weight.grad is not None, (
            "encoder.W_a.weight.grad is None — backprop failed!"
        )

        # Check task head gradients
        for i, head in enumerate(predictor.task_heads):
            first_linear = head[0]  # nn.Linear
            assert first_linear.weight.grad is not None, (
                f"Task head {i} linear.weight.grad is None"
            )

        # Gradients should be non-zero (not all zeros)
        assert predictor.encoder.W_a.weight.grad.abs().sum() > 0, (
            "encoder.W_a.weight.grad is all zeros — no gradient signal!"
        )

    def test_dropout_behavior(self, predictor, batch2):
        """Dropout active in train mode, inactive in eval."""
        predictor.train()
        preds_train = predictor(
            batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch
        )

        predictor.eval()
        with torch.no_grad():
            preds_eval = predictor(
                batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch
            )

        # In train mode with dropout > 0, predictions should differ from eval
        # (stochastic dropout vs no dropout)
        assert not torch.allclose(preds_train, preds_eval, atol=1e-5), (
            "Train/eval predictions are identical — dropout not working!"
        )

    def test_total_parameter_count(self, predictor):
        """Full model should have a reasonable number of trainable parameters.
        ~186K (encoder) + ~700K (Set2Set LSTM) + ~2.2M (12× MLP heads) ≈ 3–4M."""
        total = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
        assert 2_500_000 < total < 5_000_000, (
            f"Total parameters ({total:,}) outside expected range (2.5M–5M)"
        )

    def test_reproducibility(self, predictor, batch2):
        """Same seed → same output."""
        torch.manual_seed(42)
        predictor.eval()
        with torch.no_grad():
            out1 = predictor(
                batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch
            )

        torch.manual_seed(42)
        with torch.no_grad():
            out2 = predictor(
                batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch
            )

        assert torch.allclose(out1, out2, atol=1e-6), (
            "Predictor is not reproducible with the same seed!"
        )

    def test_output_is_finite(self, predictor, batch2):
        """Output should not contain NaN or Inf."""
        predictor.eval()
        with torch.no_grad():
            preds = predictor(
                batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch
            )
        assert torch.isfinite(preds).all(), "Predictions contain NaN or Inf!"

    def test_gradient_norms_reasonable(self, predictor, batch2):
        """Gradient norms should be finite and non-zero."""
        predictor.train()
        preds = predictor(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
        loss = F.mse_loss(preds, torch.randn(2, 12))
        loss.backward()

        for name, param in predictor.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert torch.isfinite(torch.tensor(grad_norm)), (
                    f"NaN/Inf gradient in '{name}'"
                )
                assert grad_norm < 100, (
                    f"Exploding gradient in '{name}': norm={grad_norm:.2f}"
                )

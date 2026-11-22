"""
ADMET-Engine — Directed Message Passing Neural Network (DMPNN)
Phase 2, Month 4–6: GNN Modelling & Multitask Training

Implements the DMPNN architecture as specified in Sections 4.1 and 4.8
of the ADMET-Engine Project Report.

Architecture summary (document-conformant):
  ┌─────────────────────────────────────────────────┐
  │  DMPNN Encoder                                   │
  │  • Directed bond messages, T = 3 steps           │
  │  • Atom features: 9 dim, Bond features: 4 dim    │
  │  • hidden_dim = 300                              │
  │  • Message: m_{vw}^{t+1} = Σ_{k∈N(v)\\w} h_{kv}^t │
  │  • Atom:    h_v = τ(W_o · cat(x_v, Σ incoming))  │
  ├─────────────────────────────────────────────────┤
  │  Set2Set Readout (processing_steps = 3)          │
  │  → molecular embedding: 2×300 = 600 dim          │
  ├─────────────────────────────────────────────────┤
  │  ADMETPredictor                                  │
  │  • nn.ModuleList of MLP heads (1 per task)       │
  │  • Default: 12 tasks (hepatotox, hERG, AMES, …)  │
  │  • Output: [batch_size, num_tasks]               │
  └─────────────────────────────────────────────────┘

References:
  - Yang, K. et al. "Analyzing Learned Molecular Representations for
    Property Prediction." JCIM, 2019.  (Chemprop / DMPNN paper)
  - Section 4.1, 4.8: ADMET-Engine Project Report (2026)

Author: ML Architect (Persona 2)
Project: ADMET-Engine, Phase 2 — Model Core
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import Set2Set
from torch_geometric.utils import scatter


# ===========================================================================
# DMPNN Encoder — Directed Message Passing
# ===========================================================================

class DMPNNEncoder(nn.Module):
    """
    Directed Message Passing Neural Network encoder.

    Initialises directed-edge hidden states from atom + bond features,
    then applies T rounds of message passing where each directed edge
    (v → w) receives messages from all edges that end at atom v
    (i.e. k → v for k ≠ w).

    After message passing, atom hidden states are computed by
    concatenating the initial atom features with the sum of all
    incoming edge hidden states.

    Args:
        node_dim:   Dimensionality of input atom features  (default: 9).
        edge_dim:   Dimensionality of input bond features  (default: 4).
        hidden_dim: Hidden dimension throughout the network (default: 300).
        T:          Number of message-passing steps         (default: 3).
        dropout:    Dropout probability applied to edge features (default: 0.0).
    """

    def __init__(
        self,
        node_dim: int = 9,
        edge_dim: int = 4,
        hidden_dim: int = 300,
        T: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.T = T
        self.dropout = nn.Dropout(dropout)

        # --- Initial edge projection ---
        # h_{uv}^0 = τ(W_i · cat(x_u, e_{uv}))
        self.W_i = nn.Linear(node_dim + edge_dim, hidden_dim, bias=False)

        # --- Message update ---
        # m_{vw}^{t+1} = sum of incoming, then h_{vw}^{t+1} = τ(W_m · m + h_{vw}^0)
        self.W_m = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # --- Atom embedding ---
        # h_v = τ(W_a · cat(x_v, sum of incoming edge hiddens))
        self.W_a = nn.Linear(node_dim + hidden_dim, hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_i.weight)
        nn.init.xavier_uniform_(self.W_m.weight)
        nn.init.xavier_uniform_(self.W_a.weight)
        if self.W_a.bias is not None:
            nn.init.zeros_(self.W_a.bias)

    def _build_reverse_map(self, edge_index: Tensor) -> Tensor:
        """
        For each directed edge i = (u, v), find the index of its reverse
        edge j = (v, u).  Assumes the graph contains both directions for
        every undirected bond.

        Uses an O(E) fast-path when directed edges are stored as adjacent
        (u→v, v→u) pairs (standard RDKit featurizer layout), and falls back
        to an O(E log E) sorted lookup with O(E) memory — avoiding the
        O(N^2) memory blowup on large batched graphs.

        Returns a LongTensor of shape [num_edges] where rev[i] = j.
        """
        num_edges = edge_index.size(1)
        device = edge_index.device
        if num_edges == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        src, dst = edge_index[0], edge_index[1]

        # O(E) fast-path: edges stored as consecutive (u, v), (v, u) pairs
        if num_edges % 2 == 0:
            even_src, odd_src = src[0::2], src[1::2]
            even_dst, odd_dst = dst[0::2], dst[1::2]
            if torch.equal(even_src, odd_dst) and torch.equal(even_dst, odd_src):
                return torch.arange(num_edges, dtype=torch.long, device=device) ^ 1

        # General O(E log E) time, O(E) memory fallback for arbitrary edge order
        num_nodes = int(edge_index.max().item()) + 1
        src64 = src.to(torch.int64)
        dst64 = dst.to(torch.int64)
        forward_key = src64 * num_nodes + dst64
        reverse_key = dst64 * num_nodes + src64

        order = torch.argsort(forward_key)
        sorted_fwd = forward_key[order]
        pos = torch.searchsorted(sorted_fwd, reverse_key)

        in_bounds = pos < num_edges
        clamped_pos = pos.clamp(max=num_edges - 1)
        matched = in_bounds & (sorted_fwd[clamped_pos] == reverse_key)
        if not bool(matched.all()):
            raise RuntimeError(
                "DMPNNEncoder requires both directions for every bond. "
                "Some directed edges have no reverse counterpart."
            )
        return order[clamped_pos]

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        """
        Forward pass through the DMPNN encoder.

        Args:
            x:          Atom feature matrix  [num_nodes, node_dim].
            edge_index: Edge index tensor     [2, num_edges]  (source → target).
            edge_attr:  Bond feature matrix   [num_edges, edge_dim].

        Returns:
            Atom hidden states [num_nodes, hidden_dim].
        """
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        # Edge-free molecules (e.g. single-atom ions or noble gases)
        if num_edges == 0:
            atom_msg = torch.zeros(
                (num_nodes, self.hidden_dim), dtype=x.dtype, device=x.device
            )
            atom_embedding = F.relu(self.W_a(torch.cat([x, atom_msg], dim=-1)))
            return self.dropout(atom_embedding)

        src, dst = edge_index[0], edge_index[1]

        # ── 1. Initial edge hidden states ──────────────────────────
        # h_{uv}^0 = τ(W_i · cat(x_u, e_{uv}))
        edge_init = self.W_i(torch.cat([x[src], edge_attr], dim=-1))
        edge_init = F.relu(edge_init)
        edge_init = self.dropout(edge_init)
        edge_hidden = edge_init.clone()

        # ── 2. Precompute reverse edge indices ─────────────────────
        rev_edge = self._build_reverse_map(edge_index)

        # ── 3. Message passing (T steps) ───────────────────────────
        for _ in range(self.T):
            # Sum all *incoming* edge hidden states per node
            # incoming[v] = sum_{k: (k,v) ∈ E} h_{kv}
            incoming = scatter(
                edge_hidden,
                dst,
                dim=0,
                dim_size=num_nodes,
                reduce="sum",
            )

            # For edge i = (v → w), message = incoming[v] − h_{w→v}
            # (subtract the reverse edge to satisfy k ∈ N(v) \ {w})
            messages = incoming[src] - edge_hidden[rev_edge]

            # Update: h_{vw}^{t+1} = τ(h_{vw}^0 + W_m · m_{vw})
            edge_hidden = F.relu(edge_init + self.W_m(messages))
            edge_hidden = self.dropout(edge_hidden)

        # ── 4. Atom embedding ─────────────────────────────────────
        # m_v = sum of final incoming edge hiddens
        atom_msg = scatter(
            edge_hidden,
            dst,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )

        # h_v = τ(W_a · cat(x_v, m_v))
        atom_embedding = F.relu(self.W_a(torch.cat([x, atom_msg], dim=-1)))
        atom_embedding = self.dropout(atom_embedding)

        return atom_embedding

    def extra_repr(self) -> str:
        return (
            f"node_dim={self.node_dim}, edge_dim={self.edge_dim}, "
            f"hidden_dim={self.hidden_dim}, T={self.T}"
        )


# ===========================================================================
# ADMET Predictor — Full multitask model
# ===========================================================================

class ADMETPredictor(nn.Module):
    """
    Complete ADMET prediction model: DMPNN encoder + Set2Set readout +
    per-task MLP heads.

    Forward pass:
      1. DMPNNEncoder → atom embeddings  [num_nodes, hidden_dim]
      2. Set2Set → molecular embeddings   [batch_size, 2×hidden_dim]
      3. MLP heads → per-task logits      [batch_size, num_tasks]

    Args:
        node_dim:    Atom feature dimensionality  (default: 9).
        edge_dim:    Bond feature dimensionality  (default: 4).
        hidden_dim:  Hidden dimension              (default: 300).
        num_tasks:   Number of ADMET endpoints     (default: 12).
        T:           DMPNN message-passing steps   (default: 3).
        set2set_steps: Set2Set processing steps    (default: 3).
        dropout:     Dropout probability            (default: 0.1).
    """

    def __init__(
        self,
        node_dim: int = 9,
        edge_dim: int = 4,
        hidden_dim: int = 300,
        num_tasks: int = 12,
        T: int = 3,
        set2set_steps: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.set2set_steps = set2set_steps

        # --- Core graph encoder ---
        self.encoder = DMPNNEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            T=T,
            dropout=dropout,
        )

        # --- Graph → molecule readout ---
        # Set2Set output = 2 × in_channels  (concatenation of Q_star)
        readout_dim = 2 * hidden_dim  # = 600
        self.readout = Set2Set(
            in_channels=hidden_dim,
            processing_steps=set2set_steps,
            num_layers=1,
        )
        self.readout_dropout = nn.Dropout(dropout)

        # --- Per-task MLP heads ---
        # Architecture: (600 → 300 → 1) with ReLU + Dropout
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(readout_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_tasks)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        """Re-initialise all trainable parameters."""
        self.encoder.reset_parameters()
        self.readout.reset_parameters()
        for head in self.task_heads:
            for module in head:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x:          Atom features   [total_atoms, node_dim].
            edge_index: Graph topology  [2, total_edges].
            edge_attr:  Bond features   [total_edges, edge_dim].
            batch:      Batch vector    [total_atoms] — maps each atom
                        to its molecule index (0 … batch_size−1).

        Returns:
            Predictions [batch_size, num_tasks].
        """
        # 1. DMPNN encoder → atom embeddings
        atom_emb = self.encoder(x, edge_index, edge_attr)

        # 2. Set2Set readout → molecule embeddings
        mol_emb = self.readout(atom_emb, batch)
        mol_emb = self.readout_dropout(mol_emb)

        # 3. Per-task prediction
        preds: list[Tensor] = []
        for head in self.task_heads:
            preds.append(head(mol_emb))

        return torch.cat(preds, dim=-1)

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, num_tasks={self.num_tasks}, "
            f"readout_dim={2 * self.hidden_dim}"
        )

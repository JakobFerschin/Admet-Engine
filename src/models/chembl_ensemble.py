"""
ADMET-Engine — ChEMBL v37 Multi-Modal Calibrated Ensemble
Combines four structurally diverse model families for hERG (KCNH2) cardiotoxicity
and CYP3A4 metabolic inhibition:
  1. Chemprop-grade One-Hot DMPNN (PyTorch Geometric, 45-dim atom / 10-dim bond)
  2. Residual ECFP4 + RDKit PhysChem Deep Neural Network (PyTorch)
  3. Non-linear ExtraTrees QSAR Classifier (scikit-learn)
  4. L2-Regularized Logistic QSAR Classifier (scikit-learn)

Provides:
  • Calibrated ensemble consensus probability p in [0, 1]
  • Epistemic uncertainty (weighted inter-model standard deviation)
  • Vectorized Tanimoto Applicability Domain against the ChEMBL v37 training corpus
  • Nearest-neighbor training analog lookup with experimental labels
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Batch

from cheminformatics import (
    compute_morgan_numpy,
    compute_rdkit_ml_descriptors,
)
from featurizer import RICH_EDGE_DIM, RICH_NODE_DIM, mol_to_data
from models.dmpnn import ADMETPredictor

CHEMBL_TASKS: list[str] = ["hERG_inhibition", "CYP3A4_inhibition"]
SHORT_TASKS: list[str] = ["hERG", "CYP3A4"]


# ---------------------------------------------------------------------------
# PyTorch Residual ECFP4 + PhysChem Network
# ---------------------------------------------------------------------------

class FingerprintPhysChemNet(nn.Module):
    """
    Multi-task residual MLP operating on concatenated 2048-bit Morgan ECFP4
    fingerprints and 20 normalized RDKit 2D physicochemical descriptors.
    """

    def __init__(
        self,
        in_dim: int = 2068,
        hidden_dim: int = 384,
        num_tasks: int = 2,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_tasks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = h + self.res_block(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Fast Vectorized Packed-Bit Tanimoto Applicability Domain
# ---------------------------------------------------------------------------

# Precomputed byte popcount table (0..255) for exact sub-millisecond Tanimoto
_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def pack_fingerprints(fps_float: np.ndarray) -> np.ndarray:
    """Pack [N, 2048] binary float/uint8 fingerprints into [N, 256] uint8."""
    bits = (fps_float > 0.5).astype(np.uint8)
    return np.packbits(bits, axis=1)


def compute_tanimoto_batch(
    query_packed: np.ndarray,
    ref_packed: np.ndarray,
    ref_popcounts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute exact Tanimoto similarity between each query [M, 256] and
    all reference fingerprints [N, 256] using vectorized bitwise AND + LUT.

    Returns:
        (max_sim [M], mean_top5_sim [M], best_ref_idx [M])
    """
    m = query_packed.shape[0]
    n_ref = ref_packed.shape[0]
    if n_ref == 0:
        return (
            np.zeros(m, dtype=np.float32),
            np.zeros(m, dtype=np.float32),
            np.full(m, -1, dtype=np.int64),
        )

    q_counts = _POPCOUNT_LUT[query_packed].sum(axis=1, dtype=np.float32)  # [M]
    max_sims = np.zeros(m, dtype=np.float32)
    top5_sims = np.zeros(m, dtype=np.float32)
    best_idx = np.zeros(m, dtype=np.int64)

    for i in range(m):
        inter = _POPCOUNT_LUT[np.bitwise_and(ref_packed, query_packed[i])].sum(
            axis=1, dtype=np.float32
        )
        union = q_counts[i] + ref_popcounts - inter
        sims = np.where(union > 0, inter / union, 0.0)
        idx = int(np.argmax(sims))
        best_idx[i] = idx
        max_sims[i] = float(sims[idx])
        k = min(5, n_ref)
        if k > 0:
            top_k = np.partition(sims, -k)[-k:]
            top5_sims[i] = float(np.mean(top_k))

    return max_sims, top5_sims, best_idx


# ---------------------------------------------------------------------------
# Production ChEMBL v37 Ensemble Loader & Predictor
# ---------------------------------------------------------------------------

class ChEMBLADMETEnsemble:
    """
    Production inference engine wrapping the scaffold-validated ChEMBL v37
    multi-modal ensemble (DMPNN + ECFP4-MLP + ExtraTrees + Linear-QSAR)
    with calibrated consensus probabilities, epistemic uncertainty, and
    Applicability Domain analysis.
    """

    def __init__(self, bundle_dir: Path, device: str = "cpu"):
        self.bundle_dir = Path(bundle_dir)
        self.device = torch.device(device)

        meta_path = self.bundle_dir / "chembl_v2_metrics.json"
        bundle_path = self.bundle_dir / "chembl_v2_bundle.joblib"
        dmpnn_path = self.bundle_dir / "dmpnn_chembl_v2.pt"
        mlp_path = self.bundle_dir / "fp_mlp_chembl_v2.pt"
        ad_path = self.bundle_dir / "chembl_v2_ad.npz"

        for p in (meta_path, bundle_path, dmpnn_path, mlp_path, ad_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing ChEMBL v2 ensemble artifact: {p}")

        self.metadata: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        bundle = joblib.load(bundle_path)

        self.weights: dict[str, dict[str, float]] = bundle["ensemble_weights"]
        self.temperatures: dict[str, dict[str, float]] = bundle["temperatures"]
        self.rf_models: dict[str, Any] = bundle["rf_models"]
        self.linear_models: dict[str, Any] = bundle["linear_models"]
        self.desc_mean: np.ndarray = bundle["desc_mean"]
        self.desc_std: np.ndarray = bundle["desc_std"]
        self.dmpnn_cfg: dict[str, Any] = bundle["dmpnn_cfg"]
        self.mlp_cfg: dict[str, Any] = bundle["mlp_cfg"]

        # Load Chemprop-grade One-Hot DMPNN
        self.dmpnn = ADMETPredictor(
            node_dim=self.dmpnn_cfg["node_dim"],
            edge_dim=self.dmpnn_cfg["edge_dim"],
            hidden_dim=self.dmpnn_cfg["hidden_dim"],
            num_tasks=2,
            T=self.dmpnn_cfg["T"],
            set2set_steps=self.dmpnn_cfg["set2set_steps"],
            dropout=self.dmpnn_cfg["dropout"],
        ).to(self.device)
        self.dmpnn.load_state_dict(
            torch.load(str(dmpnn_path), map_location=self.device, weights_only=True)
        )
        self.dmpnn.eval()

        # Load Residual ECFP4 + PhysChem MLP
        self.fp_mlp = FingerprintPhysChemNet(
            in_dim=self.mlp_cfg["in_dim"],
            hidden_dim=self.mlp_cfg["hidden_dim"],
            num_tasks=2,
            dropout=self.mlp_cfg["dropout"],
        ).to(self.device)
        self.fp_mlp.load_state_dict(
            torch.load(str(mlp_path), map_location=self.device, weights_only=True)
        )
        self.fp_mlp.eval()

        # Load Applicability Domain reference corpus
        ad_npz = np.load(ad_path, allow_pickle=False)
        self.ref_packed: np.ndarray = ad_npz["fp_packed"]
        self.ref_popcounts: np.ndarray = _POPCOUNT_LUT[self.ref_packed].sum(
            axis=1, dtype=np.float32
        )
        self.ref_smiles: list[str] = ad_npz["smiles"].tolist()
        self.ref_herg: np.ndarray = ad_npz["herg_label"]
        self.ref_cyp: np.ndarray = ad_npz["cyp3a4_label"]

    @staticmethod
    def _calibrate_prob(prob: np.ndarray, temp: float) -> np.ndarray:
        """Apply temperature scaling in logit space for calibrated probabilities."""
        p = np.clip(prob, 1e-6, 1.0 - 1e-6)
        logit = np.log(p / (1.0 - p))
        scaled = logit / max(temp, 1e-3)
        return 1.0 / (1.0 + np.exp(-np.clip(scaled, -30.0, 30.0)))

    @torch.no_grad()
    def predict_mols(self, mols: list[Chem.Mol]) -> list[dict[str, Any]]:
        """
        Run full multi-modal ensemble inference + Applicability Domain on a
        batch of standardized RDKit Mol objects.
        """
        if not mols:
            return []

        n = len(mols)
        graphs = []
        fp_rows = np.zeros((n, 2048), dtype=np.float32)
        desc_rows = np.zeros((n, 20), dtype=np.float32)

        for i, mol in enumerate(mols):
            g = mol_to_data(mol, rich=True)
            graphs.append(g)
            fp_rows[i] = compute_morgan_numpy(mol)
            desc_rows[i] = compute_rdkit_ml_descriptors(mol)

        # Standardize 20 physicochemical descriptors using training stats
        desc_scaled = (desc_rows - self.desc_mean) / self.desc_std
        desc_scaled = np.clip(desc_scaled, -8.0, 8.0)
        x_tab = np.concatenate([fp_rows, desc_scaled], axis=1)

        # 1. DMPNN forward pass
        batch = Batch.from_data_list(graphs).to(self.device)
        dmpnn_logits = self.dmpnn(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        ).cpu().numpy()
        p_dmpnn_raw = 1.0 / (1.0 + np.exp(-np.clip(dmpnn_logits, -30.0, 30.0)))

        # 2. ECFP4 + PhysChem MLP forward pass
        x_t = torch.from_numpy(x_tab).to(self.device)
        mlp_logits = self.fp_mlp(x_t).cpu().numpy()
        p_mlp_raw = 1.0 / (1.0 + np.exp(-np.clip(mlp_logits, -30.0, 30.0)))

        # 3. ExtraTrees + Linear QSAR per task
        p_rf_raw = np.zeros((n, 2), dtype=np.float32)
        p_lin_raw = np.zeros((n, 2), dtype=np.float32)
        for t_idx, short_name in enumerate(SHORT_TASKS):
            p_rf_raw[:, t_idx] = self.rf_models[short_name].predict_proba(x_tab)[:, 1]
            p_lin_raw[:, t_idx] = self.linear_models[short_name].predict_proba(x_tab)[:, 1]

        # 4. Applicability Domain via vectorized Tanimoto
        q_packed = pack_fingerprints(fp_rows)
        max_sims, top5_sims, best_indices = compute_tanimoto_batch(
            q_packed, self.ref_packed, self.ref_popcounts
        )

        # 5. Calibrated consensus + epistemic uncertainty per molecule
        outputs: list[dict[str, Any]] = []
        for i in range(n):
            endpoints: dict[str, dict[str, Any]] = {}
            flat_probs: dict[str, float] = {}

            for t_idx, (full_name, short_name) in enumerate(zip(CHEMBL_TASKS, SHORT_TASKS)):
                w = self.weights[short_name]
                temps = self.temperatures[short_name]

                m_probs = {
                    "dmpnn": float(self._calibrate_prob(p_dmpnn_raw[i, t_idx], temps["dmpnn"])),
                    "ecfp4_mlp": float(self._calibrate_prob(p_mlp_raw[i, t_idx], temps["ecfp4_mlp"])),
                    "extra_trees": float(self._calibrate_prob(p_rf_raw[i, t_idx], temps["extra_trees"])),
                    "linear_qsar": float(self._calibrate_prob(p_lin_raw[i, t_idx], temps["linear_qsar"])),
                }

                mean_p = sum(w[k] * m_probs[k] for k in m_probs)
                var_p = sum(w[k] * (m_probs[k] - mean_p) ** 2 for k in m_probs)
                std_p = float(np.sqrt(max(var_p, 0.0)))
                ci_low = float(np.clip(mean_p - 1.96 * std_p, 0.0, 1.0))
                ci_high = float(np.clip(mean_p + 1.96 * std_p, 0.0, 1.0))

                if mean_p >= 0.70:
                    risk_tier = "HIGH"
                elif mean_p >= 0.45:
                    risk_tier = "MODERATE"
                else:
                    risk_tier = "LOW"

                flat_probs[full_name] = round(mean_p, 4)
                endpoints[full_name] = {
                    "probability": round(mean_p, 4),
                    "epistemic_std": round(std_p, 4),
                    "ci_95": [round(ci_low, 4), round(ci_high, 4)],
                    "risk_tier": risk_tier,
                    "member_probabilities": {k: round(v, 4) for k, v in m_probs.items()},
                }

            sim = float(max_sims[i])
            if sim >= 0.40:
                ad_status = "In-Domain"
            elif sim >= 0.25:
                ad_status = "Borderline"
            else:
                ad_status = "Out-of-Domain"

            nn_idx = int(best_indices[i])
            nn_info: Optional[dict[str, Any]] = None
            if nn_idx >= 0:
                herg_v = self.ref_herg[nn_idx]
                cyp_v = self.ref_cyp[nn_idx]
                nn_info = {
                    "smiles": self.ref_smiles[nn_idx],
                    "tanimoto": round(sim, 3),
                    "herg_active": None if np.isnan(herg_v) else bool(herg_v >= 0.5),
                    "cyp3a4_active": None if np.isnan(cyp_v) else bool(cyp_v >= 0.5),
                }

            outputs.append({
                "predictions": flat_probs,
                "ml_endpoints": endpoints,
                "applicability_domain": {
                    "status": ad_status,
                    "max_tanimoto": round(sim, 3),
                    "mean_top5_tanimoto": round(float(top5_sims[i]), 3),
                    "nearest_neighbor": nn_info,
                },
            })

        return outputs

"""
ADMET-Engine v3.0 (CardioPhys-3D) — WP3: Censored Tobit Likelihood &
Mondrian Conformal Prediction
====================================================================
1. Censored Gaussian Tobit Negative Log-Likelihood (`tobit_censored_nll`):
   Properly incorporates exact ('='), left-censored ('>' IC50 -> '<' pIC50),
   and right-censored ('<' IC50 -> '>' pIC50) bioactivity assays without
   discarding valuable inactive or ultra-potent screening bounds:
     - Exact (censor = 0):
         -log N(y | mu, sigma^2) = 0.5 * log(2*pi*sigma^2) + 0.5 * ((y - mu)/sigma)^2
     - Left-censored (censor = -1, true pIC50 <= y):
         -log Phi((y - mu) / sigma)
     - Right-censored (censor = +1, true pIC50 >= y):
         -log Phi((mu - y) / sigma)

   Uses `torch.special.log_ndtr` for exact asymptotic stability down to z = -30.

2. Mondrian Conformal Prediction (`MondrianConformalRegressor`):
   Provides distribution-free, finite-sample coverage guarantees:
       P(y_true in [mu - q_{g,alpha} * sigma, mu + q_{g,alpha} * sigma] | group = g) >= 1 - alpha
   stratified per Applicability Domain regime g in {In-Domain, Borderline, Out-of-Domain}.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Censored Gaussian Tobit Loss
# ---------------------------------------------------------------------------

def tobit_censored_nll(
    mu: Tensor,
    sigma: Tensor,
    targets: Tensor,
    censor: Tensor,
    sigma_floor: float = 0.05,
    channel_weights: Tensor | None = None,
) -> Tensor:
    """
    Compute NaN-masked multi-channel censored Gaussian Tobit negative log-likelihood.

    Args:
        mu:              [B, C] predicted pIC50 means
        sigma:           [B, C] predicted pIC50 standard deviations (> 0)
        targets:         [B, C] observed pIC50 values or bounds (NaN where missing)
        censor:          [B, C] censoring codes (0=exact, -1=left/upper-bound, +1=right/lower-bound)
        sigma_floor:     Minimum std clamp for numerical stability
        channel_weights: Optional [C] per-channel loss weighting tensor
    """
    valid = ~torch.isnan(targets) & ~torch.isnan(censor)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=mu.device, dtype=mu.dtype, requires_grad=True)

    sig = torch.clamp(sigma, min=sigma_floor)
    y_clean = torch.where(valid, targets, torch.zeros_like(targets))
    c_clean = torch.where(valid, censor, torch.zeros_like(censor)).long()

    z = (y_clean - mu) / sig

    # Exact: 0.5 * log(2*pi) + log(sigma) + 0.5 * z^2
    loss_exact = HALF_LOG_2PI + torch.log(sig) + 0.5 * (z ** 2)

    # Left-censored (true pIC50 <= y): -log Phi((y - mu) / sigma)
    loss_left = -torch.special.log_ndtr(z)

    # Right-censored (true pIC50 >= y): -log Phi((mu - y) / sigma) = -log Phi(-z)
    loss_right = -torch.special.log_ndtr(-z)

    loss_mat = torch.where(
        c_clean == 0,
        loss_exact,
        torch.where(c_clean < 0, loss_left, loss_right),
    )

    if channel_weights is not None:
        w = channel_weights.to(device=mu.device, dtype=mu.dtype).view(1, -1)
        loss_mat = loss_mat * w

    return loss_mat[valid].mean()


# ---------------------------------------------------------------------------
# Evaluation Metrics for Censored Regression
# ---------------------------------------------------------------------------

def censored_concordance_rate(
    mu: np.ndarray,
    targets: np.ndarray,
    censor: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate how accurately predicted means respect censored inequality bounds:
      - Left-censored  (censor == -1, true pIC50 <= bound): satisfied if mu <= bound
      - Right-censored (censor == +1, true pIC50 >= bound): satisfied if mu >= bound
    """
    left_m = (~np.isnan(targets)) & (censor == -1)
    right_m = (~np.isnan(targets)) & (censor == +1)

    res: dict[str, float] = {}
    if left_m.sum() > 0:
        res["left_agreement"] = float(np.mean(mu[left_m] <= targets[left_m]))
        res["n_left"] = int(left_m.sum())
    if right_m.sum() > 0:
        res["right_agreement"] = float(np.mean(mu[right_m] >= targets[right_m]))
        res["n_right"] = int(right_m.sum())

    total_c = int(left_m.sum() + right_m.sum())
    if total_c > 0:
        ok = int(np.sum(mu[left_m] <= targets[left_m])) + int(np.sum(mu[right_m] >= targets[right_m]))
        res["overall_censored_agreement"] = ok / total_c
        res["n_censored"] = total_c
    return res


# ---------------------------------------------------------------------------
# Mondrian Conformal Predictor (Applicability-Domain Stratified)
# ---------------------------------------------------------------------------

class MondrianConformalRegressor:
    """
    Split Mondrian Conformal Predictor for heteroscedastic pIC50 regression.

    Calibrates normalized nonconformity scores:
        s_i = |y_i - mu_i| / sigma_i
    separately within each Applicability Domain group g (e.g., 'In-Domain',
    'Borderline', 'Out-of-Domain'), falling back to the global quantile when
    a stratum has fewer than `min_group_size` calibration examples.
    """

    def __init__(self, alpha: float = 0.10, min_group_size: int = 15):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must lie strictly in (0, 1)")
        self.alpha = alpha
        self.min_group_size = min_group_size
        self.global_quantile: float = 1.96
        self.group_quantiles: dict[str, float] = {}

    @staticmethod
    def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
        n = len(scores)
        if n == 0:
            return 1.96
        level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
        return float(np.quantile(scores, level, method="higher"))

    def fit(
        self,
        y_true: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        groups: list[str] | np.ndarray,
    ) -> "MondrianConformalRegressor":
        """Calibrate Mondrian quantiles on exact validation observations."""
        valid = (~np.isnan(y_true)) & (~np.isnan(mu)) & (sigma > 0)
        y_v = y_true[valid]
        mu_v = mu[valid]
        sig_v = np.maximum(sigma[valid], 1e-4)
        grp_v = np.asarray(groups)[valid]

        scores = np.abs(y_v - mu_v) / sig_v
        self.global_quantile = self._finite_sample_quantile(scores, self.alpha)

        self.group_quantiles = {}
        for g in np.unique(grp_v):
            g_scores = scores[grp_v == g]
            if len(g_scores) >= self.min_group_size:
                self.group_quantiles[str(g)] = self._finite_sample_quantile(g_scores, self.alpha)
            else:
                self.group_quantiles[str(g)] = self.global_quantile
        return self

    def predict_intervals(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        groups: list[str] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower_bound, upper_bound) guaranteed at nominal 1 - alpha coverage."""
        q = np.array(
            [self.group_quantiles.get(str(g), self.global_quantile) for g in groups],
            dtype=np.float32,
        )
        half_width = q * np.maximum(sigma, 1e-4)
        return mu - half_width, mu + half_width

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "nominal_coverage": round(1.0 - self.alpha, 4),
            "global_quantile": round(self.global_quantile, 4),
            "group_quantiles": {k: round(v, 4) for k, v in self.group_quantiles.items()},
        }

"""
ADMET-Engine — ChEMBL v37 Training Pipeline
Phase 7.2: Real multi-task training on hERG + CYP3A4

Trains the DMPNN on 16,321 ChEMBL molecules with 2 binary targets.
Uses BCEWithLogitsLoss with NaN masking for missing labels.
Early stopping after 5 epochs without validation improvement.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from dataset import create_dataloaders, ENDPOINT_LABELS, NUM_ENDPOINTS
from models.dmpnn import ADMETPredictor

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("admet.train")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data/processed/chembl_admet_training.csv"
MODELS_DIR = PROJECT_ROOT / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "hidden_dim": 300,
    "num_tasks": NUM_ENDPOINTS,
    "T": 3,
    "set2set_steps": 3,
    "dropout": 0.2,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "batch_size": 64,
    "max_epochs": 30,
    "patience": 5,
    "grad_clip": 5.0,
}


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def masked_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCEWithLogitsLoss ignoring NaN positions."""
    mask = ~torch.isnan(targets)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    targets_clean = torch.where(mask, targets, torch.zeros_like(targets))
    loss = F.binary_cross_entropy_with_logits(
        logits[mask], targets_clean[mask], reduction="mean",
    )
    return loss


def train_epoch(
    model: ADMETPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = masked_bce(logits, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: ADMETPredictor,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = masked_bce(logits, batch.y)
        total_loss += loss.item()
        n_batches += 1
        all_logits.append(logits.cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    y_pred = np.concatenate(all_logits, axis=0)  # [N, 2]
    y_true = np.concatenate(all_targets, axis=0)  # [N, 2]

    metrics: dict[str, float] = {}
    for i, name in enumerate(ENDPOINT_LABELS):
        mask = ~np.isnan(y_true[:, i])
        if mask.sum() > 1 and len(np.unique(y_true[mask, i])) > 1:
            try:
                auc = roc_auc_score(y_true[mask, i], y_pred[mask, i])
            except ValueError:
                auc = float("nan")
            metrics[f"{name}_roc_auc"] = auc
            metrics[f"{name}_n_samples"] = float(mask.sum())
        else:
            metrics[f"{name}_roc_auc"] = float("nan")
            metrics[f"{name}_n_samples"] = float(mask.sum())

    # Global ROC-AUC (concatenating both tasks)
    flat_mask = ~np.isnan(y_true)
    if flat_mask.sum() > 1 and len(np.unique(y_true[flat_mask])) > 1:
        metrics["global_roc_auc"] = roc_auc_score(y_true[flat_mask], y_pred[flat_mask])
    else:
        metrics["global_roc_auc"] = float("nan")

    return avg_loss, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Device: %s", DEVICE)
    logger.info("Config: %s", CONFIG)

    # --- Data ---------------------------------------------------------
    train_loader, val_loader, test_loader, info = create_dataloaders(
        CSV_PATH,
        batch_size=CONFIG["batch_size"],
        split_ratios=(0.8, 0.1, 0.1),
        seed=42,
    )
    logger.info(
        "Data: %d train / %d val / %d test  (%d targets)",
        info["n_train"], info["n_val"], info["n_test"], info["num_endpoints"],
    )

    # --- Model --------------------------------------------------------
    model = ADMETPredictor(
        node_dim=9, edge_dim=4, hidden_dim=CONFIG["hidden_dim"],
        num_tasks=CONFIG["num_tasks"], T=CONFIG["T"],
        set2set_steps=CONFIG["set2set_steps"], dropout=CONFIG["dropout"],
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d parameters on %s", n_params, DEVICE)

    optimizer = Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # --- MLflow -------------------------------------------------------
    mlflow_run = None
    if HAS_MLFLOW:
        mlflow.set_experiment("admet-engine-chembl")
        mlflow_run = mlflow.start_run(run_name=f"chembl_v1_{int(time.time())}")
        mlflow.log_params(CONFIG)
        logger.info("MLflow tracking active")

    try:
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        t_train = time.perf_counter()

        for epoch in range(1, CONFIG["max_epochs"] + 1):
            t0 = time.perf_counter()
            train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
            val_loss, val_metrics = evaluate(model, val_loader, DEVICE)
            elapsed = time.perf_counter() - t0

            scheduler.step(val_loss)

            auc_str = " | ".join(
                f"{name}={val_metrics.get(f'{name}_roc_auc', 0):.3f}"
                for name in ENDPOINT_LABELS
            )
            logger.info(
                "Epoch %2d | train_loss=%.4f | val_loss=%.4f | %s | %.1fs",
                epoch, train_loss, val_loss, auc_str, elapsed,
            )

            if HAS_MLFLOW and mlflow_run:
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }, step=epoch)

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                MODELS_DIR.mkdir(exist_ok=True)
                torch.save(model.state_dict(), str(MODELS_DIR / "dmpnn_chembl_v1.pt"))
                logger.info("  >>> best model saved (val_loss=%.4f)", best_val_loss)
            else:
                patience_counter += 1

            if patience_counter >= CONFIG["patience"]:
                logger.info("Early stopping after %d epochs (patience=%d)", epoch, CONFIG["patience"])
                break

        total_train_time = time.perf_counter() - t_train

        # --- Load best checkpoint -----------------------------------
        best_path = MODELS_DIR / "dmpnn_chembl_v1.pt"
        if best_path.exists():
            model.load_state_dict(torch.load(str(best_path), map_location=DEVICE, weights_only=True))
            logger.info("Loaded best checkpoint (epoch %d, val_loss=%.4f)", best_epoch, best_val_loss)

        # --- Final test evaluation ----------------------------------
        test_loss, test_metrics = evaluate(model, test_loader, DEVICE)
        logger.info("=" * 60)
        logger.info("FINAL TEST RESULTS")
        logger.info("=" * 60)
        for name in ENDPOINT_LABELS:
            auc = test_metrics.get(f"{name}_roc_auc", float("nan"))
            n = test_metrics.get(f"{name}_n_samples", 0)
            logger.info("  %-8s  ROC-AUC: %.4f  (n=%d)", name, auc, int(n))
        logger.info("  %-8s  ROC-AUC: %.4f", "GLOBAL", test_metrics.get("global_roc_auc", float("nan")))
        logger.info("  Test loss: %.4f", test_loss)
        logger.info("  Total train time: %.1f s  |  Best epoch: %d", total_train_time, best_epoch)
        logger.info("  Model saved: %s", best_path)

        if HAS_MLFLOW and mlflow_run:
            mlflow.log_metrics({
                "test_loss": test_loss,
                **{f"test_{k}": v for k, v in test_metrics.items()},
            })

    finally:
        if HAS_MLFLOW and mlflow_run:
            mlflow.end_run()


if __name__ == "__main__":
    main()

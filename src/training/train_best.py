"""
ADMET-Engine Phase 7.3 STEP 4: Train with best hyperparameters.
Loads models/best_hparams.json, trains with best config (max 50 epochs, patience=10).
Saves models/dmpnn_chembl_v2.pt. Outputs test report.
"""

import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

# -- Flushed safe logger
def log(msg: str) -> None:
    safe = msg.encode("ascii", errors="replace").decode("ascii")
    ts = time.strftime("%H:%M:%S")
    print(f"{ts} {safe}", flush=True)

# -- Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from dataset import create_dataloaders, ENDPOINT_LABELS
from training.hparam_search import FlexiblePredictor, masked_bce_loss

CSV_PATH = PROJECT_ROOT / "data" / "processed" / "chembl_admet_training.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

ENDPOINT_LABELS = ["hERG", "CYP3A4"]
NUM_TASKS = 2

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_logits, all_targets = [], []
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = masked_bce_loss(logits, batch.y)
        total_loss += loss.item()
        n_batches += 1
        all_logits.append(logits.numpy())
        all_targets.append(batch.y.numpy())

    y_pred = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    metrics = {"avg_loss": total_loss / max(n_batches, 1)}

    for i, name in enumerate(ENDPOINT_LABELS):
        mask = ~np.isnan(y_true[:, i])
        n = mask.sum()
        metrics[f"{name}_n"] = int(n)
        if n > 1 and len(np.unique(y_true[mask, i])) > 1:
            try:
                metrics[f"{name}_roc_auc"] = roc_auc_score(y_true[mask, i], y_pred[mask, i])
            except ValueError:
                metrics[f"{name}_roc_auc"] = float("nan")
        else:
            metrics[f"{name}_roc_auc"] = float("nan")
    return metrics

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log("=== PHASE 7.3 STEP 4: Train with Best HParams ===")

    # -- Load best config
    hparams_path = MODELS_DIR / "best_hparams.json"
    if not hparams_path.exists():
        log(f"ERROR: {hparams_path} not found. Run hparam_search.py first.")
        sys.exit(1)

    best = json.loads(hparams_path.read_text())
    params = best["params"]
    log(f"Loaded best config: score={best['best_val_score']:.4f}  "
        f"hERG={best['best_hERG_auc']:.4f}  CYP3A4={best['best_CYP3A4_auc']:.4f}")
    log(f"Params: {params}")

    # -- Data
    train_loader, val_loader, test_loader, info = create_dataloaders(
        CSV_PATH, batch_size=params.get("batch_size", 64),
        split_ratios=(0.8, 0.1, 0.1), seed=SEED,
    )
    log(f"Data: {info['n_train']} train / {info['n_val']} val / {info['n_test']} test")

    # -- Model
    model = FlexiblePredictor(
        node_dim=9, edge_dim=4,
        hidden_dim=params["hidden_dim"],
        num_tasks=NUM_TASKS,
        T=params["depth"],
        readout=params["readout"],
        set2set_steps=3,
        dropout=params["dropout"],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model: {n_params:,} params")

    # -- Optimizer
    lr = params["lr"]
    weight_decay = params.get("weight_decay", 0.0)
    w_herg = params.get("loss_weight_herg", 0.6)
    use_pos_weight = params.get("use_pos_weight", False)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    grad_clip = 5.0
    max_epochs = 50
    patience = 10

    # -- Pos weights
    pos_weight = None
    if use_pos_weight:
        pos = torch.zeros(NUM_TASKS)
        neg = torch.zeros(NUM_TASKS)
        for batch in train_loader:
            for t in range(NUM_TASKS):
                y_t = batch.y[:, t]
                mask = ~torch.isnan(y_t)
                pos[t] += (y_t[mask] == 1.0).sum()
                neg[t] += (y_t[mask] == 0.0).sum()
        pos_weight = torch.where(pos > 0, neg / pos, torch.ones(NUM_TASKS))
        log(f"Pos weights: hERG={pos_weight[0]:.2f} CYP3A4={pos_weight[1]:.2f}")

    # -- Training
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    t_start = time.perf_counter()

    log("=" * 60)
    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()

        # Train
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = torch.tensor(0.0)
            for t in range(NUM_TASKS):
                pw_t = pos_weight[t] if pos_weight is not None else None
                loss = loss + (w_herg if t == 0 else 1.0 - w_herg) * masked_bce_loss(
                    logits[:, t], batch.y[:, t], pw_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # Validate
        val_metrics = evaluate(model, val_loader)
        val_loss = val_metrics["avg_loss"]
        scheduler.step(val_loss)

        elapsed = time.perf_counter() - t0
        h_auc = val_metrics.get("hERG_roc_auc", float("nan"))
        c_auc = val_metrics.get("CYP3A4_roc_auc", float("nan"))
        log(f"Epoch {epoch:2d} | val_loss={val_loss:.4f} | "
            f"hERG={h_auc:.4f} | CYP3A4={c_auc:.4f} | {elapsed:.0f}s")

        # Checkpoint
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            ckpt_path = MODELS_DIR / "dmpnn_chembl_v2.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_hERG_auc": h_auc,
                "val_CYP3A4_auc": c_auc,
                "hparams": params,
            }, str(ckpt_path))
            log(f"  >> best model saved ({ckpt_path})")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            log(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    total_time = time.perf_counter() - t_start

    # -- Load best checkpoint
    best_ckpt = MODELS_DIR / "dmpnn_chembl_v2.pt"
    state = torch.load(str(best_ckpt), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    log(f"Loaded best checkpoint (epoch {best_epoch}, val_loss={best_val_loss:.4f})")

    # -- Final test evaluation
    test_metrics = evaluate(model, test_loader)
    log("=" * 60)
    log("FINAL TEST RESULTS (dmpnn_chembl_v2)")
    log("=" * 60)
    for name in ENDPOINT_LABELS:
        auc = test_metrics.get(f"{name}_roc_auc", float("nan"))
        n = test_metrics.get(f"{name}_n", 0)
        log(f"  {name:8s}  ROC-AUC: {auc:.4f}  (n={n})")
    log(f"  Test loss:       {test_metrics['avg_loss']:.4f}")
    log(f"  Train time:      {total_time:.0f}s ({total_time/60:.1f} min)")
    log(f"  Best epoch:      {best_epoch}")
    log(f"  Saved:           {best_ckpt}")

    # -- Save results JSON
    results = {
        "model": "dmpnn_chembl_v2",
        "hERG_roc_auc": test_metrics.get("hERG_roc_auc"),
        "CYP3A4_roc_auc": test_metrics.get("CYP3A4_roc_auc"),
        "test_loss": test_metrics["avg_loss"],
        "hparams": params,
        "train_time_s": total_time,
        "best_epoch": best_epoch,
        "n_params": n_params,
    }
    out_json = RESULTS_DIR / "phase73_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {out_json}")

    log("=== STEP 4: DONE ===")

if __name__ == "__main__":
    main()

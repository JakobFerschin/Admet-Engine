"""
ADMET-Engine — Phase 7.3: Optuna Hyperparameter Search
CPU-optimised, Windows-compatible, crash-resilient.

Searches over hidden_dim, depth (T), dropout, readout type, lr,
weight_decay, batch_size, loss_weight, and pos_weight strategy.
Uses MedianPruner to abort bad trials early.
Saves best config to models/best_hparams.json.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Set2Set, global_add_pool, global_mean_pool

# -- Safe logger for Windows cp1252 ------------------------------------------
def log(msg: str) -> None:
    safe = msg.encode("ascii", errors="replace").decode("ascii")
    ts = time.strftime("%H:%M:%S")
    print(f"{ts} {safe}", flush=True)
    sys.stdout.flush()

# -- Project paths ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "chembl_admet_training.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")
SEED = 42
ENDPOINT_LABELS = ["hERG", "CYP3A4"]
NUM_TASKS = 2

# -- Seeding ------------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -- Imports from src ---------------------------------------------------------
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from dataset import create_dataloaders, _load_molecules

# ===========================================================================
# Flexible ADMET Predictor (supports mean / sum / set2set readout)
# ===========================================================================

class FlexiblePredictor(nn.Module):
    def __init__(
        self,
        node_dim: int = 9,
        edge_dim: int = 4,
        hidden_dim: int = 300,
        num_tasks: int = 2,
        T: int = 3,
        readout: str = "set2set",
        set2set_steps: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.readout_type = readout

        # -- DMPNN encoder -- reused from existing code
        from models.dmpnn import DMPNNEncoder
        self.encoder = DMPNNEncoder(
            node_dim=node_dim, edge_dim=edge_dim,
            hidden_dim=hidden_dim, T=T, dropout=dropout,
        )

        if readout == "set2set":
            readout_dim = 2 * hidden_dim
            self.readout = Set2Set(
                in_channels=hidden_dim, processing_steps=set2set_steps, num_layers=1,
            )
        else:
            readout_dim = hidden_dim
            self.readout = None  # global pool used in forward

        self.dropout_layer = nn.Dropout(dropout)
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(readout_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_tasks)
        ])

    def forward(self, x, edge_index, edge_attr, batch):
        atom_emb = self.encoder(x, edge_index, edge_attr)

        if self.readout_type == "set2set":
            mol_emb = self.readout(atom_emb, batch)
        elif self.readout_type == "mean":
            mol_emb = global_mean_pool(atom_emb, batch)
        elif self.readout_type == "sum":
            mol_emb = global_add_pool(atom_emb, batch)
        else:
            raise ValueError(f"Unknown readout: {self.readout_type}")

        mol_emb = self.dropout_layer(mol_emb)
        preds = [head(mol_emb) for head in self.task_heads]
        return torch.cat(preds, dim=-1)


# ===========================================================================
# Training utilities
# ===========================================================================

def masked_bce_loss(pred, target, pos_weight=None):
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    pred_m = pred[mask]
    target_m = torch.where(mask, target, torch.zeros_like(target))[mask]
    return F.binary_cross_entropy_with_logits(pred_m, target_m, pos_weight=pos_weight)


def weighted_auc(val_herg, val_cyp, w_herg=0.6, w_cyp=0.4):
    obj = 0.0
    if not np.isnan(val_herg):
        obj += w_herg * val_herg
    if not np.isnan(val_cyp):
        obj += w_cyp * val_cyp
    # Normalise if only one is valid
    total = 0.0
    if not np.isnan(val_herg):
        total += w_herg
    if not np.isnan(val_cyp):
        total += w_cyp
    return obj / total if total > 0 else 0.0


@torch.no_grad()
def quick_eval(model, loader):
    model.eval()
    all_logits, all_targets = [], []
    for batch in loader:
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        all_logits.append(logits.numpy())
        all_targets.append(batch.y.numpy())

    y_pred = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    aucs = {}
    for i, name in enumerate(ENDPOINT_LABELS):
        mask = ~np.isnan(y_true[:, i])
        if mask.sum() > 1 and len(np.unique(y_true[mask, i])) > 1:
            try:
                aucs[f"{name}_roc_auc"] = roc_auc_score(y_true[mask, i], y_pred[mask, i])
            except ValueError:
                aucs[f"{name}_roc_auc"] = float("nan")
        else:
            aucs[f"{name}_roc_auc"] = float("nan")
    return aucs


# ===========================================================================
# Trial runner
# ===========================================================================

def _compute_pos_weights(train_loader):
    """pos_weight = (#inactive / #active) per task."""
    pos = torch.zeros(NUM_TASKS)
    neg = torch.zeros(NUM_TASKS)
    for batch in train_loader:
        for t in range(NUM_TASKS):
            y_t = batch.y[:, t]
            mask = ~torch.isnan(y_t)
            pos[t] += (y_t[mask] == 1.0).sum()
            neg[t] += (y_t[mask] == 0.0).sum()
    pw = torch.zeros(NUM_TASKS)
    for t in range(NUM_TASKS):
        if pos[t] > 0:
            pw[t] = neg[t] / pos[t]
    return pw


def run_trial(trial, train_loader, val_loader):
    """Optuna objective: returns weighted validation AUC."""
    import optuna

    # -- Sample hyperparameters ------------------------------------------
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 300, 400, 512])
    depth = trial.suggest_int("depth", 2, 5)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    readout = trial.suggest_categorical("readout", ["mean", "sum", "set2set"])
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 1e-3)
    # batch_size is sampled in objective() and passed via DataLoader
    w_herg = trial.suggest_float("loss_weight_herg", 0.5, 0.8)
    use_pos_weight = trial.suggest_categorical("use_pos_weight", [True, False])

    max_epochs = 8
    patience = 4
    grad_clip = 5.0

    # -- Build model ----------------------------------------------------
    model = FlexiblePredictor(
        node_dim=9, edge_dim=4, hidden_dim=hidden_dim,
        num_tasks=NUM_TASKS, T=depth, readout=readout,
        set2set_steps=3, dropout=dropout,
    ).to(DEVICE)

    # -- Pos weights ----------------------------------------------------
    pos_weight = None
    if use_pos_weight:
        pw = _compute_pos_weights(train_loader)
        pos_weight = pw

    # -- Optimizer ------------------------------------------------------
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_score = -1.0
    patience_counter = 0
    t_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        # -- Train one epoch --------------------------------------------
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = torch.tensor(0.0)
            for t in range(NUM_TASKS):
                pw_t = pos_weight[t] if pos_weight is not None else None
                loss = loss + (w_herg if t == 0 else 1.0 - w_herg) * masked_bce_loss(
                    logits[:, t], batch.y[:, t], pw_t,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # -- Validate ---------------------------------------------------
        model.eval()
        val_loss = 0.0
        n_batches = 0
        all_logits, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                batch_loss = torch.tensor(0.0)
                for t in range(NUM_TASKS):
                    pw_t = pos_weight[t] if pos_weight is not None else None
                    batch_loss = batch_loss + masked_bce_loss(
                        logits[:, t], batch.y[:, t], pw_t,
                    )
                val_loss += batch_loss.item()
                n_batches += 1
                all_logits.append(logits.numpy())
                all_targets.append(batch.y.numpy())

        val_loss = val_loss / max(n_batches, 1)
        y_pred = np.concatenate(all_logits, axis=0)
        y_true = np.concatenate(all_targets, axis=0)

        herg_auc, cyp_auc = float("nan"), float("nan")
        for i, name in enumerate(ENDPOINT_LABELS):
            mask = ~np.isnan(y_true[:, i])
            if mask.sum() > 1 and len(np.unique(y_true[mask, i])) > 1:
                try:
                    auc = roc_auc_score(y_true[mask, i], y_pred[mask, i])
                    if i == 0:
                        herg_auc = auc
                    else:
                        cyp_auc = auc
                except ValueError:
                    pass

        score = weighted_auc(herg_auc, cyp_auc, 0.6, 0.4)

        scheduler.step(val_loss)

        # -- Optuna pruning ---------------------------------------------
        trial.report(score, epoch)
        if trial.should_prune():
            elapsed = time.perf_counter() - t_start
            log(f"  Trial {trial.number:2d}: PRUNED at epoch {epoch} | "
                f"hERG={herg_auc:.4f} CYP={cyp_auc:.4f} score={score:.4f} | {elapsed:.0f}s")
            raise optuna.exceptions.TrialPruned()

        if score > best_score:
            best_score = score
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    elapsed = time.perf_counter() - t_start
    trial.set_user_attr("hERG_auc", float(herg_auc))
    trial.set_user_attr("CYP3A4_auc", float(cyp_auc))
    trial.set_user_attr("wall_clock_s", float(elapsed))
    trial.set_user_attr("epochs_run", epoch)

    log(f"  Trial {trial.number:2d}: h={hidden_dim} d={depth} drop={dropout:.2f} "
        f"ro={readout} lr={lr:.1e} wd={weight_decay:.1e} bs={batch_size} "
        f"-> hERG={herg_auc:.4f} CYP={cyp_auc:.4f} score={score:.4f} [{elapsed:.0f}s]")

    return best_score


# ===========================================================================
# Main
# ===========================================================================

def main():
    import optuna

    # -- Silence optuna + rdkit -------------------------------------------------
    logging.getLogger("optuna").setLevel(logging.WARNING)
    logging.getLogger("rdkit").setLevel(logging.ERROR)

    log("=== PHASE 7.3: Optuna Hyperparameter Search ===")

    # -- Load data ---------------------------------------------------------------
    log("Loading and featurising data...")
    t0 = time.perf_counter()
    all_data = _load_molecules(CSV_PATH)
    n = len(all_data)
    log(f"Featurised {n} molecules ({time.perf_counter()-t0:.0f}s)")

    # -- Split (same seed as Phase 7.2) ------------------------------------------
    rng = random.Random(SEED)
    shuffled = list(all_data)
    rng.shuffle(shuffled)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_data = shuffled[:n_train]
    val_data = shuffled[n_train:n_train + n_val]

    log(f"Split: {n_train} train / {len(val_data)} val (fixed seed={SEED})")

    # -- Compute pos weights once ------------------------------------------------
    # We need train loader for this
    probe_loader = DataLoader(train_data, batch_size=512, shuffle=False)
    pw = _compute_pos_weights(probe_loader)
    log(f"Class balance: hERG pos/neg={pw[0]:.2f}, CYP3A4 pos/neg={pw[1]:.2f}")

    # -- Pre-built dataloaders (batch_size will vary per trial) -------------------
    # For efficiency, we use fixed data and re-wrap with trial batch_size.
    # Build all three at default batch_size for reusability.

    # -- Optuna study ------------------------------------------------------------
    log("Creating Optuna study (MedianPruner, n_trials=50)...")

    # Use in-memory to avoid optuna SQLite bugs on Windows.
    # Manual save via trial callbacks.
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        storage=None,
    )

    # Periodic save callback
    def _save_progress(_study, _trial):
        if _trial.state == optuna.trial.TrialState.COMPLETE:
            completed = [t for t in _study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            # Save all completed trials for resume
            save_data = []
            for t in completed:
                save_data.append({
                    "number": t.number,
                    "value": float(t.value) if t.value is not None else None,
                    "params": t.params,
                    "hERG_auc": float(t.user_attrs.get("hERG_auc", 0)),
                    "CYP3A4_auc": float(t.user_attrs.get("CYP3A4_auc", 0)),
                    "wall_clock_s": float(t.user_attrs.get("wall_clock_s", 0)),
                })
            with open(MODELS_DIR / "phase73_trials.json", "w") as f:
                json.dump(save_data, f, indent=2)
            # Also save best config snapshot
            if _study.best_trial:
                best_cfg = {
                    "best_val_score": float(_study.best_trial.value),
                    "best_hERG_auc": float(_study.best_trial.user_attrs.get("hERG_auc", 0)),
                    "best_CYP3A4_auc": float(_study.best_trial.user_attrs.get("CYP3A4_auc", 0)),
                    "params": _study.best_trial.params,
                    "n_completed": len(completed),
                    "n_total": len(_study.trials),
                }
                with open(MODELS_DIR / "best_hparams.json", "w") as f:
                    json.dump(best_cfg, f, indent=2)
            log(f"[saved] {len(completed)}/{len(_study.trials)} trials saved to phase73_trials.json")

    # -- Context manager for trial execution ------------------------------------
    def objective(trial):
        bs = trial.suggest_categorical("batch_size", [32, 64, 128])
        train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=bs, shuffle=False)
        return run_trial(trial, train_loader, val_loader)

    log("Starting 50 trials (max 15 epochs each, MedianPruner active)...")
    log("=" * 70)

    t_start_all = time.perf_counter()

    try:
        study.optimize(objective, n_trials=30, show_progress_bar=False, callbacks=[_save_progress])
    except KeyboardInterrupt:
        log("Interrupted by user.")
    except Exception as e:
        log(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    total_time = time.perf_counter() - t_start_all
    log("=" * 70)
    log(f"Study complete. {len(study.trials)} trials in {total_time:.0f}s "
        f"({total_time/60:.1f} min)")

    # -- Top-5 trials ----------------------------------------------------------
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)

    log("\nTop-5 trials:")
    log("-" * 70)
    for i, t in enumerate(completed[:5]):
        p = t.params
        ua = t.user_attrs
        log(f"  #{i+1}: val_score={t.value:.4f}  hERG={ua.get('hERG_auc','?'):.4f} "
            f"CYP3A4={ua.get('CYP3A4_auc','?'):.4f}  "
            f"h={p['hidden_dim']} d={p['depth']} drop={p['dropout']:.2f} "
            f"ro={p['readout']} lr={p['lr']:.1e} bs={p['batch_size']} "
            f"wd={p['weight_decay']:.1e} pw={p.get('use_pos_weight',False)} "
            f"[{ua.get('wall_clock_s',0):.0f}s]")

    # -- Save best config -------------------------------------------------------
    if study.best_trial:
        best = study.best_trial
        best_config = {
            "best_val_score": float(best.value),
            "best_hERG_auc": float(best.user_attrs.get("hERG_auc", 0)),
            "best_CYP3A4_auc": float(best.user_attrs.get("CYP3A4_auc", 0)),
            "params": best.params,
            "seed": SEED,
            "n_trials": len(study.trials),
            "wall_clock_total_s": float(total_time),
        }
        out_path = MODELS_DIR / "best_hparams.json"
        with open(out_path, "w") as f:
            json.dump(best_config, f, indent=2)
        log(f"\nBest config saved to {out_path}")
        log(f"  config: {best.params}")

    log("\n=== Phase 7.3 hparam_search.py: DONE ===")


if __name__ == "__main__":
    main()

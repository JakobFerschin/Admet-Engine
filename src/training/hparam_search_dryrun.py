"""Dry-run: 2 trials x 3 epochs to verify no crashes before full 50-trial run."""
from pathlib import Path
import sys, random, numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Monkey-patch the main to reduce trials/epochs
import importlib
spec = importlib.util.spec_from_file_location("hparam_search", __file__.replace("_dryrun", ""))
mod = importlib.util.module_from_spec(spec)

# Override before import
import optuna
import logging
logging.getLogger("optuna").setLevel(logging.WARNING)
logging.getLogger("rdkit").setLevel(logging.ERROR)

def log(msg):
    safe = msg.encode("ascii", errors="replace").decode("ascii")
    import time
    print(f"{time.strftime('%H:%M:%S')} {safe}", flush=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "chembl_admet_training.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from dataset import _load_molecules
from torch_geometric.loader import DataLoader
from hparam_search import FlexiblePredictor, run_trial, masked_bce_loss, weighted_auc, _compute_pos_weights

log("=== DRY RUN: 2 trials, 3 epochs ===")

t0 = __import__("time").perf_counter()
all_data = _load_molecules(CSV_PATH)
n = len(all_data)
log(f"Featurised {n} molecules ({__import__('time').perf_counter()-t0:.0f}s)")

rng = random.Random(SEED)
shuffled = list(all_data)
rng.shuffle(shuffled)
n_train = int(n * 0.8)
n_val = int(n * 0.1)
train_data = shuffled[:n_train]
val_data = shuffled[n_train:n_train + n_val]

# Monkey-patch run_trial to limit epochs
_orig_run_trial = run_trial
def quick_trial(trial, tl, vl):
    import types
    # Patch the inner max_epochs
    src = __import__("inspect").getsource(_orig_run_trial)
    # Simpler: just run 3 epochs via a wrapper
    return _orig_run_trial(trial, tl, vl)

# Actually let's just call the real function - it already limits to 15 epochs
# and pruning will handle it. But we want to force max 3 epochs.
# Let's do a simpler approach: modify trial params temporarily.

# Easier: just write a standalone quick test
import types

def quick_objective(trial):
    bs = trial.suggest_categorical("batch_size", [32, 64, 128])
    tl = DataLoader(train_data, batch_size=bs, shuffle=True)
    vl = DataLoader(val_data, batch_size=bs, shuffle=False)
    # Run with max 3 epochs
    return _quick_trial_3e(trial, tl, vl)

def _quick_trial_3e(trial, tl, vl):
    """Same as run_trial but hard-coded 3 epochs."""
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 300, 512, 600])
    depth = trial.suggest_int("depth", 2, 5)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    readout = trial.suggest_categorical("readout", ["mean", "sum", "set2set"])
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    w_herg = trial.suggest_float("loss_weight_herg", 0.5, 0.8)
    use_pos_weight = trial.suggest_categorical("use_pos_weight", [True, False])

    max_epochs = 3  # DRY RUN
    grad_clip = 5.0

    model = FlexiblePredictor(
        node_dim=9, edge_dim=4, hidden_dim=hidden_dim,
        num_tasks=2, T=depth, readout=readout, set2set_steps=3, dropout=dropout,
    ).to(DEVICE)

    pos_weight = None
    if use_pos_weight:
        pw = _compute_pos_weights(tl)
        pos_weight = pw

    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    best_score = -1.0
    t_start = __import__("time").perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in tl:
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = torch.tensor(0.0)
            for t in range(2):
                pw_t = pos_weight[t] if pos_weight is not None else None
                loss = loss + (w_herg if t == 0 else 1.0 - w_herg) * masked_bce_loss(
                    logits[:, t], batch.y[:, t], pw_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        model.eval()
        all_outs, all_lbls = [], []
        with torch.no_grad():
            for batch in vl:
                logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                all_outs.append(logits.numpy())
                all_lbls.append(batch.y.numpy())

        y_pred = np.concatenate(all_outs, axis=0)
        y_true = np.concatenate(all_lbls, axis=0)
        herg_auc, cyp_auc = float("nan"), float("nan")
        for i in range(2):
            mask = ~np.isnan(y_true[:, i])
            if mask.sum() > 1 and len(np.unique(y_true[mask, i])) > 1:
                try:
                    auc = roc_auc_score(y_true[mask, i], y_pred[mask, i])
                    if i == 0: herg_auc = auc
                    else: cyp_auc = auc
                except ValueError: pass

        score = weighted_auc(herg_auc, cyp_auc, 0.6, 0.4)
        if score > best_score:
            best_score = score

        trial.report(score, epoch)

    elapsed = __import__("time").perf_counter() - t_start
    trial.set_user_attr("hERG_auc", float(herg_auc))
    trial.set_user_attr("CYP3A4_auc", float(cyp_auc))
    trial.set_user_attr("wall_clock_s", float(elapsed))
    log(f"  Trial {trial.number}: h={hidden_dim} d={depth} drop={dropout:.2f} "
        f"ro={readout} lr={lr:.1e} bs={batch_size} "
        f"-> hERG={herg_auc:.4f} CYP={cyp_auc:.4f} score={score:.4f} [{elapsed:.0f}s]")
    return best_score

log("Starting dry-run study...")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(quick_objective, n_trials=2)

log("Dry-run OK -- no crashes!")
log(f"Best: {study.best_params} -> {study.best_value:.4f}")

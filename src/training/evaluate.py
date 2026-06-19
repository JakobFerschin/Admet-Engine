"""
ADMET-Engine Phase 7.3 STEP 5: Evaluation & Comparison.
Loads checkpoints, computes ROC-AUC / PR-AUC / F1 / Accuracy per task.
Generates ROC curve plot (PNG). Comparison table: v1 vs v1_ext vs v2.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, roc_curve,
)

# -- Flushed safe logger
def log(msg: str) -> None:
    safe = msg.encode("ascii", errors="replace").decode("ascii")
    ts = time.strftime("%H:%M:%S")
    print(f"{ts} {safe}", flush=True)

# -- Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from dataset import create_dataloaders, ENDPOINT_LABELS
from training.hparam_search import FlexiblePredictor

CSV_PATH = PROJECT_ROOT / "data" / "processed" / "chembl_admet_training.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")
ENDPOINT_LABELS = ["hERG", "CYP3A4"]
NUM_TASKS = 2
SEED = 42

# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(ckpt_path, hparams=None):
    if hparams is None:
        hparams = {"hidden_dim": 300, "depth": 3, "readout": "set2set", "dropout": 0.2}

    model = FlexiblePredictor(
        node_dim=9, edge_dim=4,
        hidden_dim=hparams["hidden_dim"],
        num_tasks=NUM_TASKS,
        T=hparams.get("depth", 3),
        readout=hparams.get("readout", "set2set"),
        set2set_steps=3,
        dropout=hparams.get("dropout", 0.2),
    )

    state = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=True)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
            log(f"  loaded from checkpoint wrapper (epoch {state.get('epoch','?')})")
        else:
            model.load_state_dict(state)
            log(f"  loaded state_dict ({len(state)} keys)")
    else:
        model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_full(model, loader):
    all_logits, all_targets = [], []
    for batch in loader:
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        all_logits.append(logits.numpy())
        all_targets.append(batch.y.numpy())

    y_pred = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    results = {}
    for i, name in enumerate(ENDPOINT_LABELS):
        mask = ~np.isnan(y_true[:, i])
        n = mask.sum()
        if n <= 1 or len(np.unique(y_true[mask, i])) <= 1:
            results[name] = {"n": int(n), "roc_auc": None, "pr_auc": None,
                             "f1": None, "accuracy": None}
            continue

        yt = y_true[mask, i]
        yp = y_pred[mask, i]

        # ROC-AUC
        try:
            roc = roc_auc_score(yt, yp)
        except ValueError:
            roc = None

        # PR-AUC
        try:
            pr = average_precision_score(yt, yp)
        except ValueError:
            pr = None

        # F1 @ threshold = 0.0 (sigmoid default)
        yp_bin = (yp > 0.0).astype(int)
        try:
            f1 = f1_score(yt, yp_bin)
        except Exception:
            f1 = None
        try:
            acc = accuracy_score(yt, yp_bin)
        except Exception:
            acc = None

        results[name] = {
            "n": int(n), "roc_auc": roc, "pr_auc": pr, "f1": f1, "accuracy": acc,
            "y_true": yt, "y_pred": yp,  # for plotting
        }

    return results

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def main():
    log("=== PHASE 7.3 STEP 5: Evaluate & Compare ===")

    # -- Load data
    _, _, test_loader, info = create_dataloaders(
        CSV_PATH, batch_size=64, seed=SEED,
    )
    log(f"Test set: {info['n_test']} molecules")

    # -- Checkpoint definitions
    checkpoints = {
        "v1 (Phase 7.2)": {
            "path": MODELS_DIR / "dmpnn_chembl_v1.pt",
            "hparams": {"hidden_dim": 300, "depth": 3, "readout": "set2set", "dropout": 0.2},
        },
        "v1_ext (Phase 7.2.1)": {
            "path": MODELS_DIR / "dmpnn_chembl_v1_ext.pt",
            "hparams": {"hidden_dim": 300, "depth": 3, "readout": "set2set", "dropout": 0.2},
        },
    }

    # -- Add v2 if exists
    v2_path = MODELS_DIR / "dmpnn_chembl_v2.pt"
    if v2_path.exists():
        v2_hparams = {"hidden_dim": 300, "depth": 3, "readout": "set2set", "dropout": 0.2}
        if (MODELS_DIR / "best_hparams.json").exists():
            best = json.loads((MODELS_DIR / "best_hparams.json").read_text())
            v2_hparams = best["params"]
        checkpoints["v2 (Phase 7.3)"] = {
            "path": v2_path,
            "hparams": v2_hparams,
        }

    # -- Evaluate each
    all_results = {}
    for label, cfg in checkpoints.items():
        ckpt = cfg["path"]
        if not ckpt.exists():
            log(f"Skipping {label}: {ckpt} not found")
            continue

        log(f"\nEvaluating {label}...")
        model = load_model(ckpt, cfg["hparams"])
        metrics = evaluate_full(model, test_loader)
        all_results[label] = metrics

    # -- Print comparison table
    log("\n" + "=" * 70)
    log("COMPARISON TABLE")
    log("=" * 70)
    header = f"{'Model':24s} | {'Task':8s} | {'ROC-AUC':>8s} | {'PR-AUC':>8s} | {'F1':>6s} | {'Acc':>6s} | {'n':>5s}"
    log(header)
    log("-" * 70)

    for label, metrics in all_results.items():
        for name in ENDPOINT_LABELS:
            m = metrics.get(name, {})
            roc_str = f"{m.get('roc_auc', 0):.4f}" if m.get('roc_auc') is not None else "N/A"
            pr_str = f"{m.get('pr_auc', 0):.4f}" if m.get('pr_auc') is not None else "N/A"
            f1_str = f"{m.get('f1', 0):.4f}" if m.get('f1') is not None else "N/A"
            acc_str = f"{m.get('accuracy', 0):.4f}" if m.get('accuracy') is not None else "N/A"
            n_str = str(m.get("n", 0))
            log(f"{label:24s} | {name:8s} | {roc_str:>8s} | {pr_str:>8s} | {f1_str:>6s} | {acc_str:>6s} | {n_str:>5s}")

    # -- ROC Plot (if matplotlib available)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = {"v1": "#22d3ee", "v1_ext": "#f59e0b", "v2": "#10b981"}

        for task_idx, task_name in enumerate(ENDPOINT_LABELS):
            ax = axes[task_idx]
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")

            for label, metrics in all_results.items():
                m = metrics.get(task_name, {})
                if "y_true" not in m:
                    continue

                short = "v1" if "v1 " in label else ("v2" if "v2" in label else "v1_ext")
                yt, yp = m["y_true"], m["y_pred"]
                fpr, tpr, _ = roc_curve(yt, yp)
                roc = m.get("roc_auc", 0) or 0
                ax.plot(fpr, tpr, color=colors.get(short, "#888"),
                        label=f"{short} (AUC={roc:.3f})", linewidth=2)

            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"{task_name} — ROC Curve")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = RESULTS_DIR / "phase73_comparison.png"
        plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        log(f"\nROC plot saved: {plot_path}")
        plt.close()
    except ImportError:
        log("matplotlib not available — skipping ROC plot")
    except Exception as e:
        log(f"Plot error: {e}")

    log("\n=== STEP 5: DONE ===")

if __name__ == "__main__":
    main()

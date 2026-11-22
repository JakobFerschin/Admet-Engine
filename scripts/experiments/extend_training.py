"""
Phase 7.2.1 — Extended Training Run
Continues from models/dmpnn_chembl_v1.pt for up to 10 more epochs.
"""
import time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import create_dataloaders, ENDPOINT_LABELS
from models.dmpnn import ADMETPredictor
from train import masked_bce, evaluate, train_epoch

PROJECT = Path(__file__).resolve().parent
CSV = PROJECT / "data/processed/chembl_admet_training.csv"
CKPT_IN = PROJECT / "models/dmpnn_chembl_v1.pt"
CKPT_OUT = PROJECT / "models/dmpnn_chembl_v1_ext.pt"
DEVICE = torch.device("cpu")

# Baseline from Phase 7.2
BASELINE = {"hERG": 0.5676, "CYP3A4": 0.5506}

# Extended config
MAX_EPOCHS = 10
PATIENCE = 5
LR = 1e-4
WD = 1e-5
BATCH = 64
GRAD_CLIP = 5.0

print("=" * 60)
print("PHASE 7.2.1 — EXTENDED TRAINING")
print("=" * 60)

# Data
train_loader, val_loader, test_loader, info = create_dataloaders(CSV, batch_size=BATCH, seed=42)
print(f"Data: {info['n_train']} train / {info['n_val']} val / {info['n_test']} test")

# Model
model = ADMETPredictor(node_dim=9, edge_dim=4, hidden_dim=300, num_tasks=2, T=3, set2set_steps=3, dropout=0.2)
state = torch.load(str(CKPT_IN), map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model.to(DEVICE)
print(f"Loaded: {CKPT_IN.name} ({sum(p.numel() for p in model.parameters()):,} params)")

# Pre-training eval
val_loss_start, val_metrics_start = evaluate(model, val_loader, DEVICE)
print(f"Starting: val_loss={val_loss_start:.4f}  hERG={val_metrics_start.get('hERG_roc_auc',0):.4f}  CYP={val_metrics_start.get('CYP3A4_roc_auc',0):.4f}")

optimizer = Adam(model.parameters(), lr=LR, weight_decay=WD)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0
history = []
t_total = time.perf_counter()

for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.perf_counter()
    train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
    val_loss, val_metrics = evaluate(model, val_loader, DEVICE)
    elapsed = time.perf_counter() - t0

    scheduler.step(val_loss)

    herg = val_metrics.get("hERG_roc_auc", 0)
    cyp = val_metrics.get("CYP3A4_roc_auc", 0)

    print(f"Ext Epoch {epoch:2d} | tr={train_loss:.4f} | vl={val_loss:.4f} | hERG={herg:.4f} | CYP={cyp:.4f} | {elapsed:.0f}s", end="")

    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "hERG_auc": herg, "CYP3A4_auc": cyp})

    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0
        torch.save(model.state_dict(), str(CKPT_OUT))
        print(" ** BEST")
    else:
        patience_counter += 1
        print()

    if patience_counter >= PATIENCE:
        print(f"Early stopping at ext epoch {epoch} (patience={PATIENCE})")
        break

total_time = time.perf_counter() - t_total

# Load best
model.load_state_dict(torch.load(str(CKPT_OUT), map_location=DEVICE, weights_only=True))

# Final test eval
test_loss, test_metrics = evaluate(model, test_loader, DEVICE)
herg_final = test_metrics.get("hERG_roc_auc", 0)
cyp_final = test_metrics.get("CYP3A4_roc_auc", 0)

# ── Report ──
print()
print("=" * 60)
print("FINAL RESULTS — PHASE 7.2.1")
print("=" * 60)
print(f"  Best ext epoch:   {best_epoch}/{epoch}")
print(f"  Total ext time:   {total_time:.0f}s")
print(f"  Checkpoint:       {CKPT_OUT.name}")
print()
print(f"  hERG   ROC-AUC:   {herg_final:.4f}  (baseline: {BASELINE['hERG']:.4f})")
print(f"  CYP3A4 ROC-AUC:   {cyp_final:.4f}  (baseline: {BASELINE['CYP3A4']:.4f})")
print(f"  Test loss:        {test_loss:.4f}")
print()
delta_h = herg_final - BASELINE["hERG"]
delta_c = cyp_final - BASELINE["CYP3A4"]
print(f"  Δ hERG:           {delta_h:+.4f}")
print(f"  Δ CYP3A4:         {delta_c:+.4f}")
print()

# Learning curve analysis
losses = [h["val_loss"] for h in history]
aucs_h = [h["hERG_auc"] for h in history]
aucs_c = [h["CYP3A4_auc"] for h in history]

print("─ Learning Curve Analysis ─")
if len(losses) >= 3:
    loss_trend = losses[-1] - losses[0]
    print(f"  Val loss trend:   {loss_trend:+.4f} (start→end)")
    if abs(loss_trend) < 0.002:
        print("  → PLATEAU reached — minimal loss change")
    elif loss_trend < -0.005:
        print("  → Still LEARNING — loss decreasing")
    else:
        print("  → Possible OVERFITTING — val loss rising")

print(f"  hERG AUC range:   {min(aucs_h):.4f} – {max(aucs_h):.4f}")
print(f"  CYP AUC range:    {min(aucs_c):.4f} – {max(aucs_c):.4f}")

# Potential estimate
print()
print("─ Potential Estimate ─")
print("  At 50 epochs (extrapolated):")
# Rough linear extrapolation: assume diminishing returns
# If current gain per 10 epochs is small, plateau is real
avg_gain_h = delta_h / max(len(history), 1)
ext_h = herg_final + avg_gain_h * 4  # 40 more epochs
print(f"    hERG:   ~{ext_h:.4f} AUC (diminishing returns likely)")

avg_gain_c = delta_c / max(len(history), 1)
ext_c = cyp_final + avg_gain_c * 4
print(f"    CYP3A4: ~{ext_c:.4f} AUC")

if herg_final > 0.60:
    print("  → hERG > 0.60: RECOMMEND Phase 7.3 Hyperparameter Optimization")
else:
    print("  → hERG ≤ 0.60: RECOMMEND Architecture upgrade:")
    print("      - Larger hidden_dim (300→512)")
    print("      - More message-passing steps (T=3→5)")
    print("      - Add more ChEMBL targets (5→12 endpoints)")
    print("      - GPU training (currently CPU-bound)")

print("=" * 60)

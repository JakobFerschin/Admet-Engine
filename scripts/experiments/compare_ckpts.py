import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import torch
from dataset import create_dataloaders, ENDPOINT_LABELS
from models.dmpnn import ADMETPredictor
from train import evaluate

DEVICE = torch.device("cpu")
CSV = Path(__file__).resolve().parent / "data/processed/chembl_admet_training.csv"

# Load data once
_, _, test_loader, info = create_dataloaders(CSV, batch_size=64, seed=42)
print(f"Test: {info['n_test']} molecules")

for name, ckpt in [("Phase 7.2", "models/dmpnn_chembl_v1.pt"), ("Phase 7.2.1 ext", "models/dmpnn_chembl_v1_ext.pt")]:
    model = ADMETPredictor(node_dim=9, edge_dim=4, hidden_dim=300, num_tasks=2, T=3, set2set_steps=3, dropout=0.2)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    test_loss, metrics = evaluate(model, test_loader, DEVICE)
    herg = metrics.get("hERG_roc_auc", 0)
    cyp = metrics.get("CYP3A4_roc_auc", 0)
    print(f"\n{name}:")
    print(f"  hERG   ROC-AUC: {herg:.4f}")
    print(f"  CYP3A4 ROC-AUC: {cyp:.4f}")
    print(f"  Test loss:      {test_loss:.4f}")

baseline = {"hERG": 0.5676, "CYP3A4": 0.5506}
delta_h = herg - baseline["hERG"]
delta_c = cyp - baseline["CYP3A4"]
print(f"\nDelta vs Phase 7.2 baseline:")
print(f"  d hERG:   {delta_h:+.4f}")
print(f"  d CYP3A4: {delta_c:+.4f}")

if herg > 0.60:
    print("\n>> hERG > 0.60: RECOMMEND Phase 7.3 Hyperparameter Optimization")
else:
    print("\n>> No significant improvement: RECOMMEND Architecture Upgrade")
    print("   - Larger hidden_dim (300->512)")
    print("   - More msg-passing steps (T=3->5)")
    print("   - Add 3+ more ChEMBL targets")
    print("   - GPU training needed for scale")

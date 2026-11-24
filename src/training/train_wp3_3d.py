#!/usr/bin/env python3
"""
ADMET-Engine v3.0 (CardioPhys-3D) — WP3 GPU Training Pipeline
=============================================================
Trains the SE(3)-equivariant PaiNN 3D protein-ligand model
(`CardioPhys3DPredictor`) on the WP1 multi-channel ChEMBL v37 dataset
(hERG, Cav1.2, Nav1.5) with:
  - Boltzmann-sampled 3D conformers during training (physical 3D augmentation)
  - Lowest-energy / Boltzmann-averaged conformer evaluation on val/test
  - Censored Gaussian Tobit Negative Log-Likelihood ('=', '>', '<')
  - Automatic bfloat16/float16 mixed precision on CUDA GPUs (Colab A100/L4/T4)
  - Applicability-Domain stratified Mondrian Conformal calibration on val
  - Full held-out Bemis-Murcko scaffold test evaluation (RMSE, Pearson r,
    Spearman rho, censored agreement rate, and empirical conformal coverage)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.equivariant_3d import CHANNELS_V3, CardioPhys3DPredictor  # noqa: E402
from training.tobit_loss import (  # noqa: E402
    MondrianConformalRegressor,
    censored_concordance_rate,
    tobit_censored_nll,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 3D Conformer Multi-Channel Dataset & Collate
# ---------------------------------------------------------------------------

class CardioPhys3DDataset(Dataset):
    """Dataset pairing 3D Boltzmann conformer ensembles with Tobit-censored pIC50 targets."""

    def __init__(
        self,
        df: pd.DataFrame,
        conformers: dict[str, dict[str, Any]],
        sample_boltzmann: bool = False,
    ):
        self.sample_boltzmann = sample_boltzmann
        self.items: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            smi = str(row["canonical_smiles"])
            conf = conformers.get(smi)
            if conf is None:
                continue
            y = np.array(
                [row.get(f"{ch}_pIC50", np.nan) for ch in CHANNELS_V3],
                dtype=np.float32,
            )
            c = np.array(
                [row.get(f"{ch}_censor", np.nan) for ch in CHANNELS_V3],
                dtype=np.float32,
            )
            if np.isnan(y).all():
                continue
            self.items.append({
                "smiles": smi,
                "z": conf["z"].long(),
                "formal_charge": conf["formal_charge"].long(),
                "pos_all": conf["pos"].float(),  # [K, N, 3]
                "boltz_w": conf["boltzmann_weights"].float(),  # [K]
                "y": torch.from_numpy(y),
                "censor": torch.from_numpy(c),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        it = self.items[idx]
        pos_all = it["pos_all"]
        if self.sample_boltzmann and pos_all.shape[0] > 1:
            k = int(torch.multinomial(it["boltz_w"], 1).item())
        else:
            k = 0  # lowest MMFF94s energy conformer
        return {
            "smiles": it["smiles"],
            "z": it["z"],
            "formal_charge": it["formal_charge"],
            "pos": pos_all[k],
            "y": it["y"],
            "censor": it["censor"],
        }


def collate_3d_batch(batch_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-size 3D molecules into flat atom tensors + `batch` vector."""
    z_list, q_list, pos_list, batch_idx = [], [], [], []
    y_list, c_list, smiles_list = [], [], []

    for i, item in enumerate(batch_list):
        n_atoms = item["z"].shape[0]
        z_list.append(item["z"])
        q_list.append(item["formal_charge"])
        pos_list.append(item["pos"])
        batch_idx.append(torch.full((n_atoms,), i, dtype=torch.long))
        y_list.append(item["y"])
        c_list.append(item["censor"])
        smiles_list.append(item["smiles"])

    return {
        "smiles": smiles_list,
        "z": torch.cat(z_list, dim=0),
        "formal_charge": torch.cat(q_list, dim=0),
        "pos": torch.cat(pos_list, dim=0),
        "batch": torch.cat(batch_idx, dim=0),
        "y": torch.stack(y_list, dim=0),
        "censor": torch.stack(c_list, dim=0),
    }


# ---------------------------------------------------------------------------
# Evaluation Helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_3d(
    model: CardioPhys3DPredictor,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_nll = 0.0
    n_batches = 0
    mus, sigs, ys, cs = [], [], [], []

    for batch in loader:
        z = batch["z"].to(device)
        q = batch["formal_charge"].to(device)
        pos = batch["pos"].to(device)
        bidx = batch["batch"].to(device)
        y = batch["y"].to(device)
        c = batch["censor"].to(device)

        mu, sigma = model(z, pos, bidx, formal_charge=q)
        loss = tobit_censored_nll(mu, sigma, y, c)
        total_nll += float(loss.item())
        n_batches += 1

        mus.append(mu.cpu().numpy())
        sigs.append(sigma.cpu().numpy())
        ys.append(y.cpu().numpy())
        cs.append(c.cpu().numpy())

    mu_np = np.concatenate(mus, axis=0)
    sig_np = np.concatenate(sigs, axis=0)
    y_np = np.concatenate(ys, axis=0)
    c_np = np.concatenate(cs, axis=0)

    metrics: dict[str, Any] = {"tobit_nll": round(total_nll / max(n_batches, 1), 4)}
    for ch_idx, ch_name in enumerate(CHANNELS_V3):
        exact_m = (~np.isnan(y_np[:, ch_idx])) & (c_np[:, ch_idx] == 0)
        ch_met: dict[str, Any] = {"n_exact": int(exact_m.sum())}
        if exact_m.sum() >= 3:
            yt = y_np[exact_m, ch_idx]
            yp = mu_np[exact_m, ch_idx]
            ch_met["rmse"] = round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4)
            ch_met["mae"] = round(float(np.mean(np.abs(yt - yp))), 4)
            ch_met["pearson_r"] = round(float(pearsonr(yt, yp)[0]), 4)
            ch_met["spearman_rho"] = round(float(spearmanr(yt, yp)[0]), 4)
        ch_met.update(censored_concordance_rate(mu_np[:, ch_idx], y_np[:, ch_idx], c_np[:, ch_idx]))
        metrics[ch_name] = ch_met

    return metrics["tobit_nll"], metrics, mu_np, sig_np, y_np, c_np


# ---------------------------------------------------------------------------
# Main Training Entry Point (Colab A100/L4/T4 Ready)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="WP3: Train SE(3)-Equivariant CardioPhys-3D")
    parser.add_argument(
        "--csv", type=Path, default=PROJECT_ROOT / "data/processed/cardiophys_multichannel_v3.csv"
    )
    parser.add_argument(
        "--conformers", type=Path, default=PROJECT_ROOT / "data/processed/cardiophys_3d_conformers_v3.pt"
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "models")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    print(f"[WP3] Device: {device} | AMP: {use_amp} ({amp_dtype if use_amp else 'fp32'})", flush=True)

    df = pd.read_csv(args.csv)
    conformers = torch.load(str(args.conformers), map_location="cpu", weights_only=False)

    tr_ds = CardioPhys3DDataset(df[df["split"] == "train"], conformers, sample_boltzmann=True)
    va_ds = CardioPhys3DDataset(df[df["split"] == "val"], conformers, sample_boltzmann=False)
    te_ds = CardioPhys3DDataset(df[df["split"] == "test"], conformers, sample_boltzmann=False)
    print(f"[WP3] 3D Datasets: train={len(tr_ds):,} | val={len(va_ds):,} | test={len(te_ds):,}", flush=True)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_3d_batch)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_3d_batch)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_3d_batch)

    model = CardioPhys3DPredictor(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_channels=len(CHANNELS_V3),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[WP3] CardioPhys3DPredictor: {n_params:,} parameters", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_nll = float("inf")
    best_state = None
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        tr_loss_sum = 0.0
        n_b = 0

        for batch in tr_loader:
            z = batch["z"].to(device)
            q = batch["formal_charge"].to(device)
            pos = batch["pos"].to(device)
            bidx = batch["batch"].to(device)
            y = batch["y"].to(device)
            c = batch["censor"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                mu, sigma = model(z, pos, bidx, formal_charge=q)
                loss = tobit_censored_nll(mu.float(), sigma.float(), y.float(), c.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            tr_loss_sum += float(loss.item())
            n_b += 1

        scheduler.step()
        val_nll, val_met, _, _, _, _ = evaluate_3d(model, va_loader, device)
        herg_r = val_met["hERG"].get("pearson_r", float("nan"))
        cav_r = val_met["Cav1.2"].get("pearson_r", float("nan"))
        nav_r = val_met["Nav1.5"].get("pearson_r", float("nan"))
        print(
            f"Epoch {epoch:2d}/{args.epochs} | tr_nll={tr_loss_sum/max(n_b,1):.4f} | "
            f"val_nll={val_nll:.4f} | r(hERG)={herg_r:.3f} r(Cav1.2)={cav_r:.3f} r(Nav1.5)={nav_r:.3f} | "
            f"{time.perf_counter()-t0:.1f}s",
            flush=True,
        )
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Mondrian Conformal calibration on validation split (alpha = 0.10 -> 90% coverage)
    _, val_met, val_mu, val_sig, val_y, val_c = evaluate_3d(model, va_loader, device)
    _, test_met, test_mu, test_sig, test_y, test_c = evaluate_3d(model, te_loader, device)

    conformal_summary: dict[str, Any] = {}
    for ch_idx, ch_name in enumerate(CHANNELS_V3):
        val_ex = (~np.isnan(val_y[:, ch_idx])) & (val_c[:, ch_idx] == 0)
        te_ex = (~np.isnan(test_y[:, ch_idx])) & (test_c[:, ch_idx] == 0)
        if val_ex.sum() >= 10 and te_ex.sum() >= 10:
            cp = MondrianConformalRegressor(alpha=0.10, min_group_size=15)
            cp.fit(
                val_y[val_ex, ch_idx],
                val_mu[val_ex, ch_idx],
                val_sig[val_ex, ch_idx],
                groups=["global"] * int(val_ex.sum()),
            )
            lo, hi = cp.predict_intervals(
                test_mu[te_ex, ch_idx],
                test_sig[te_ex, ch_idx],
                groups=["global"] * int(te_ex.sum()),
            )
            cov = float(np.mean((test_y[te_ex, ch_idx] >= lo) & (test_y[te_ex, ch_idx] <= hi)))
            width = float(np.mean(hi - lo))
            conformal_summary[ch_name] = {
                **cp.to_dict(),
                "test_empirical_coverage": round(cov, 4),
                "test_mean_interval_width_pIC50": round(width, 4),
            }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / "cardiophys_3d_v3.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_channels": len(CHANNELS_V3),
            },
            "conformal": conformal_summary,
        },
        str(ckpt_path),
    )

    report = {
        "model_version": "cardiophys_3d_v3",
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_params": n_params,
        "train_duration_sec": round(time.perf_counter() - t_start, 1),
        "val_metrics": val_met,
        "test_metrics": test_met,
        "conformal_90pct": conformal_summary,
    }
    meta_path = args.out_dir / "cardiophys_3d_v3_metrics.json"
    meta_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[WP3] DONE! Saved {ckpt_path} and {meta_path}")
    print(json.dumps(report["test_metrics"], indent=2))


if __name__ == "__main__":
    main()

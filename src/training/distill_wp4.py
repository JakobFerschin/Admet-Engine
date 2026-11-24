#!/usr/bin/env python3
"""
ADMET-Engine v3.0 (CardioPhys-3D) — WP4: Teacher -> Student Knowledge Distillation
==================================================================================
Distills the heavy SE(3)-equivariant 3D PaiNN teacher (`CardioPhys3DPredictor`)
into a sub-2ms edge student network (`CardioPhysStudentNet`) via closed-form
Gaussian KL-divergence + censored Tobit NLL:

    L_distill = (1 - lam) * L_Tobit(mu_S, sigma_S; y, c)
              + lam * KL( N(mu_T, sigma_T^2) || N(mu_S, sigma_S^2) )

Benchmarks:
  - Held-out Bemis-Murcko scaffold RMSE, Pearson r, Spearman rho per channel
  - Teacher-Student fidelity (Pearson r & MAE between mu_T and mu_S)
  - Edge CPU inference latency (ms / molecule) for WP5 closed-loop molecular repair
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cheminformatics import compute_morgan_numpy, compute_rdkit_ml_descriptors  # noqa: E402
from models.distillation import (  # noqa: E402
    CardioPhysStudentNet,
    distillation_tobit_loss,
)
from models.equivariant_3d import CHANNELS_V3, CardioPhys3DPredictor  # noqa: E402
from training.train_wp3_3d import (  # noqa: E402
    CardioPhys3DDataset,
    collate_3d_batch,
    set_seed,
)


def compute_2d_features(smiles_list: list[str]) -> np.ndarray:
    """Compute 2068-dim (2048 ECFP4 + 20 RDKit physicochemical) feature matrix."""
    n = len(smiles_list)
    mat = np.zeros((n, 2068), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mat[i, :2048] = compute_morgan_numpy(mol)
        mat[i, 2048:] = compute_rdkit_ml_descriptors(mol)
    return mat


@torch.no_grad()
def extract_teacher_targets(
    teacher: CardioPhys3DPredictor,
    df: pd.DataFrame,
    conformers: dict[str, dict[str, Any]],
    device: torch.device,
    batch_size: int = 64,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the 3D teacher once over a split to cache (mu_T, sigma_T, y, censor)."""
    ds = CardioPhys3DDataset(df, conformers, sample_boltzmann=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_3d_batch)

    teacher.eval()
    smiles_all: list[str] = []
    mu_list, sig_list, y_list, c_list = [], [], [], []
    for batch in loader:
        mu_t, sig_t = teacher(
            batch["z"].to(device),
            batch["pos"].to(device),
            batch["batch"].to(device),
            batch["formal_charge"].to(device),
        )
        smiles_all.extend(batch["smiles"])
        mu_list.append(mu_t.cpu().numpy())
        sig_list.append(sig_t.cpu().numpy())
        y_list.append(batch["y"].numpy())
        c_list.append(batch["censor"].numpy())

    return (
        smiles_all,
        np.concatenate(mu_list, axis=0),
        np.concatenate(sig_list, axis=0),
        np.concatenate(y_list, axis=0),
        np.concatenate(c_list, axis=0),
    )


def evaluate_student(
    student: CardioPhysStudentNet,
    x_np: np.ndarray,
    mu_t_np: np.ndarray,
    y_np: np.ndarray,
    c_np: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    student.eval()
    with torch.no_grad():
        mu_s, sig_s = student(torch.from_numpy(x_np).to(device))
        mu_s_np = mu_s.cpu().numpy()

    metrics: dict[str, Any] = {}
    for ch_idx, ch_name in enumerate(CHANNELS_V3):
        exact = (~np.isnan(y_np[:, ch_idx])) & (c_np[:, ch_idx] == 0)
        ch_m: dict[str, Any] = {"n_exact": int(exact.sum())}
        if exact.sum() >= 5:
            yt = y_np[exact, ch_idx]
            yp = mu_s_np[exact, ch_idx]
            ch_m["rmse_pIC50"] = round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4)
            ch_m["pearson_r"] = round(float(pearsonr(yt, yp)[0]), 4)
            ch_m["spearman_rho"] = round(float(spearmanr(yt, yp)[0]), 4)

        # Teacher-student fidelity across all molecules in split
        ch_m["teacher_fidelity_r"] = round(float(pearsonr(mu_t_np[:, ch_idx], mu_s_np[:, ch_idx])[0]), 4)
        ch_m["teacher_fidelity_mae"] = round(float(np.mean(np.abs(mu_t_np[:, ch_idx] - mu_s_np[:, ch_idx]))), 4)
        metrics[ch_name] = ch_m

    return metrics


def benchmark_cpu_latency_ms(student: CardioPhysStudentNet, n_runs: int = 100) -> float:
    """Measure single-molecule CPU inference latency in milliseconds."""
    student_cpu = student.to("cpu").eval()
    x1 = torch.randn(1, student.in_dim)
    with torch.no_grad():
        for _ in range(10):  # warmup
            _ = student_cpu(x1)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = student_cpu(x1)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / n_runs
    return round(float(elapsed_ms), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="WP4 Knowledge Distillation (3D Teacher -> Edge Student)")
    parser.add_argument("--data-csv", type=Path, default=PROJECT_ROOT / "data" / "v3_multichannel_chembl37.csv")
    parser.add_argument("--conformers-pt", type=Path, default=PROJECT_ROOT / "data" / "v3_conformers_3d.pt")
    parser.add_argument("--teacher-ckpt", type=Path, default=PROJECT_ROOT / "models" / "cardiophys_3d_v3.pt")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--distill-weight", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[WP4] Device: {device} | Loading WP3 teacher from {args.teacher_ckpt}")

    ckpt = torch.load(str(args.teacher_ckpt), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    teacher = CardioPhys3DPredictor(
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        num_layers=int(cfg.get("num_layers", 3)),
        num_channels=int(cfg.get("num_channels", len(CHANNELS_V3))),
    ).to(device)
    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher.eval()

    df = pd.read_csv(args.data_csv)
    conformers = torch.load(str(args.conformers_pt), map_location="cpu", weights_only=False)

    splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for split_name in ("train", "val", "test"):
        df_sp = df[df["split"] == split_name].reset_index(drop=True)
        smis, mu_t, sig_t, y, c = extract_teacher_targets(teacher, df_sp, conformers, device)
        x_2d = compute_2d_features(smis)
        splits[split_name] = (x_2d, mu_t, sig_t, y, c)
        print(f"[WP4] Cached teacher outputs & 2D features for {split_name}: n={len(smis)}")

    x_tr, mu_t_tr, sig_t_tr, y_tr, c_tr = splits["train"]
    x_va, mu_t_va, sig_t_va, y_va, c_va = splits["val"]
    x_te, mu_t_te, sig_t_te, y_te, c_te = splits["test"]

    train_ds = TensorDataset(
        torch.from_numpy(x_tr),
        torch.from_numpy(mu_t_tr),
        torch.from_numpy(sig_t_tr),
        torch.from_numpy(y_tr),
        torch.from_numpy(c_tr),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    student = CardioPhysStudentNet(
        in_dim=2068,
        hidden_dim=args.hidden_dim,
        num_channels=len(CHANNELS_V3),
    ).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_state = None
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        student.train()
        tr_loss = 0.0
        n_b = 0
        for bx, bmu_t, bsig_t, by, bc in train_loader:
            bx = bx.to(device)
            bmu_t = bmu_t.to(device)
            bsig_t = bsig_t.to(device)
            by = by.to(device)
            bc = bc.to(device)

            optimizer.zero_grad(set_to_none=True)
            mu_s, sig_s = student(bx)
            loss = distillation_tobit_loss(
                mu_s, sig_s, bmu_t, bsig_t,
                targets=by, censor=bc,
                distill_weight=args.distill_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
            optimizer.step()
            tr_loss += float(loss.item())
            n_b += 1

        scheduler.step()

        # Validation distillation loss
        student.eval()
        with torch.no_grad():
            va_mu_s, va_sig_s = student(torch.from_numpy(x_va).to(device))
            va_loss = float(
                distillation_tobit_loss(
                    va_mu_s, va_sig_s,
                    torch.from_numpy(mu_t_va).to(device),
                    torch.from_numpy(sig_t_va).to(device),
                    targets=torch.from_numpy(y_va).to(device),
                    censor=torch.from_numpy(c_va).to(device),
                    distill_weight=args.distill_weight,
                ).item()
            )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | train_loss={tr_loss/max(n_b,1):.4f} | val_loss={va_loss:.4f}")

    if best_state is not None:
        student.load_state_dict(best_state)

    val_metrics = evaluate_student(student, x_va, mu_t_va, y_va, c_va, device)
    test_metrics = evaluate_student(student, x_te, mu_t_te, y_te, c_te, device)
    cpu_latency_ms = benchmark_cpu_latency_ms(student)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    student_ckpt = args.out_dir / "cardiophys_student_v3.pt"
    torch.save(
        {
            "model_state_dict": student.cpu().state_dict(),
            "config": {
                "in_dim": 2068,
                "hidden_dim": args.hidden_dim,
                "num_channels": len(CHANNELS_V3),
            },
            "distill_weight": args.distill_weight,
        },
        str(student_ckpt),
    )

    report = {
        "model_version": "cardiophys_student_v3",
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_params": int(sum(p.numel() for p in student.parameters())),
        "train_duration_sec": round(time.perf_counter() - t0, 1),
        "single_mol_cpu_latency_ms": cpu_latency_ms,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    meta_path = args.out_dir / "cardiophys_student_v3_metrics.json"
    meta_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[WP4] DONE! Saved {student_ckpt} (CPU latency: {cpu_latency_ms:.3f} ms/mol)")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()

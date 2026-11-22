"""
ADMET-Engine v2.0 — Publication Figure & Statistical Analysis Generator
Generates vector PDF figures directly from the held-out Bemis-Murcko scaffold
test set (n=1,627 unseen scaffolds) and clinical pharmacology benchmark.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cheminformatics import get_murcko_scaffold  # noqa: E402
from data_pipeline import standardize_molecule  # noqa: E402
from models.chembl_ensemble import CHEMBL_TASKS, ChEMBLADMETEnsemble  # noqa: E402

PAPER_DIR = PROJECT_ROOT / "paper"
FIG_DIR = PAPER_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "chembl_admet_training.csv"

# Harvard / Nature academic color palette
CRIMSON = "#A51C30"
NAVY = "#1E3A5F"
TEAL = "#0F766E"
OCHRE = "#B45309"
SLATE = "#64748B"
LIGHT_GRAY = "#CBD5E1"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.35,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

_FAST_LFC = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)


def _fast_standardize(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    if "." in smiles:
        mol = _FAST_LFC.choose(mol)
    return Chem.RemoveHs(mol)


def reconstruct_scaffold_test_split():
    """Reproduce the exact deterministic Bemis-Murcko scaffold test split."""
    df = pd.read_csv(CSV_PATH)
    records = {}
    for _, row in df.iterrows():
        mol = _fast_standardize(str(row["canonical_smiles"]))
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        can = Chem.MolToSmiles(mol, canonical=True)
        if can not in records:
            records[can] = {"mol": mol, "herg": [], "cyp": []}
        if not pd.isna(row["hERG_label"]):
            records[can]["herg"].append(float(row["hERG_label"]))
        if not pd.isna(row["CYP3A4_label"]):
            records[can]["cyp"].append(float(row["CYP3A4_label"]))

    smiles, mols, scaffolds, labels = [], [], [], []
    for can, rec in records.items():
        y0 = float(np.mean(rec["herg"]) >= 0.5) if rec["herg"] else np.nan
        y1 = float(np.mean(rec["cyp"]) >= 0.5) if rec["cyp"] else np.nan
        smiles.append(can)
        mols.append(rec["mol"])
        scaffolds.append(get_murcko_scaffold(rec["mol"]))
        labels.append([y0, y1])

    labels = np.array(labels, dtype=np.float32)
    n = len(smiles)

    scaf_groups = defaultdict(list)
    for i, sc in enumerate(scaffolds):
        scaf_groups[sc].append(i)
    sorted_groups = sorted(scaf_groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    train_cut = int(0.80 * n)
    val_cut = int(0.90 * n)
    train_idx, val_idx, test_idx = [], [], []
    for _, indices in sorted_groups:
        if len(train_idx) + len(indices) <= train_cut or not train_idx:
            train_idx.extend(indices)
        elif len(train_idx) + len(val_idx) + len(indices) <= val_cut or not val_idx:
            val_idx.extend(indices)
        else:
            test_idx.extend(indices)

    test_mols = [mols[i] for i in test_idx]
    test_labels = labels[test_idx]
    return test_mols, test_labels


def compute_calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mean_pred, frac_pos, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi if i == n_bins - 1 else y_prob < hi)
        if mask.sum() > 0:
            mean_pred.append(float(y_prob[mask].mean()))
            frac_pos.append(float(y_true[mask].mean()))
            counts.append(int(mask.sum()))
    return np.array(mean_pred), np.array(frac_pos), np.array(counts)


def plot_fig1_roc_pr_cal(preds: list[dict], test_labels: np.ndarray):
    """2x3 multi-panel ROC, Precision-Recall, and Reliability diagram."""
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4), dpi=300)
    member_styles = [
        ("dmpnn", "Chemprop DMPNN", "#64748B", "--", 1.15),
        ("linear_qsar", "L2 Logistic QSAR", "#94A3B8", ":", 1.15),
        ("ecfp4_mlp", "Residual ECFP4-MLP", NAVY, "-.", 1.25),
        ("extra_trees", "ExtraTrees QSAR", TEAL, "--", 1.25),
        ("consensus", "Consensus Ensemble", CRIMSON, "-", 2.1),
    ]

    task_titles = [
        ("hERG_inhibition", "hERG (KCNH2)"),
        ("CYP3A4_inhibition", "CYP3A4"),
    ]
    panel_labels = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]

    for r_idx, (ep_key, ep_short) in enumerate(task_titles):
        mask = ~np.isnan(test_labels[:, r_idx])
        yt = test_labels[mask, r_idx].astype(int)
        n_pos = int(yt.sum())
        prev = yt.mean()

        probs_map = {
            "consensus": np.array([preds[i]["ml_endpoints"][ep_key]["probability"] for i in np.where(mask)[0]]),
            "dmpnn": np.array([preds[i]["ml_endpoints"][ep_key]["member_probabilities"]["dmpnn"] for i in np.where(mask)[0]]),
            "ecfp4_mlp": np.array([preds[i]["ml_endpoints"][ep_key]["member_probabilities"]["ecfp4_mlp"] for i in np.where(mask)[0]]),
            "extra_trees": np.array([preds[i]["ml_endpoints"][ep_key]["member_probabilities"]["extra_trees"] for i in np.where(mask)[0]]),
            "linear_qsar": np.array([preds[i]["ml_endpoints"][ep_key]["member_probabilities"]["linear_qsar"] for i in np.where(mask)[0]]),
        }

        # --- Col 0: ROC Curve ---
        ax = axes[r_idx, 0]
        ax.plot([0, 1], [0, 1], color=LIGHT_GRAY, lw=0.9, ls="--")
        for m_key, m_label, color, ls, lw in member_styles:
            yp = probs_map[m_key]
            fpr, tpr, _ = roc_curve(yt, yp)
            auc_v = roc_auc_score(yt, yp)
            ax.plot(fpr, tpr, color=color, ls=ls, lw=lw, label=f"{m_label} ({auc_v:.3f})")
        ax.set_title(f"{panel_labels[r_idx][0]} {ep_short} ROC (n={len(yt)})", loc="left")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(True)
        ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#E2E8F0")

        # --- Col 1: PR Curve ---
        ax = axes[r_idx, 1]
        ax.axhline(prev, color=LIGHT_GRAY, lw=0.9, ls="--", label=f"No-skill ({prev:.2f})")
        for m_key, m_label, color, ls, lw in member_styles:
            yp = probs_map[m_key]
            prec, rec, _ = precision_recall_curve(yt, yp)
            ap = average_precision_score(yt, yp)
            ax.plot(rec, prec, color=color, ls=ls, lw=lw, label=f"{m_label} ({ap:.3f})")
        ax.set_title(f"{panel_labels[r_idx][1]} {ep_short} Precision–Recall", loc="left")
        ax.set_xlabel("Recall (Sensitivity)")
        ax.set_ylabel("Precision (PPV)")
        ax.set_ylim(0.35, 1.02)
        ax.grid(True)
        ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#E2E8F0")

        # --- Col 2: Calibration / Reliability Diagram ---
        ax = axes[r_idx, 2]
        yp_ens = probs_map["consensus"]
        mp, fp, cnt = compute_calibration_bins(yt, yp_ens, n_bins=10)
        brier = brier_score_loss(yt, yp_ens)
        ece = float(np.sum((cnt / cnt.sum()) * np.abs(fp - mp)))

        ax.plot([0, 1], [0, 1], color=SLATE, lw=0.9, ls="--", label="Ideal calibration")
        ax.plot(mp, fp, color=CRIMSON, lw=1.9, marker="o", markersize=4.8, label="Consensus ensemble")

        # Subtle background density histogram of predicted probabilities
        ax_hist = ax.twinx()
        ax_hist.hist(yp_ens, bins=np.linspace(0, 1, 11), color=NAVY, alpha=0.12, edgecolor="none")
        ax_hist.set_ylim(0, len(yt) * 0.85)
        ax_hist.set_yticks([])
        ax_hist.spines["right"].set_visible(False)
        ax_hist.spines["top"].set_visible(False)

        ax.text(
            0.05, 0.82,
            f"Brier = {brier:.4f}\nECE = {ece:.4f}\nPrev. = {n_pos}/{len(yt)}",
            transform=ax.transAxes,
            fontsize=7.4,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CBD5E1", alpha=0.95),
        )
        ax.set_title(f"{panel_labels[r_idx][2]} {ep_short} Reliability Diagram", loc="left")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Observed Positive Fraction")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True)
        ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#E2E8F0")

    plt.tight_layout(w_pad=1.8, h_pad=2.0)
    out_path = FIG_DIR / "fig1_roc_pr_calibration.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_fig2_ad_stratification(preds: list[dict], test_labels: np.ndarray) -> dict:
    """Applicability Domain stratified ROC-AUC, Brier score, and epistemic uncertainty."""
    tanimotos = np.array([p["applicability_domain"]["max_tanimoto"] for p in preds])

    bins = [
        ("Out-of-Domain\n($T_{\\max} < 0.25$)", 0.0, 0.25),
        ("Borderline\n($0.25–0.40$)", 0.25, 0.40),
        ("In-Domain Mod.\n($0.40–0.60$)", 0.40, 0.60),
        ("In-Domain High\n($T_{\\max} \\geq 0.60$)", 0.60, 1.01),
    ]

    stats_out = {}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.8), dpi=300)

    x = np.arange(len(bins))
    width = 0.34

    for t_idx, (ep_key, ep_label, color) in enumerate([
        ("hERG_inhibition", "hERG (KCNH2)", CRIMSON),
        ("CYP3A4_inhibition", "CYP3A4", NAVY),
    ]):
        valid_mask = ~np.isnan(test_labels[:, t_idx])
        yt_all = test_labels[:, t_idx]
        yp_all = np.array([p["ml_endpoints"][ep_key]["probability"] for p in preds])
        std_all = np.array([p["ml_endpoints"][ep_key]["epistemic_std"] for p in preds])

        aucs, briers, mean_stds, ns = [], [], [], []
        bin_records = []
        for b_name, lo, hi in bins:
            m = valid_mask & (tanimotos >= lo) & (tanimotos < hi)
            n_b = int(m.sum())
            ns.append(n_b)
            if n_b >= 5 and len(np.unique(yt_all[m])) > 1:
                auc_v = float(roc_auc_score(yt_all[m], yp_all[m]))
                brier_v = float(brier_score_loss(yt_all[m], yp_all[m]))
            else:
                auc_v, brier_v = float("nan"), float("nan")
            std_v = float(np.mean(std_all[m])) if n_b > 0 else float("nan")
            aucs.append(auc_v)
            briers.append(brier_v)
            mean_stds.append(std_v)
            bin_records.append({
                "bin": b_name.replace("\n", " "),
                "n": n_b,
                "roc_auc": None if math.isnan(auc_v) else round(auc_v, 4),
                "brier": None if math.isnan(brier_v) else round(brier_v, 4),
                "mean_epistemic_std": None if math.isnan(std_v) else round(std_v, 4),
            })

        stats_out[ep_key] = bin_records
        offset = -width / 2 if t_idx == 0 else width / 2
        bars = ax1.bar(x + offset, [0 if math.isnan(v) else v for v in aucs], width=width * 0.92, color=color, alpha=0.88, label=ep_label)
        for bar, v, n_b in zip(bars, aucs, ns):
            if not math.isnan(v):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.016,
                    f"{v:.2f}\n({n_b})",
                    ha="center",
                    va="bottom",
                    fontsize=6.2,
                    color="#1E293B",
                )

        ax2.plot(
            x,
            mean_stds,
            color=color,
            marker="o" if t_idx == 0 else "s",
            lw=1.8,
            markersize=5.5,
            label=f"{ep_label} mean $\\sigma_{{\\mathrm{{epi}}}}$",
        )

    ax1.axhline(0.50, color=SLATE, ls=":", lw=0.9, alpha=0.75, label="Random (0.50)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([b[0] for b in bins], fontsize=7.4)
    ax1.set_ylim(0.15, 0.98)
    ax1.set_ylabel("Held-Out Scaffold ROC-AUC")
    ax1.set_title("(a) Predictive Accuracy vs. Applicability Domain", loc="left")
    ax1.grid(axis="y")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E2E8F0")

    ax2.set_xticks(x)
    ax2.set_xticklabels([b[0] for b in bins], fontsize=7.4)
    ax2.set_ylabel("Mean Epistemic Uncertainty ($\\sigma_{\\mathrm{epi}}$)")
    ax2.set_title("(b) Epistemic Disagreement Across Similarity Regimes", loc="left")
    ax2.grid(True)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#E2E8F0")

    plt.tight_layout(w_pad=2.2)
    out_path = FIG_DIR / "fig2_ad_stratification.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")
    return stats_out


def _ellipse_xy(cx, cy, a, b, angle_deg, n=200):
    t = np.linspace(0, 2 * np.pi, n)
    rad = math.radians(angle_deg)
    lx = a * np.cos(t)
    ly = b * np.sin(t)
    x = cx + lx * math.cos(rad) - ly * math.sin(rad)
    y = cy + lx * math.sin(rad) + ly * math.cos(rad)
    return x, y


def plot_fig3_boiled_egg_clinical(test_mols: list, metadata: dict):
    """2-panel BOILED-Egg pharmacokinetic map + Clinical Pharmacology Forest Plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=300, gridspec_kw={"width_ratios": [1.05, 1.15]})

    # --- Panel (a): BOILED-Egg ---
    bg_tpsa = [rdMolDescriptors.CalcTPSA(m) for m in test_mols]
    bg_logp = [Crippen.MolLogP(m) for m in test_mols]

    ax1.set_facecolor("#F1F5F9")
    hx, hy = _ellipse_xy(71.051, 2.292, 142.081 / 2.0, 8.740 / 2.0, -1.031325)
    bx, by = _ellipse_xy(38.117, 3.177, 82.061 / 2.0, 5.557 / 2.0, 0.171887)
    ax1.fill(hx, hy, facecolor="white", edgecolor="#64748B", lw=1.1, zorder=1, label="HIA Region (White)")
    ax1.fill(bx, by, facecolor="#FEF3C7", edgecolor="#D97706", lw=1.25, zorder=2, label="BBB Permeant (Yolk)")

    ax1.scatter(bg_tpsa, bg_logp, s=6, color="#94A3B8", alpha=0.28, edgecolors="none", zorder=3, label="ChEMBL Test Set (n=1,627)")

    bench = metadata["clinical_benchmark"]
    label_offsets = {
        "Terfenadine": (7, 4),
        "Astemizole": (-54, 4),
        "Cisapride": (7, -9),
        "Ketoconazole": (7, 5),
        "Mibefradil": (7, 4),
        "Haloperidol": (-48, 5),
        "Aspirin": (7, -8),
        "Caffeine": (7, -4),
        "Acetaminophen": (-60, -3),
        "Metformin": (-12, 9),
    }

    for row in bench:
        short_name = row["name"].split(" (")[0]
        mol = standardize_molecule(row["smiles"])
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        wlogp = Crippen.MolLogP(mol)
        is_toxic = max(row["hERG_prob"], row["CYP3A4_prob"]) >= 0.70
        ax1.scatter(
            tpsa, wlogp,
            s=36,
            color=CRIMSON if is_toxic else TEAL,
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )
        dx, dy = label_offsets.get(short_name, (5, 4))
        ax1.annotate(
            short_name,
            (tpsa, wlogp),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.8,
            fontweight="bold" if is_toxic else "normal",
            color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.82),
            zorder=6,
        )

    ax1.set_xlim(-5, 195)
    ax1.set_ylim(-3.2, 7.6)
    ax1.set_xlabel("Topological Polar Surface Area — Ertl TPSA ($\\AA^2$)")
    ax1.set_ylabel("Wildman–Crippen WLOGP")
    ax1.set_title("(a) BOILED-Egg Permeation Map (Daina & Zoete, 2016)", loc="left")
    ax1.grid(True, zorder=0)

    toxic_h = mpatches.Patch(color=CRIMSON, label="Clinical Risk (hERG/CYP ≥ 0.70)")
    safe_h = mpatches.Patch(color=TEAL, label="Safe Clinical Control")
    ax1.legend(handles=[
        mpatches.Patch(facecolor="#FEF3C7", edgecolor="#D97706", label="BBB Yolk"),
        mpatches.Patch(facecolor="white", edgecolor="#64748B", label="HIA White"),
        toxic_h,
        safe_h,
    ], loc="lower right", fontsize=6.6, frameon=True, facecolor="white", edgecolor="#CBD5E1")

    # --- Panel (b): Clinical Pharmacology Forest Plot ---
    names = [r["name"].split(" (")[0] for r in bench]
    y_pos = np.arange(len(names))[::-1]

    ax2.axvline(0.45, color=OCHRE, ls=":", lw=0.9, alpha=0.8)
    ax2.axvline(0.70, color=CRIMSON, ls="--", lw=0.9, alpha=0.8)

    for i, row in enumerate(bench):
        y = y_pos[i]
        # hERG point + 95% CI
        h_p, h_s = row["hERG_prob"], row["hERG_std"]
        h_lo = max(0.0, h_p - 1.96 * h_s)
        h_hi = min(1.0, h_p + 1.96 * h_s)
        ax2.errorbar(
            h_p, y + 0.14,
            xerr=[[h_p - h_lo], [h_hi - h_p]],
            fmt="o", color=CRIMSON, markersize=4.5, elinewidth=1.2, capsize=2,
            label="hERG (KCNH2) ± 95% CI" if i == 0 else None,
        )
        # CYP3A4 point + 95% CI
        c_p, c_s = row["CYP3A4_prob"], row["CYP3A4_std"]
        c_lo = max(0.0, c_p - 1.96 * c_s)
        c_hi = min(1.0, c_p + 1.96 * c_s)
        ax2.errorbar(
            c_p, y - 0.14,
            xerr=[[c_p - c_lo], [c_hi - c_p]],
            fmt="s", color=NAVY, markersize=4.2, elinewidth=1.2, capsize=2,
            label="CYP3A4 ± 95% CI" if i == 0 else None,
        )
        # Annotate Applicability Domain T_max on right margin
        ax2.text(
            1.02, y,
            f"T={row['max_tanimoto']:.2f}",
            va="center", ha="left", fontsize=6.8,
            color=TEAL if row["max_tanimoto"] >= 0.40 else (OCHRE if row["max_tanimoto"] >= 0.25 else CRIMSON),
        )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=7.8)
    ax2.set_xlim(0.0, 1.0)
    ax2.set_xlabel("Calibrated Inhibition Probability $\\bar{p} \\pm 1.96\\sigma_{\\mathrm{epi}}$")
    ax2.set_title("(b) Clinical Benchmark & Applicability Domain ($T_{\\max}$)", loc="left")
    ax2.grid(axis="x")
    ax2.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#CBD5E1")

    plt.tight_layout(w_pad=2.4)
    out_path = FIG_DIR / "fig3_boiled_egg_clinical.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    print("Reconstructing Bemis-Murcko scaffold test split...")
    test_mols, test_labels = reconstruct_scaffold_test_split()
    print(f"Loaded {len(test_mols)} held-out scaffold test compounds.")

    ensemble = ChEMBLADMETEnsemble(MODELS_DIR, device="cpu")
    preds = ensemble.predict_mols(test_mols)

    plot_fig1_roc_pr_cal(preds, test_labels)
    ad_stats = plot_fig2_ad_stratification(preds, test_labels)
    plot_fig3_boiled_egg_clinical(test_mols, ensemble.metadata)

    stats_path = PAPER_DIR / "ad_stratification_stats.json"
    stats_path.write_text(json.dumps(ad_stats, indent=2), encoding="utf-8")
    print(f"Saved AD stratification stats to {stats_path}")
    print(json.dumps(ad_stats, indent=2))


if __name__ == "__main__":
    main()

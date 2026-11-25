"""
ADMET-Engine v3.0 (CardioPhys-3D) — WP6 Reproducible Benchmarks & Figures
=========================================================================
Executes all real CPU benchmarks and generates publication-grade vector PDF
figures for the CardioPhys-3D manuscript:

  (a) Full FDA-CiPA 28 clinical validation:
        - Multi-Channel ORd-CiPA ODE (hERG + Cav1.2 + Nav1.5 late/peak)
        - Single-Channel hERG-only ODE ablation
        - Classical hERG IC50 / Cmax safety-margin baseline
        - Verapamil & Ranolazine multi-channel rescue verification
  (b) Monte-Carlo uncertainty propagation (`propagate_uncertainty_mc`) through
      the ORd-CiPA ODE across 6 representative CiPA-28 drugs (2 High, 2 Int, 2 Low)
      reporting qNet mean +/- 95% CI and P(TdP).
  (c) Live WP5 Closed-Loop Molecular Repair benchmark on 6 High/Intermediate-Risk
      CiPA-28 drugs (Bepridil, Terfenadine, Astemizole, Cisapride, Droperidol,
      Domperidone) using the real trained `ChEMBLADMETEnsemble` hERG oracle.
  (d) Publication-quality vector PDF figures in `paper_v3/figures/`:
        - fig1_cipa28_action_potentials.pdf
        - fig2_mc_uncertainty_qnet.pdf
        - fig3_wp5_repair_pareto.pdf
  (e) Structured JSON metrics saved to `paper_v3/wp6_results.json`.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from rdkit import Chem
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.chembl_ensemble import ChEMBLADMETEnsemble
from physiology.cipa28 import CIPA_28_DRUGS, get_cipa28_drug
from physiology.ohara_rudy import CardiomyocyteSimulator, ChannelBlockProfile
from repair.molecular_repair import ClosedLoopRepairAgent


# ---------------------------------------------------------------------------
# Publication Matplotlib Styling (Nature / Science Vector Aesthetic)
# ---------------------------------------------------------------------------

def _configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#cccccc",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
    })


def _ic50_to_pic50(ic50_uM: float) -> float:
    return 6.0 - math.log10(max(ic50_uM, 1e-12))


# ---------------------------------------------------------------------------
# Experiment (a): Full FDA-CiPA 28 Validation & Multi-Channel Rescue
# ---------------------------------------------------------------------------

def run_cipa28_benchmark(sim: CardiomyocyteSimulator) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Compare Single-Channel hERG vs Multi-Channel ORd-CiPA ODE across all 28
    FDA-CiPA reference drugs, plus Verapamil/Ranolazine rescue and AP traces.
    """
    def _t(vals: list[float | None]) -> torch.Tensor:
        return torch.tensor(
            [1e6 if v is None else float(v) for v in vals], dtype=torch.float32
        )

    cmax = _t([d.cmax_free_uM for d in CIPA_28_DRUGS])
    herg = _t([d.ic50_herg_uM for d in CIPA_28_DRUGS])
    cav12 = _t([d.ic50_cav12_uM for d in CIPA_28_DRUGS])
    nal = _t([d.ic50_nav15_late_uM for d in CIPA_28_DRUGS])
    nap = _t([d.ic50_nav15_peak_uM for d in CIPA_28_DRUGS])
    h_herg = _t([d.h_herg for d in CIPA_28_DRUGS])
    h_cav = _t([d.h_cav12 for d in CIPA_28_DRUGS])
    h_nal = _t([d.h_nav15_late for d in CIPA_28_DRUGS])
    h_nap = _t([d.h_nav15_peak for d in CIPA_28_DRUGS])

    # 1. Full Multi-Channel ORd-CiPA ODE (1x-4x Cmax_free)
    with torch.no_grad():
        mc_out = sim.simulate_cipa_protocol(
            cmax_free_uM=cmax,
            ic50_herg_uM=herg,
            ic50_cav12_uM=cav12,
            ic50_nav15_late_uM=nal,
            ic50_nav15_peak_uM=nap,
            h_herg=h_herg,
            h_cav12=h_cav,
            h_nav15_late=h_nal,
            h_nav15_peak=h_nap,
        )

        # 2. Single-Channel hERG-Only ODE Ablation (Cav1.2, Nav1.5 unblocked)
        no_block = torch.full_like(herg, 1e6)
        herg_out = sim.simulate_cipa_protocol(
            cmax_free_uM=cmax,
            ic50_herg_uM=herg,
            ic50_cav12_uM=no_block,
            ic50_nav15_late_uM=no_block,
            ic50_nav15_peak_uM=no_block,
            h_herg=h_herg,
        )

    risks = [d.tdp_risk for d in CIPA_28_DRUGS]
    y_hi_int = np.array([1 if r in ("High", "Intermediate") else 0 for r in risks])
    hi_lo_mask = np.array([r != "Intermediate" for r in risks])
    y_hi_lo = np.array([1 if r == "High" else 0 for r in risks if r != "Intermediate"])

    mc_qnet = mc_out["mean_qnet"].cpu().numpy()
    herg_qnet = herg_out["mean_qnet"].cpu().numpy()
    mc_apd90_1x = mc_out["apd90_1x_ms"].cpu().numpy()
    mc_apd90_max = mc_out["max_apd90_ms"].cpu().numpy()
    herg_apd90_max = herg_out["max_apd90_ms"].cpu().numpy()
    mc_tdp = mc_out["tdp_risk_score"].cpu().numpy()
    herg_tdp = herg_out["tdp_risk_score"].cpu().numpy()

    # Classical hERG margin score: log10(Cmax_free / IC50_hERG) (higher = more risk)
    herg_margin_score = np.array(
        [math.log10(d.cmax_free_uM / (d.ic50_herg_uM or 1e6)) for d in CIPA_28_DRUGS],
        dtype=np.float64,
    )

    auc_mc_all = float(roc_auc_score(y_hi_int, -mc_qnet))
    auc_mc_hilo = float(roc_auc_score(y_hi_lo, -mc_qnet[hi_lo_mask]))
    auc_herg_ode_all = float(roc_auc_score(y_hi_int, -herg_qnet))
    auc_herg_ode_hilo = float(roc_auc_score(y_hi_lo, -herg_qnet[hi_lo_mask]))
    auc_herg_margin_all = float(roc_auc_score(y_hi_int, herg_margin_score))
    auc_herg_margin_hilo = float(roc_auc_score(y_hi_lo, herg_margin_score[hi_lo_mask]))

    # Verapamil & Ranolazine rescue verification at 4x Cmax_free
    ver = get_cipa28_drug("Verapamil")
    ran = get_cipa28_drug("Ranolazine")
    with torch.no_grad():
        rescue_conc = torch.tensor(
            [4.0 * ver.cmax_free_uM, 4.0 * ver.cmax_free_uM, 4.0 * ran.cmax_free_uM, 4.0 * ran.cmax_free_uM],
            dtype=torch.float32,
        )
        rescue_block = ChannelBlockProfile.from_ic50_uM(
            conc_uM=rescue_conc,
            ic50_herg_uM=torch.tensor([ver.ic50_herg_uM, ver.ic50_herg_uM, ran.ic50_herg_uM, ran.ic50_herg_uM]),
            ic50_cav12_uM=torch.tensor([1e6, ver.ic50_cav12_uM, 1e6, ran.ic50_cav12_uM]),
            ic50_nav15_late_uM=torch.tensor([1e6, 1e6, 1e6, ran.ic50_nav15_late_uM]),
            ic50_nav15_peak_uM=torch.tensor([1e6, ver.ic50_nav15_peak_uM, 1e6, ran.ic50_nav15_peak_uM]),
            h_herg=torch.tensor([ver.h_herg, ver.h_herg, ran.h_herg, ran.h_herg]),
            h_cav12=torch.tensor([1.0, ver.h_cav12, 1.0, ran.h_cav12]),
            h_nav15_late=torch.tensor([1.0, 1.0, 1.0, ran.h_nav15_late]),
            h_nav15_peak=torch.tensor([1.0, ver.h_nav15_peak, 1.0, ran.h_nav15_peak]),
        )
        rescue_res = sim.simulate(rescue_block, return_traces=False)

    rescue_verification = {
        "Verapamil_4xCmax": {
            "herg_only_qnet": round(float(rescue_res.qnet[0].item()), 5),
            "multichannel_qnet": round(float(rescue_res.qnet[1].item()), 5),
            "delta_qnet": round(float((rescue_res.qnet[1] - rescue_res.qnet[0]).item()), 5),
            "herg_only_apd90_ms": round(float(rescue_res.apd90_ms[0].item()), 1),
            "multichannel_apd90_ms": round(float(rescue_res.apd90_ms[1].item()), 1),
            "delta_apd90_ms": round(float((rescue_res.apd90_ms[1] - rescue_res.apd90_ms[0]).item()), 1),
            "herg_only_tdp_risk": round(float(rescue_res.tdp_risk_score[0].item()), 4),
            "multichannel_tdp_risk": round(float(rescue_res.tdp_risk_score[1].item()), 4),
            "rescued": bool(
                rescue_res.qnet[1].item() > rescue_res.qnet[0].item()
                and rescue_res.apd90_ms[1].item() < rescue_res.apd90_ms[0].item()
            ),
        },
        "Ranolazine_4xCmax": {
            "herg_only_qnet": round(float(rescue_res.qnet[2].item()), 5),
            "multichannel_qnet": round(float(rescue_res.qnet[3].item()), 5),
            "delta_qnet": round(float((rescue_res.qnet[3] - rescue_res.qnet[2]).item()), 5),
            "herg_only_apd90_ms": round(float(rescue_res.apd90_ms[2].item()), 1),
            "multichannel_apd90_ms": round(float(rescue_res.apd90_ms[3].item()), 1),
            "delta_apd90_ms": round(float((rescue_res.apd90_ms[3] - rescue_res.apd90_ms[2]).item()), 1),
            "herg_only_tdp_risk": round(float(rescue_res.tdp_risk_score[2].item()), 4),
            "multichannel_tdp_risk": round(float(rescue_res.tdp_risk_score[3].item()), 4),
            "rescued": bool(
                rescue_res.qnet[3].item() > rescue_res.qnet[2].item()
                and rescue_res.apd90_ms[3].item() < rescue_res.apd90_ms[2].item()
            ),
        },
    }

    # Generate high-resolution action potential traces for Fig 1a:
    # Control, Verapamil (4x Cmax, multi-channel), Verapamil (4x Cmax, hERG-only),
    # Dofetilide (4x Cmax), Terfenadine (50x Cmax CYP3A4 DDI clinical exposure)
    dof = get_cipa28_drug("Dofetilide")
    terf = get_cipa28_drug("Terfenadine")
    trace_sim = CardiomyocyteSimulator(dt_ms=0.25, n_beats=2)
    with torch.no_grad():
        trace_block = ChannelBlockProfile.from_ic50_uM(
            conc_uM=torch.tensor(
                [0.0, 4.0 * ver.cmax_free_uM, 4.0 * ver.cmax_free_uM, 4.0 * dof.cmax_free_uM, 50.0 * terf.cmax_free_uM],
                dtype=torch.float32,
            ),
            ic50_herg_uM=torch.tensor([1e6, ver.ic50_herg_uM, ver.ic50_herg_uM, dof.ic50_herg_uM, terf.ic50_herg_uM]),
            ic50_cav12_uM=torch.tensor([1e6, ver.ic50_cav12_uM, 1e6, 1e6, terf.ic50_cav12_uM]),
            ic50_nav15_late_uM=torch.tensor([1e6, 1e6, 1e6, 1e6, terf.ic50_nav15_late_uM]),
            ic50_nav15_peak_uM=torch.tensor([1e6, ver.ic50_nav15_peak_uM, 1e6, 1e6, terf.ic50_nav15_peak_uM]),
            h_herg=torch.tensor([1.0, ver.h_herg, ver.h_herg, dof.h_herg, terf.h_herg]),
            h_cav12=torch.tensor([1.0, ver.h_cav12, 1.0, 1.0, terf.h_cav12]),
            h_nav15_late=torch.tensor([1.0, 1.0, 1.0, 1.0, terf.h_nav15_late]),
            h_nav15_peak=torch.tensor([1.0, ver.h_nav15_peak, 1.0, 1.0, terf.h_nav15_peak]),
        )
        trace_res = trace_sim.simulate(trace_block, return_traces=True, trace_stride=4)

    per_drug_rows: list[dict[str, Any]] = []
    for i, d in enumerate(CIPA_28_DRUGS):
        per_drug_rows.append({
            "name": d.name,
            "tdp_risk": d.tdp_risk,
            "cipa_split": d.cipa_split,
            "cmax_free_uM": d.cmax_free_uM,
            "multichannel_mean_qnet": round(float(mc_qnet[i]), 5),
            "herg_only_mean_qnet": round(float(herg_qnet[i]), 5),
            "multichannel_apd90_1x_ms": round(float(mc_apd90_1x[i]), 1),
            "multichannel_max_apd90_ms": round(float(mc_apd90_max[i]), 1),
            "herg_only_max_apd90_ms": round(float(herg_apd90_max[i]), 1),
            "multichannel_tdp_score": round(float(mc_tdp[i]), 4),
            "herg_only_tdp_score": round(float(herg_tdp[i]), 4),
        })

    metrics = {
        "n_drugs": len(CIPA_28_DRUGS),
        "roc_auc": {
            "multichannel_ode_high_vs_low": round(auc_mc_hilo, 4),
            "multichannel_ode_high_int_vs_low": round(auc_mc_all, 4),
            "herg_only_ode_high_vs_low": round(auc_herg_ode_hilo, 4),
            "herg_only_ode_high_int_vs_low": round(auc_herg_ode_all, 4),
            "herg_margin_high_vs_low": round(auc_herg_margin_hilo, 4),
            "herg_margin_high_int_vs_low": round(auc_herg_margin_all, 4),
            "delta_auc_high_vs_low": round(auc_mc_hilo - auc_herg_ode_hilo, 4),
            "delta_auc_high_int_vs_low": round(auc_mc_all - auc_herg_ode_all, 4),
        },
        "control_baseline": {
            "qnet_uC_uF": round(float(trace_res.qnet[0].item()), 5),
            "apd90_ms": round(float(trace_res.apd90_ms[0].item()), 1),
            "v_peak_mV": round(float(trace_res.v_peak_mV[0].item()), 1),
            "v_rest_mV": round(float(trace_res.v_rest_mV[0].item()), 1),
        },
        "rescue_verification": rescue_verification,
        "per_drug": per_drug_rows,
    }

    plot_data = {
        "time_ms": trace_res.time_ms.cpu().numpy(),
        "v_traces": trace_res.v_trace.cpu().numpy(),
        "trace_qnet": trace_res.qnet.cpu().numpy(),
        "trace_apd90": trace_res.apd90_ms.cpu().numpy(),
        "y_hi_int": y_hi_int,
        "y_hi_lo": y_hi_lo,
        "hi_lo_mask": hi_lo_mask,
        "mc_qnet": mc_qnet,
        "herg_qnet": herg_qnet,
        "herg_margin_score": herg_margin_score,
    }
    return metrics, plot_data


# ---------------------------------------------------------------------------
# Experiment (b): Monte-Carlo Uncertainty Propagation through ORd-CiPA ODE
# ---------------------------------------------------------------------------

def run_mc_uncertainty_benchmark(
    sim: CardiomyocyteSimulator,
    n_samples: int = 64,
    pic50_sigma: float = 0.35,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Propagate continuous Tobit-style pIC50 Gaussian uncertainty N(mu, sigma^2)
    through the ORd-CiPA ODE for 6 representative CiPA-28 drugs (2 High, 2 Int, 2 Low)
    at 4x Cmax_free.
    """
    selected_names = [
        "Dofetilide",   # High
        "Bepridil",     # High
        "Cisapride",    # Intermediate
        "Domperidone",  # Intermediate
        "Verapamil",    # Low (Cav1.2 rescue)
        "Ranolazine",   # Low (Nav1.5 late rescue)
    ]
    drugs = [get_cipa28_drug(n) for n in selected_names]
    conc_4x = torch.tensor([4.0 * d.cmax_free_uM for d in drugs], dtype=torch.float32)

    def _build_pic50_tensor(attr: str) -> tuple[torch.Tensor, torch.Tensor]:
        means: list[float] = []
        stds: list[float] = []
        for d in drugs:
            val = getattr(d, attr)
            if val is None:
                means.append(0.0)   # IC50 = 1e6 uM -> negligible block
                stds.append(0.0)
            else:
                means.append(_ic50_to_pic50(float(val)))
                stds.append(pic50_sigma)
        return torch.tensor(means, dtype=torch.float32), torch.tensor(stds, dtype=torch.float32)

    herg_mu, herg_sd = _build_pic50_tensor("ic50_herg_uM")
    cav_mu, cav_sd = _build_pic50_tensor("ic50_cav12_uM")
    nal_mu, nal_sd = _build_pic50_tensor("ic50_nav15_late_uM")
    nap_mu, nap_sd = _build_pic50_tensor("ic50_nav15_peak_uM")

    with torch.no_grad():
        mc = sim.propagate_uncertainty_mc(
            conc_uM=conc_4x,
            pic50_mean={
                "hERG": herg_mu,
                "Cav1.2": cav_mu,
                "Nav1.5_late": nal_mu,
                "Nav1.5_peak": nap_mu,
            },
            pic50_std={
                "hERG": herg_sd,
                "Cav1.2": cav_sd,
                "Nav1.5_late": nal_sd,
                "Nav1.5_peak": nap_sd,
            },
            n_samples=n_samples,
            seed=seed,
        )

    results: list[dict[str, Any]] = []
    for i, d in enumerate(drugs):
        q_mean = float(mc["qnet_mean"][i].item())
        q_std = float(mc["qnet_std"][i].item())
        ci95 = 1.96 * q_std
        results.append({
            "name": d.name,
            "tdp_risk": d.tdp_risk,
            "conc_4x_uM": round(4.0 * d.cmax_free_uM, 6),
            "n_samples": n_samples,
            "pic50_sigma": pic50_sigma,
            "qnet_mean": round(q_mean, 5),
            "qnet_std": round(q_std, 5),
            "qnet_ci95_low": round(q_mean - ci95, 5),
            "qnet_ci95_high": round(q_mean + ci95, 5),
            "apd90_mean_ms": round(float(mc["apd90_mean_ms"][i].item()), 1),
            "apd90_std_ms": round(float(mc["apd90_std_ms"][i].item()), 1),
            "tdp_prob_mean": round(float(mc["tdp_prob_mean"][i].item()), 4),
            "tdp_prob_std": round(float(mc["tdp_prob_std"][i].item()), 4),
            "ead_probability": round(float(mc["ead_probability"][i].item()), 4),
        })
    return results


# ---------------------------------------------------------------------------
# Experiment (c): Live WP5 Closed-Loop Molecular Repair Benchmark
# ---------------------------------------------------------------------------

def run_wp5_repair_benchmark() -> list[dict[str, Any]]:
    """
    Execute the WP5 Closed-Loop Molecular Repair Agent on 6 High/Intermediate-Risk
    CiPA-28 drugs using the real trained ChEMBL v37 ensemble (`hERG_inhibition`).
    """
    ensemble = ChEMBLADMETEnsemble(bundle_dir=PROJECT_ROOT / "models", device="cpu")

    def herg_oracle(smiles_list: list[str]) -> list[float]:
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        preds = ensemble.predict_mols(mols)
        return [float(p["predictions"]["hERG_inhibition"]) for p in preds]

    target_drugs = [
        "Bepridil",
        "Terfenadine",
        "Astemizole",
        "Cisapride",
        "Droperidol",
        "Domperidone",
    ]

    agent = ClosedLoopRepairAgent(
        risk_oracle=herg_oracle,
        min_tanimoto=0.55,
        max_sa_score=5.0,
        min_qed=0.25,
        require_zero_pains=True,
    )

    records: list[dict[str, Any]] = []
    for name in target_drugs:
        drug = get_cipa28_drug(name)
        report = agent.repair(drug.smiles, n_rounds=2, beam_width=5, top_k=5)
        seed = report["seed"]
        cands = report["repaired_candidates"]
        best = cands[0] if cands else None

        records.append({
            "name": drug.name,
            "tdp_risk": drug.tdp_risk,
            "seed_smiles": seed["smiles"],
            "seed_herg_risk": float(seed["risk_score"]),
            "seed_clogp": float(seed["clogp"]),
            "seed_tpsa": float(seed["tpsa"]),
            "seed_qed": float(seed["qed"]),
            "seed_sa": float(seed["sa_score"]),
            "total_evaluated": int(report["total_evaluated"]),
            "repaired_smiles": best.smiles if best else seed["smiles"],
            "repaired_herg_risk": float(best.risk_score) if best else float(seed["risk_score"]),
            "delta_risk": float(best.delta_risk) if best else 0.0,
            "relative_reduction_pct": round(
                100.0 * (-best.delta_risk) / max(float(seed["risk_score"]), 1e-6), 1
            ) if best else 0.0,
            "tanimoto": float(best.tanimoto) if best else 1.0,
            "repaired_clogp": float(best.clogp) if best else float(seed["clogp"]),
            "repaired_tpsa": float(best.tpsa) if best else float(seed["tpsa"]),
            "repaired_qed": float(best.qed) if best else float(seed["qed"]),
            "repaired_sa": float(best.sa_score) if best else float(seed["sa_score"]),
            "pains_count": int(best.pains_count) if best else 0,
            "edit_path": list(best.edit_path) if best else [],
        })
    return records


# ---------------------------------------------------------------------------
# Publication Figures (Vector PDFs in paper_v3/figures/)
# ---------------------------------------------------------------------------

def _plot_fig1_cipa28(
    cipa_metrics: dict[str, Any],
    plot_data: dict[str, Any],
    out_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.25), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # --- Panel (a): Action Potential Traces V_m(t) ---
    t = plot_data["time_ms"]
    v = plot_data["v_traces"]
    q = plot_data["trace_qnet"]
    apd = plot_data["trace_apd90"]

    mask = t <= 650.0
    t_sub = t[mask]

    ax1.plot(
        t_sub, v[0][mask], color="#1f2937", lw=2.0,
        label=f"Control (APD90={apd[0]:.0f} ms, qNet={q[0]:.3f})",
    )
    ax1.plot(
        t_sub, v[1][mask], color="#0d9488", lw=2.0,
        label=f"Verapamil 4x Cmax Multi-Ch (APD90={apd[1]:.0f} ms, qNet={q[1]:.3f})",
    )
    ax1.plot(
        t_sub, v[2][mask], color="#0d9488", lw=1.5, ls="--", alpha=0.80,
        label=f"Verapamil 4x Cmax hERG-only (APD90={apd[2]:.0f} ms, qNet={q[2]:.3f})",
    )
    ax1.plot(
        t_sub, v[3][mask], color="#dc2626", lw=2.0,
        label=f"Dofetilide 4x Cmax (APD90={apd[3]:.0f} ms, qNet={q[3]:.3f})",
    )
    ax1.plot(
        t_sub, v[4][mask], color="#d97706", lw=1.9, ls="-.",
        label=f"Terfenadine DDI (APD90={apd[4]:.0f} ms, qNet={q[4]:.3f})",
    )

    ax1.set_title("(a) Human Ventricular Action Potential (ORd-CiPA ODE)")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Membrane Potential $V_m$ (mV)")
    ax1.set_xlim(0, 650)
    ax1.set_ylim(-98, 52)
    ax1.grid(True, alpha=0.25, lw=0.6)
    ax1.legend(loc="upper right", fontsize=7.3)

    # --- Panel (b): FDA-CiPA 28 ROC Curves ---
    y_all = plot_data["y_hi_int"]
    y_hl = plot_data["y_hi_lo"]
    hl_mask = plot_data["hi_lo_mask"]
    mc_q = plot_data["mc_qnet"]
    hg_q = plot_data["herg_qnet"]

    fpr_mc_hl, tpr_mc_hl, _ = roc_curve(y_hl, -mc_q[hl_mask])
    fpr_mc_all, tpr_mc_all, _ = roc_curve(y_all, -mc_q)
    fpr_hg_hl, tpr_hg_hl, _ = roc_curve(y_hl, -hg_q[hl_mask])
    fpr_hg_all, tpr_hg_all, _ = roc_curve(y_all, -hg_q)

    aucs = cipa_metrics["roc_auc"]
    ax2.plot(
        fpr_mc_hl, tpr_mc_hl, color="#0d9488", lw=2.2,
        label=f"Multi-Ch ORd-CiPA (High vs Low, AUC={aucs['multichannel_ode_high_vs_low']:.3f})",
    )
    ax2.plot(
        fpr_mc_all, tpr_mc_all, color="#2563eb", lw=2.0,
        label=f"Multi-Ch ORd-CiPA (High+Int vs Low, AUC={aucs['multichannel_ode_high_int_vs_low']:.3f})",
    )
    ax2.plot(
        fpr_hg_hl, tpr_hg_hl, color="#dc2626", lw=1.7, ls="--",
        label=f"hERG-Only ODE (High vs Low, AUC={aucs['herg_only_ode_high_vs_low']:.3f})",
    )
    ax2.plot(
        fpr_hg_all, tpr_hg_all, color="#d97706", lw=1.7, ls=":",
        label=f"hERG-Only ODE (High+Int vs Low, AUC={aucs['herg_only_ode_high_int_vs_low']:.3f})",
    )
    ax2.plot([0, 1], [0, 1], color="#9ca3af", lw=1.0, ls="--", alpha=0.7)

    ax2.set_title("(b) Official FDA-CiPA 28 Clinical TdP Classification")
    ax2.set_xlabel("False Positive Rate (1 - Specificity)")
    ax2.set_ylabel("True Positive Rate (Sensitivity)")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.04)
    ax2.grid(True, alpha=0.25, lw=0.6)
    ax2.legend(loc="lower right", fontsize=7.4)

    fig.tight_layout(pad=1.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_fig2_mc_uncertainty(
    mc_results: list[dict[str, Any]],
    control_qnet: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.4))

    color_map = {
        "High": "#dc2626",
        "Intermediate": "#d97706",
        "Low": "#0d9488",
    }

    y_pos = np.arange(len(mc_results))[::-1]
    for y, row in zip(y_pos, mc_results):
        col = color_map[row["tdp_risk"]]
        mean_q = row["qnet_mean"]
        ci95 = 1.96 * row["qnet_std"]

        ax.errorbar(
            mean_q,
            y,
            xerr=ci95,
            fmt="o",
            color=col,
            ecolor=col,
            elinewidth=2.2,
            capsize=5,
            capthick=1.6,
            markersize=7.5,
            markeredgecolor="white",
            markeredgewidth=0.9,
            zorder=4,
        )
        # Annotate P(TdP) right beside the upper CI bound
        ax.text(
            mean_q + ci95 + 0.0025,
            y,
            f"P(TdP)={row['tdp_prob_mean']:.2f} ± {row['tdp_prob_std']:.2f}",
            va="center",
            ha="left",
            fontsize=8.0,
            color="#374151",
        )

    ax.axvline(
        0.1675, color="#b91c1c", ls="--", lw=1.4, alpha=0.85,
        label="CiPA TdP Decision Boundary (0.1675 uC/uF)",
    )
    ax.axvline(
        control_qnet, color="#1f2937", ls=":", lw=1.4, alpha=0.85,
        label=f"Control Baseline ({control_qnet:.4f} uC/uF)",
    )

    # Shaded proarrhythmic zone
    xmin = min(r["qnet_ci95_low"] for r in mc_results) - 0.012
    xmax = max(r["qnet_ci95_high"] for r in mc_results) + 0.052
    ax.axvspan(xmin, 0.1675, color="#fee2e2", alpha=0.35, zorder=1)
    ax.set_xlim(xmin, xmax)

    labels = [f"{r['name']} ({r['tdp_risk']})" for r in mc_results]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Ventricular Net Charge Transfer $q_{\\mathrm{Net}}$ ($\\mu$C/$\\mu$F) — Mean $\\pm$ 95% CI")
    ax.set_title("Monte-Carlo Uncertainty Propagation Through ORd-CiPA ODE (4x $C_{\\max,\\mathrm{free}}$, $S=64$)")
    ax.grid(True, axis="x", alpha=0.25, lw=0.6)

    # Legend with risk category markers + vertical lines
    handles, leg_labels = ax.get_legend_handles_labels()
    for cat in ("High", "Intermediate", "Low"):
        handles.append(mpatches.Patch(color=color_map[cat], label=f"{cat} TdP Risk"))
    ax.legend(handles=handles, loc="lower left", fontsize=7.6, ncol=2)

    fig.tight_layout(pad=1.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_fig3_wp5_repair(
    repair_results: list[dict[str, Any]],
    out_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.4), gridspec_kw={"width_ratios": [1.1, 1.0]})

    # --- Panel (a): Before vs After hERG Cardiotoxicity Risk ---
    names = [r["name"] for r in repair_results]
    before = [r["seed_herg_risk"] for r in repair_results]
    after = [r["repaired_herg_risk"] for r in repair_results]
    tanimotos = [r["tanimoto"] for r in repair_results]

    x = np.arange(len(names))
    w = 0.36
    bars1 = ax1.bar(x - w / 2, before, width=w, color="#ef4444", alpha=0.88, label="Original CiPA Drug")
    bars2 = ax1.bar(x + w / 2, after, width=w, color="#0d9488", alpha=0.92, label="WP5 Repaired Analog")

    for i, (b_val, a_val, tan) in enumerate(zip(before, after, tanimotos)):
        delta = a_val - b_val
        ax1.text(
            x[i] + w / 2,
            a_val + 0.025,
            f"{delta:+.2f}\n(T={tan:.2f})",
            ha="center",
            va="bottom",
            fontsize=7.1,
            color="#111827",
            fontweight="bold",
        )

    ax1.axhline(0.50, color="#6b7280", ls="--", lw=1.1, alpha=0.75, label="hERG Activity Threshold (0.50)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=22, ha="right")
    ax1.set_ylim(0.0, 1.18)
    ax1.set_ylabel("Predicted hERG Risk $P(\\mathrm{active})$")
    ax1.set_title("(a) Closed-Loop Detoxification (ChEMBL v37 Oracle)")
    ax1.grid(True, axis="y", alpha=0.25, lw=0.6)
    ax1.legend(loc="upper right", fontsize=7.6)

    # --- Panel (b): Physicochemical Shift Trajectories (cLogP vs TPSA) ---
    # Shade Pfizer 3/75 promiscuity risk zone (cLogP > 3.0 and TPSA < 75 A^2)
    ax2.axvspan(0, 75, ymin=0, ymax=1, color="#fef2f2", alpha=0.55, zorder=0)
    tpsa_all = [r["seed_tpsa"] for r in repair_results] + [r["repaired_tpsa"] for r in repair_results]
    clogp_all = [r["seed_clogp"] for r in repair_results] + [r["repaired_clogp"] for r in repair_results]
    xmin = max(0.0, min(tpsa_all) - 12.0)
    xmax = max(tpsa_all) + 18.0
    ymin = min(clogp_all) - 0.8
    ymax = max(clogp_all) + 0.8

    rect = mpatches.Rectangle(
        (xmin, 3.0), max(0.0, 75.0 - xmin), max(0.0, ymax - 3.0),
        facecolor="#fecaca", alpha=0.45, edgecolor="#dc2626", lw=0.9, ls="--", zorder=1,
        label="Pfizer 3/75 Risk Zone (cLogP>3, TPSA<75)",
    )
    ax2.add_patch(rect)

    for idx, r in enumerate(repair_results):
        x0, y0 = r["seed_tpsa"], r["seed_clogp"]
        x1, y1 = r["repaired_tpsa"], r["repaired_clogp"]
        ax2.scatter(x0, y0, color="#dc2626", s=52, zorder=4, label="Original" if idx == 0 else None)
        ax2.scatter(x1, y1, color="#0d9488", marker="D", s=52, zorder=5, label="Repaired" if idx == 0 else None)
        ax2.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="#374151", lw=1.3, shrinkA=4, shrinkB=4),
            zorder=3,
        )
        # Offset label slightly to avoid overlaps
        dx = 3.0 if x1 >= x0 else -3.0
        dy = 0.16 if (idx % 2 == 0) else -0.24
        ax2.text(x1 + dx, y1 + dy, r["name"], fontsize=7.6, color="#1f2937", fontweight="bold")

    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.set_xlabel("Topological Polar Surface Area TPSA ($\\AA^2$)")
    ax2.set_ylabel("Lipophilicity (Wildman-Crippen cLogP)")
    ax2.set_title("(b) Physicochemical Shift Out of hERG Risk Space")
    ax2.grid(True, alpha=0.25, lw=0.6)
    ax2.legend(loc="lower left", fontsize=7.4)

    fig.tight_layout(pad=1.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Master Runner
# ---------------------------------------------------------------------------

def run_all_wp6_benchmarks(
    output_dir: Path | None = None,
    mc_samples: int = 64,
) -> dict[str, Any]:
    """Run all WP6 experiments, save `wp6_results.json`, and render all 3 PDF figures."""
    _configure_matplotlib()
    t0 = time.perf_counter()

    out_dir = output_dir or (PROJECT_ROOT / "paper_v3")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sim = CardiomyocyteSimulator(dt_ms=0.4, n_beats=1)

    print("[WP6] (a) Running full FDA-CiPA 28 validation (Multi-Channel vs hERG-Only)...")
    cipa_metrics, plot_data = run_cipa28_benchmark(sim)

    print(f"[WP6] (b) Running Monte-Carlo uncertainty propagation (S={mc_samples})...")
    mc_results = run_mc_uncertainty_benchmark(sim, n_samples=mc_samples, pic50_sigma=0.35, seed=42)

    print("[WP6] (c) Running live WP5 Closed-Loop Molecular Repair on 6 CiPA-28 drugs...")
    repair_results = run_wp5_repair_benchmark()

    # Render publication figures
    fig1_path = fig_dir / "fig1_cipa28_action_potentials.pdf"
    fig2_path = fig_dir / "fig2_mc_uncertainty_qnet.pdf"
    fig3_path = fig_dir / "fig3_wp5_repair_pareto.pdf"

    print("[WP6] Rendering vector PDF figures...")
    _plot_fig1_cipa28(cipa_metrics, plot_data, fig1_path)
    _plot_fig2_mc_uncertainty(mc_results, cipa_metrics["control_baseline"]["qnet_uC_uF"], fig2_path)
    _plot_fig3_wp5_repair(repair_results, fig3_path)

    mean_delta = float(np.mean([r["delta_risk"] for r in repair_results]))
    mean_tan = float(np.mean([r["tanimoto"] for r in repair_results]))
    mean_d_clogp = float(np.mean([r["repaired_clogp"] - r["seed_clogp"] for r in repair_results]))
    mean_d_tpsa = float(np.mean([r["repaired_tpsa"] - r["seed_tpsa"] for r in repair_results]))

    elapsed = round(time.perf_counter() - t0, 2)
    payload: dict[str, Any] = {
        "benchmark": "ADMET-Engine v3.0 (CardioPhys-3D) WP6 Ablations",
        "elapsed_seconds": elapsed,
        "cipa28_validation": cipa_metrics,
        "mc_uncertainty_propagation": mc_results,
        "wp5_closed_loop_repair": {
            "n_drugs": len(repair_results),
            "summary": {
                "mean_delta_risk": round(mean_delta, 4),
                "mean_tanimoto": round(mean_tan, 3),
                "mean_delta_clogp": round(mean_d_clogp, 2),
                "mean_delta_tpsa": round(mean_d_tpsa, 1),
                "zero_pains_rate": round(
                    sum(1 for r in repair_results if r["pains_count"] == 0) / max(len(repair_results), 1), 3
                ),
            },
            "drugs": repair_results,
        },
        "figures": {
            "fig1": str(fig1_path.relative_to(PROJECT_ROOT)),
            "fig2": str(fig2_path.relative_to(PROJECT_ROOT)),
            "fig3": str(fig3_path.relative_to(PROJECT_ROOT)),
        },
    }

    results_path = out_dir / "wp6_results.json"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[WP6] Done in {elapsed:.1f}s -> saved {results_path}")
    return payload


if __name__ == "__main__":
    run_all_wp6_benchmarks()

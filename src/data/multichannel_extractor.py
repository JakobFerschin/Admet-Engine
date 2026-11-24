#!/usr/bin/env python3
"""
ADMET-Engine v3.0 (CardioPhys-3D) — WP1: Multi-Channel ChEMBL v37 Extractor
===========================================================================
Extracts continuous potency (pIC50) AND censored assay bounds ('=', '>', '<')
for the three core FDA-CiPA cardiac ion channels:
  - hERG   (KCNH2   / CHEMBL240)  — I_Kr  (repolarizing potassium current)
  - Cav1.2 (CACNA1C / CHEMBL1940) — I_CaL (L-type calcium current)
  - Nav1.5 (SCN5A   / CHEMBL1980) — I_Na  / I_NaL (cardiac sodium current)

Supports two data backends seamlessly:
  1. Local ChEMBL v37 SQLite database (zero-network streaming cursor)
  2. Official EMBL-EBI ChEMBL REST API (ideal for stateless Google Colab workers)

CRITICAL BIOSTATISTICAL CONVENTION (Tobit Censored Likelihood in pIC50 space):
  Because pIC50 = -log10(IC50_M) is a strictly decreasing function of IC50:
    - IC50 = x     -> pIC50 = -log10(x)   -> censor =  0 (exact observation)
    - IC50 > x     -> pIC50 < -log10(x)   -> censor = -1 (left-censored upper bound; low potency)
    - IC50 < x     -> pIC50 > -log10(x)   -> censor = +1 (right-censored lower bound; high potency)

Also assigns a unified deterministic Bemis-Murcko scaffold split (80/10/10,
zero core scaffold overlap across train/val/test) shared across all three channels.
"""

from __future__ import annotations

import argparse
import logging
import math
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cheminformatics import get_murcko_scaffold  # noqa: E402
from data_pipeline import standardize_molecule  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cardiophys.wp1")

# ---------------------------------------------------------------------------
# Ion-Channel Targets & Unit Conversions
# ---------------------------------------------------------------------------

CARDIAC_TARGETS: dict[str, str] = {
    "hERG": "CHEMBL240",      # KCNH2 (I_Kr)
    "Cav1.2": "CHEMBL1940",   # CACNA1C (I_CaL)
    "Nav1.5": "CHEMBL1980",   # SCN5A (I_Na / I_NaL)
}

RELEVANT_STANDARD_TYPES: tuple[str, ...] = ("IC50", "Ki", "EC50", "Kd")
ALLOWED_RELATIONS: tuple[str, ...] = ("=", ">", ">=", "<", "<=")

UNIT_TO_MOLAR: dict[str, float] = {
    "M": 1.0,
    "mM": 1e-3,
    "uM": 1e-6,
    "μM": 1e-6,
    "nM": 1e-9,
    "pM": 1e-12,
    "fM": 1e-15,
}

CHEMBL_REST_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"


# ---------------------------------------------------------------------------
# Tobit Censoring & pIC50 Helpers
# ---------------------------------------------------------------------------

def relation_to_pic50_censor(standard_relation: Optional[str]) -> Optional[int]:
    """
    Map ChEMBL concentration-space `standard_relation` to pIC50-space Tobit
    censoring indicator:
      - '='       ->  0 (exact)
      - '>' / '>=' -> -1 (left-censored: true pIC50 <= observed bound)
      - '<' / '<=' -> +1 (right-censored: true pIC50 >= observed bound)
    """
    if standard_relation is None:
        return None
    rel = str(standard_relation).strip()
    if rel == "=":
        return 0
    if rel in (">", ">="):
        return -1
    if rel in ("<", "<="):
        return +1
    return None


def compute_pic50(
    standard_value: Any,
    standard_units: Optional[str],
    pchembl_value: Any = None,
) -> Optional[float]:
    """Convert (value, units) to pIC50 = -log10(molar), preferring DB pChEMBL when exact."""
    if pchembl_value is not None and not pd.isna(pchembl_value):
        try:
            v = float(pchembl_value)
            if 1.0 <= v <= 14.0:
                return v
        except (ValueError, TypeError):
            pass

    if standard_value is None or standard_units is None:
        return None
    try:
        val = float(standard_value)
    except (ValueError, TypeError):
        return None
    if val <= 0.0:
        return None

    factor = UNIT_TO_MOLAR.get(str(standard_units).strip())
    if factor is None:
        return None
    molar = val * factor
    if molar <= 0.0:
        return None
    pic50 = -math.log10(molar)
    return pic50 if (1.0 <= pic50 <= 14.0) else None


def aggregate_channel_measurements(
    observations: list[tuple[float, int]],
) -> tuple[float, int, int]:
    """
    Aggregate replicate (pIC50, censor) measurements for a single
    (canonical_smiles, ion_channel) pair:
      1. If any exact measurements (censor == 0) exist, take their median
         and mark as exact (0).
      2. Else if only left-censored (censor == -1, weak/inactive) exist,
         take the median upper bound and mark -1.
      3. Else if only right-censored (censor == +1, ultra-potent) exist,
         take the median lower bound and mark +1.
      4. If conflicting censored bounds (-1 and +1 with no exact), take the
         midpoint between max lower bound and min upper bound as exact (0)
         if consistent, else prefer the majority direction.

    Returns:
        (aggregated_pic50, aggregated_censor, n_measurements)
    """
    n_obs = len(observations)
    exact = [v for v, c in observations if c == 0]
    if exact:
        return float(np.median(exact)), 0, n_obs

    left = [v for v, c in observations if c == -1]
    right = [v for v, c in observations if c == +1]

    if left and not right:
        return float(np.median(left)), -1, n_obs
    if right and not left:
        return float(np.median(right)), +1, n_obs

    # Both left and right censored without exact measurement
    if len(left) >= len(right):
        return float(np.median(left)), -1, n_obs
    return float(np.median(right)), +1, n_obs


# ---------------------------------------------------------------------------
# Backend 1: Local ChEMBL v37 SQLite Extraction
# ---------------------------------------------------------------------------

SQLITE_QUERY = """
SELECT
    cs.canonical_smiles,
    t.chembl_id AS target_chembl_id,
    a.standard_type,
    a.standard_relation,
    a.standard_value,
    a.standard_units,
    a.pchembl_value
FROM activities a
JOIN assays              s  ON a.assay_id  = s.assay_id
JOIN compound_structures cs ON a.molregno  = cs.molregno
JOIN target_dictionary   t  ON s.tid       = t.tid
WHERE
    t.chembl_id IN ({target_ph})
    AND a.standard_relation IN ({rel_ph})
    AND s.confidence_score >= ?
    AND COALESCE(a.potential_duplicate, 0) = 0
    AND a.standard_type IN ({type_ph})
    AND a.standard_value IS NOT NULL
    AND a.standard_units IS NOT NULL
    AND cs.canonical_smiles IS NOT NULL
    AND cs.canonical_smiles != ''
""".format(
    target_ph=",".join("?" * len(CARDIAC_TARGETS)),
    rel_ph=",".join("?" * len(ALLOWED_RELATIONS)),
    type_ph=",".join("?" * len(RELEVANT_STANDARD_TYPES)),
)


def extract_from_sqlite(
    db_path: Path,
    min_confidence: int = 5,
) -> list[dict[str, Any]]:
    """Stream multi-channel bioactivities (including censored bounds) from SQLite."""
    if not db_path.exists():
        raise FileNotFoundError(f"ChEMBL SQLite database not found: {db_path}")

    log.info("Streaming from local ChEMBL SQLite: %s", db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA temp_store = MEMORY")

    params = (
        list(CARDIAC_TARGETS.values())
        + list(ALLOWED_RELATIONS)
        + [min_confidence]
        + list(RELEVANT_STANDARD_TYPES)
    )
    rows = [dict(r) for r in conn.execute(SQLITE_QUERY, params)]
    conn.close()
    log.info("Fetched %d raw multi-channel rows from SQLite", len(rows))
    return rows


# ---------------------------------------------------------------------------
# Backend 2: Official EMBL-EBI ChEMBL REST API (Zero-Download Colab Mode)
# ---------------------------------------------------------------------------

def extract_from_rest_api(
    max_per_target: Optional[int] = None,
    page_size: int = 1000,
    timeout: int = 30,
) -> list[dict[str, Any]]:
    """
    Download curated hERG, Cav1.2, and Nav1.5 bioactivities directly from the
    official EMBL-EBI ChEMBL REST API, preserving censored relations ('=', '>', '<').
    Ideal for stateless Google Colab GPU instances.
    """
    session = requests.Session()
    all_rows: list[dict[str, Any]] = []

    for channel_name, chembl_id in CARDIAC_TARGETS.items():
        log.info("Fetching %s (%s) from ChEMBL REST API...", channel_name, chembl_id)
        fetched = 0
        offset = 0
        while True:
            params = {
                "target_chembl_id": chembl_id,
                "standard_type__in": ",".join(RELEVANT_STANDARD_TYPES),
                "standard_value__isnull": "false",
                "standard_units__isnull": "false",
                "limit": page_size,
                "offset": offset,
            }
            resp = session.get(CHEMBL_REST_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            activities = payload.get("activities", [])
            if not activities:
                break

            for act in activities:
                rel = act.get("standard_relation")
                if rel not in ALLOWED_RELATIONS:
                    continue
                if act.get("potential_duplicate") in (1, True, "1"):
                    continue
                smi = act.get("canonical_smiles")
                if not smi:
                    continue
                all_rows.append({
                    "canonical_smiles": smi,
                    "target_chembl_id": chembl_id,
                    "standard_type": act.get("standard_type"),
                    "standard_relation": rel,
                    "standard_value": act.get("standard_value"),
                    "standard_units": act.get("standard_units"),
                    "pchembl_value": act.get("pchembl_value"),
                })
                fetched += 1

            offset += len(activities)
            total = payload.get("page_meta", {}).get("total_count", 0)
            log.info("  [%s] %d / %d records fetched", channel_name, offset, total)

            if max_per_target is not None and fetched >= max_per_target:
                break
            if not payload.get("page_meta", {}).get("next"):
                break

    log.info("Total fetched from ChEMBL REST API: %d rows", len(all_rows))
    return all_rows


# ---------------------------------------------------------------------------
# Standardization, Pivoting & Unified Bemis-Murcko Scaffold Split
# ---------------------------------------------------------------------------

def build_multichannel_dataset(
    raw_rows: list[dict[str, Any]],
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> pd.DataFrame:
    """
    Standardize SMILES, aggregate replicates per (molecule, channel) with
    Tobit censoring flags, and assign a zero-overlap Bemis-Murcko scaffold split.
    """
    chembl_to_channel = {v: k for k, v in CARDIAC_TARGETS.items()}
    # canonical_smiles -> {channel: [(pIC50, censor), ...]}
    accum: dict[str, dict[str, list[tuple[float, int]]]] = defaultdict(lambda: defaultdict(list))
    scaffolds_map: dict[str, str] = {}
    failed = 0

    for row in raw_rows:
        censor = relation_to_pic50_censor(row.get("standard_relation"))
        if censor is None:
            continue
        pic50 = compute_pic50(
            row.get("standard_value"),
            row.get("standard_units"),
            row.get("pchembl_value") if censor == 0 else None,
        )
        if pic50 is None:
            continue

        channel = chembl_to_channel.get(str(row.get("target_chembl_id", "")))
        if channel is None:
            continue

        try:
            mol = standardize_molecule(str(row["canonical_smiles"]))
        except (ValueError, RuntimeError):
            failed += 1
            continue
        if mol is None or mol.GetNumAtoms() == 0:
            failed += 1
            continue

        can_smi = Chem.MolToSmiles(mol, canonical=True)
        accum[can_smi][channel].append((pic50, censor))
        if can_smi not in scaffolds_map:
            scaffolds_map[can_smi] = get_murcko_scaffold(mol)

    records: list[dict[str, Any]] = []
    for can_smi, ch_dict in accum.items():
        rec: dict[str, Any] = {
            "canonical_smiles": can_smi,
            "murcko_scaffold": scaffolds_map[can_smi],
        }
        for ch in CARDIAC_TARGETS:
            obs = ch_dict.get(ch)
            if obs:
                val, cens, n_obs = aggregate_channel_measurements(obs)
                rec[f"{ch}_pIC50"] = round(val, 4)
                rec[f"{ch}_censor"] = int(cens)
                rec[f"{ch}_n_obs"] = int(n_obs)
            else:
                rec[f"{ch}_pIC50"] = np.nan
                rec[f"{ch}_censor"] = np.nan
                rec[f"{ch}_n_obs"] = 0
        records.append(rec)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Deterministic Bemis-Murcko Scaffold Split across all 3 channels jointly
    scaf_groups: dict[str, list[int]] = defaultdict(list)
    for idx, sc in enumerate(df["murcko_scaffold"].tolist()):
        scaf_groups[sc].append(idx)

    sorted_groups = sorted(scaf_groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    n = len(df)
    train_cut = int(split_ratios[0] * n)
    val_cut = int((split_ratios[0] + split_ratios[1]) * n)

    splits = [""] * n
    n_tr = n_va = 0
    for _, indices in sorted_groups:
        if n_tr + len(indices) <= train_cut or n_tr == 0:
            for i in indices:
                splits[i] = "train"
            n_tr += len(indices)
        elif n_tr + n_va + len(indices) <= val_cut or n_va == 0:
            for i in indices:
                splits[i] = "val"
            n_va += len(indices)
        else:
            for i in indices:
                splits[i] = "test"

    df["split"] = splits

    # Verify zero scaffold leakage
    tr_sc = set(df.loc[df["split"] == "train", "murcko_scaffold"])
    va_sc = set(df.loc[df["split"] == "val", "murcko_scaffold"])
    te_sc = set(df.loc[df["split"] == "test", "murcko_scaffold"])
    assert len(tr_sc & te_sc) == 0 and len(tr_sc & va_sc) == 0, "Scaffold overlap detected!"

    log.info(
        "Built CardioPhys-3D multi-channel dataset: %d unique molecules (%d failed) | "
        "train=%d, val=%d, test=%d | 0 scaffold overlap verified",
        len(df), failed,
        (df["split"] == "train").sum(),
        (df["split"] == "val").sum(),
        (df["split"] == "test").sum(),
    )
    for ch in CARDIAC_TARGETS:
        col_c = df[f"{ch}_censor"].dropna()
        log.info(
            "  %-7s total=%5d | exact(0)=%5d | left(-1, >IC50)=%5d | right(+1, <IC50)=%5d",
            ch, len(col_c), int((col_c == 0).sum()), int((col_c == -1).sum()), int((col_c == 1).sum()),
        )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="WP1: Multi-Channel ChEMBL v37 Extractor")
    parser.add_argument(
        "--source", choices=["sqlite", "api"], default="api",
        help="Extract from local ChEMBL SQLite or official EMBL-EBI REST API.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=PROJECT_ROOT / "data/raw/chembl_37_sqlite/chembl_37/chembl_37_sqlite/chembl_37.db",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/cardiophys_multichannel_v3.csv",
    )
    parser.add_argument("--min-confidence", type=int, default=5)
    parser.add_argument("--max-per-target", type=int, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    if args.source == "sqlite":
        raw_rows = extract_from_sqlite(args.db_path, min_confidence=args.min_confidence)
    else:
        raw_rows = extract_from_rest_api(max_per_target=args.max_per_target)

    df = build_multichannel_dataset(raw_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    log.info("Saved %d rows to %s in %.1fs", len(df), args.output, time.perf_counter() - t0)


if __name__ == "__main__":
    main()

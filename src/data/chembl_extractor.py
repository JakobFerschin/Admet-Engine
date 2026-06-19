#!/usr/bin/env python3
"""
chembl_extractor.py — ChEMBL v37 Big Data ETL
Phase 7.1: Replace mock data with real bioactivity assays.

Targets:  hERG (CHEMBL240, tid=165), CYP3A4 (CHEMBL340, tid=17045)

Strategy:
  - SQL-level JOIN + WHERE to keep the working set small
  - sqlite3 cursor iteration (never load full DB into RAM)
  - Per-molecule median aggregation per target
  - pChEMBL threshold → binary label (1 = toxic risk)

Output: data/processed/chembl_admet_training.csv
Columns: canonical_smiles, hERG_label, CYP3A4_label, hERG_pchembl, CYP3A4_pchembl
"""

from __future__ import annotations

import csv
import logging
import math
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # src/data/ → admet123/
DB_PATH = (
    PROJECT_ROOT
    / "data/raw/chembl_37_sqlite/chembl_37/chembl_37_sqlite/chembl_37.db"
)
OUTPUT_PATH = PROJECT_ROOT / "data/processed/chembl_admet_training.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Target TIDs from target_dictionary
TARGET_TIDS = {
    "hERG": 165,
    "CYP3A4": 17045,
}

# pChEMBL ≥ THRESHOLD → label=1 (toxic risk / strong inhibition)
PCHEMBL_THRESHOLD = 5.0

# Activity filters (Section 4.2, Master-Konzept)
RELEVANT_TYPES = ("IC50", "Ki", "EC50", "Kd")
MIN_CONFIDENCE = 5

# Unit → multiplier to convert to molar (M)
UNIT_TO_MOLAR: dict[str, float] = {
    "M":  1.0,
    "mM": 1e-3,
    "uM": 1e-6,
    "μM": 1e-6,
    "nM": 1e-9,
    "pM": 1e-12,
    "fM": 1e-15,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chembl_extractor")

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

# Single JOIN query — filters 99%+ of rows at the DB level.
# We fetch only rows where target_tid ∈ (165, 17045), leaving a manageable
# working set for Python-side deduplication.
EXTRACT_SQL = """
SELECT
    cs.canonical_smiles,
    a.standard_type,
    a.standard_value,
    a.standard_units,
    a.pchembl_value,
    a.standard_relation,
    t.tid
FROM activities a
JOIN assays        s  ON a.assay_id      = s.assay_id
JOIN compound_structures cs ON a.molregno = cs.molregno
JOIN target_dictionary   t  ON s.tid      = t.tid
WHERE
    t.tid IN (165, 17045)                   -- hERG + CYP3A4 only
    AND a.standard_relation = '='           -- exact measurements
    AND s.confidence_score >= ?             -- confidence in assays table
    AND a.potential_duplicate = 0
    AND a.standard_type IN ({placeholders})
    AND a.standard_value IS NOT NULL
    AND a.standard_units IS NOT NULL
    AND cs.canonical_smiles IS NOT NULL
    AND cs.canonical_smiles != ''
ORDER BY cs.canonical_smiles, t.tid
""".format(placeholders=",".join("?" * len(RELEVANT_TYPES)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_pchembl(
    standard_value: float,
    standard_units: str,
) -> Optional[float]:
    """Convert (value, unit) → molar concentration → -log10(molar).
    Returns None for invalid / zero / negative inputs."""
    if standard_value is None:
        return None
    try:
        val = float(standard_value)
    except (ValueError, TypeError):
        return None
    if val <= 0:
        return None
    unit_factor = UNIT_TO_MOLAR.get(standard_units)
    if unit_factor is None:
        return None
    molar = val * unit_factor
    if molar <= 0:
        return None
    return -math.log10(molar)


def median(values: list[float]) -> float:
    """Compute the median of a non-empty list of floats."""
    n = len(values)
    if n == 0:
        return float("nan")
    s = sorted(values)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def main() -> None:
    if not DB_PATH.exists():
        log.error("Database not found: %s", DB_PATH)
        sys.exit(1)

    log.info("Connecting to: %s (%.1f GB)", DB_PATH.name, DB_PATH.stat().st_size / 1e9)
    t_total = time.perf_counter()

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA mmap_size = 268435456")  # 256 MB memory-map

    params = [MIN_CONFIDENCE] + list(RELEVANT_TYPES)
    cursor = conn.execute(EXTRACT_SQL, params)

    # Per-molecule, per-target accumulator: (smiles, tid) → list of pchembl
    accum: dict[tuple[str, int], list[float]] = defaultdict(list)

    row_count = 0
    computed_count = 0
    db_pchembl_count = 0
    last_report = time.perf_counter()

    log.info("Streaming rows from SQL…")
    for row in cursor:
        row_count += 1

        smiles = row["canonical_smiles"]
        tid = row["tid"]
        db_pchembl = row["pchembl_value"]

        if db_pchembl is not None:
            db_pchembl_count += 1
            pchembl = float(db_pchembl)
        else:
            pchembl = compute_pchembl(row["standard_value"], row["standard_units"])
            if pchembl is not None:
                computed_count += 1

        if pchembl is not None:
            accum[(smiles, tid)].append(pchembl)

        # Progress every 15 seconds
        now = time.perf_counter()
        if now - last_report > 15:
            log.info(
                "  … %d rows streamed, %d pchembl-able (%d DB + %d computed)",
                row_count, len(accum), db_pchembl_count, computed_count,
            )
            last_report = now

    cursor.close()
    conn.close()

    elapsed_sql = time.perf_counter() - t_total
    log.info(
        "SQL done: %d rows in %.1fs (%.0f rows/s).  %d pchembl-able values.",
        row_count, elapsed_sql, row_count / elapsed_sql if elapsed_sql else 0,
        len(accum),
    )

    # -----------------------------------------------------------------------
    # Deduplication: median pchembl per (SMILES, target)
    # -----------------------------------------------------------------------
    t_dedup = time.perf_counter()

    # Build: smiles → {target_name: (median_pchembl, label)}
    tids_reverse = {v: k for k, v in TARGET_TIDS.items()}
    molecules: dict[str, dict[str, float]] = defaultdict(dict)

    for (smiles, tid), values in accum.items():
        target_name = tids_reverse[tid]
        med = median(values)
        molecules[smiles][target_name] = med

    log.info("Dedup: %d unique molecules in %.1fs", len(molecules), time.perf_counter() - t_dedup)

    # -----------------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------------
    t_write = time.perf_counter()

    target_names = list(TARGET_TIDS.keys())  # ["hERG", "CYP3A4"]
    header = ["canonical_smiles"] + [f"{t}_label" for t in target_names] + [f"{t}_pchembl" for t in target_names]
    written = 0

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for smiles, targets in molecules.items():
            row_out = [smiles]
            for tn in target_names:
                pchembl = targets.get(tn)
                if pchembl is not None:
                    label = 1 if pchembl >= PCHEMBL_THRESHOLD else 0
                    row_out.append(label)
                else:
                    row_out.append("")  # missing
            for tn in target_names:
                pchembl = targets.get(tn)
                row_out.append(f"{pchembl:.4f}" if pchembl is not None else "")
            writer.writerow(row_out)
            written += 1

    log.info("Wrote %d rows to %s in %.1fs", written, OUTPUT_PATH.name, time.perf_counter() - t_write)

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------
    hERG_active = sum(
        1 for t in molecules.values() if t.get("hERG", 0) >= PCHEMBL_THRESHOLD
    )
    CYP3A4_active = sum(
        1 for t in molecules.values() if t.get("CYP3A4", 0) >= PCHEMBL_THRESHOLD
    )
    both_active = sum(
        1 for t in molecules.values()
        if t.get("hERG", 0) >= PCHEMBL_THRESHOLD and t.get("CYP3A4", 0) >= PCHEMBL_THRESHOLD
    )
    only_hERG = sum(
        1 for t in molecules.values() if "hERG" in t and "CYP3A4" not in t
    )
    only_CYP3A4 = sum(
        1 for t in molecules.values() if "CYP3A4" in t and "hERG" not in t
    )
    both_targets = sum(
        1 for t in molecules.values() if "hERG" in t and "CYP3A4" in t
    )

    total_time = time.perf_counter() - t_total
    output_mb = OUTPUT_PATH.stat().st_size / 1e6 if OUTPUT_PATH.exists() else 0

    log.info("=" * 60)
    log.info("EXTRACTION COMPLETE — %s", OUTPUT_PATH)
    log.info("=" * 60)
    log.info("  Total time:              %6.1f s", total_time)
    log.info("  SQL rows returned:       %6d", row_count)
    log.info("  Unique molecules:        %6d", len(molecules))
    log.info("  hERG active (pCh>=%.1f):  %6d", PCHEMBL_THRESHOLD, hERG_active)
    log.info("  CYP3A4 active (pCh>=%.1f):%6d", PCHEMBL_THRESHOLD, CYP3A4_active)
    log.info("  Both targets active:     %6d", both_active)
    log.info("  hERG only:               %6d", only_hERG)
    log.info("  CYP3A4 only:             %6d", only_CYP3A4)
    log.info("  Both targets present:    %6d", both_targets)
    log.info("  Output size:             %5.1f MB", output_mb)
    log.info("  RAM: streamed via cursor — no full load into memory")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

"""
ADMET-Engine — ChEMBL ETL Pipeline
Phase 1, Week 3–6: Data Ingestion & Cleaning

Extracts, transforms, and loads ADMET-relevant bioactivity data from
a ChEMBL-formatted SQLite database. Applies the RDKit standardization
pipeline, computes pChEMBL values, deduplicates by InChI Key, and
produces a cleaned, ML-ready dataset.

Schema JOIN path:
  activities ──molregno──▶ compound_structures
  activities ──assay_id──▶ assays ──tid──▶ target_dictionary

Key filters:
  - confidence_score >= 4 (high-confidence assays only)
  - standard_type in {IC50, Ki, EC50, Kd}
  - potential_duplicate = 0
  - Standardized SMILES must survive RDKit pipeline
  - InChI Key deduplication → median aggregation

Author: Senior Cheminformatics Data Engineer (Persona 1)
Project: ADMET-Engine, Phase 1 — Data Infrastructure
"""

import sqlite3
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem

# Make src/ importable when running this file directly or via import
_this_dir = Path(__file__).resolve().parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

from data_pipeline import standardize_molecule  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("chembl_etl")
logger.setLevel(logging.DEBUG)

_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
if not logger.handlers:
    logger.addHandler(_ch)


# ---------------------------------------------------------------------------
# Constants — ChEMBL schema & ADMET filters
# ---------------------------------------------------------------------------

# ChEMBL uses 5-letter assay_type codes:
#   B = Binding, F = Functional, A = ADME, T = Toxicity, P = Physicochemical
# We keep everything for ADMET but the join is unconditional; actual filtering
# is on confidence_score and standard_type.
CHASSIS_ASSAY_TYPES = frozenset({"B", "F", "A", "T", "P", "U"})

# ADMET-relevant activity standard_types (dose-response measurements)
RELEVANT_STANDARD_TYPES: frozenset = frozenset({"IC50", "Ki", "EC50", "Kd"})

# Minimum confidence score (ChEMBL scale: 0–9)
MIN_CONFIDENCE_SCORE: int = 4

# Unit → molar conversion factor
UNIT_TO_MOLAR: dict = {
    "M": 1.0,
    "mM": 1e-3,
    "uM": 1e-6,
    "nM": 1e-9,
    "pM": 1e-12,
    "fM": 1e-15,
}

# SQL query template: JOIN all 4 tables with pre-filters pushed down to
# reduce the result set before Python-side processing.
EXTRACT_SQL: str = """
SELECT
    cs.molregno,
    cs.canonical_smiles,
    cs.standard_inchi_key,
    a.activity_id,
    a.standard_relation,
    a.standard_value,
    a.standard_units,
    a.standard_type,
    a.pchembl_value,
    a.potential_duplicate,
    s.assay_id,
    s.description    AS assay_description,
    s.confidence_score,
    s.chembl_id      AS assay_chembl_id,
    t.tid,
    t.pref_name      AS target_name,
    t.target_type,
    t.organism       AS target_organism,
    t.chembl_id      AS target_chembl_id
FROM activities a
JOIN compound_structures cs ON a.molregno = cs.molregno
JOIN assays              s  ON a.assay_id  = s.assay_id
LEFT JOIN target_dictionary t ON s.tid = t.tid
WHERE s.confidence_score >= ?
  AND a.standard_type IN (?, ?, ?, ?)
  AND a.standard_value IS NOT NULL
  AND a.standard_units IS NOT NULL
  AND COALESCE(a.potential_duplicate, 0) = 0
"""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_pchembl(
    standard_value: Optional[float],
    standard_units: Optional[str],
) -> Optional[float]:
    """
    Compute pChEMBL = -log₁₀(concentration in Molar).

    pChEMBL normalises potency values across unit regimes so that a higher
    number always means "more potent".  Units are converted to molar before
    the log transformation.

    Args:
        standard_value: Numeric activity value (e.g. 50.0).
        standard_units:  Unit string (e.g. 'nM', 'uM').

    Returns:
        pChEMBL value (float) or None if the input is invalid / non-positive.
    """
    if standard_value is None or standard_units is None:
        return None
    try:
        factor = UNIT_TO_MOLAR.get(standard_units)
        if factor is None:
            logger.debug("Unknown unit '%s' — skipping pChEMBL calc", standard_units)
            return None
        molar = float(standard_value) * factor
        if molar <= 0:
            return None
        return round(-np.log10(molar), 4)
    except (ValueError, TypeError, OverflowError):
        return None


def extract_raw(db_path: str | Path) -> pd.DataFrame:
    """
    Run the parameterised SQL query against a ChEMBL SQLite database.

    Args:
        db_path: Path to the ChEMBL SQLite file.

    Returns:
        DataFrame with all columns from the JOIN query.

    Raises:
        FileNotFoundError: If the database file does not exist.
        sqlite3.Error: On database errors.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"ChEMBL database not found: {db_path}")

    params = (MIN_CONFIDENCE_SCORE, *RELEVANT_STANDARD_TYPES)
    logger.info("Connecting to %s …", db_path.name)

    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(EXTRACT_SQL, conn, params=params)
    finally:
        conn.close()

    logger.info("Raw extraction: %d rows after SQL pre-filter", len(df))
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full cleaning & enrichment pipeline.

    1. Fill missing pChEMBL values via compute_pchembl().
    2. Standardize every SMILES through the RDKit pipeline.
    3. Compute InChI Key as deduplication identifier.
    4. Deduplicate by (inchi_key, standard_type) → median pChEMBL.

    Args:
        df: Raw DataFrame from extract_raw().

    Returns:
        Cleaned, deduplicated DataFrame with canonical SMILES and InChI Keys.
    """
    initial_count = len(df)
    logger.info("Transforming %d raw records …", initial_count)

    # --- Step 1: Fill pChEMBL where missing --------------------------------
    missing_pchembl = df["pchembl_value"].isna().sum()
    if missing_pchembl:
        logger.info("Computing pChEMBL for %d records with missing values", missing_pchembl)
        mask = df["pchembl_value"].isna()
        df.loc[mask, "pchembl_value"] = df.loc[mask].apply(
            lambda r: compute_pchembl(r["standard_value"], r["standard_units"]),
            axis=1,
        )

    # Drop rows where pChEMBL still couldn't be computed
    before_drop = len(df)
    df = df.dropna(subset=["pchembl_value"]).copy()
    logger.info("Dropped %d rows with uncomputable pChEMBL", before_drop - len(df))

    # --- Step 2: RDKit standardization -------------------------------------
    standard_smiles: list[str] = []
    inchi_keys: list[str] = []
    keep_mask: list[bool] = []

    for smi in df["canonical_smiles"]:
        try:
            mol = standardize_molecule(smi)
            std_smi = Chem.MolToSmiles(mol, canonical=True)
            ikey = Chem.MolToInchiKey(mol)
            standard_smiles.append(std_smi)
            inchi_keys.append(ikey)
            keep_mask.append(True)
        except (ValueError, RuntimeError, Exception) as exc:
            logger.debug("RDKit failed for SMILES '%s': %s", smi, exc)
            standard_smiles.append("")
            inchi_keys.append("")
            keep_mask.append(False)

    df["standardized_smiles"] = standard_smiles
    df["inchi_key"] = inchi_keys
    dropped_rdkit = len(keep_mask) - sum(keep_mask)
    df = df[keep_mask].copy()
    logger.info(
        "RDKit pipeline: %d molecules failed, %d survived",
        dropped_rdkit,
        len(df),
    )

    if df.empty:
        logger.warning("No molecules survived RDKit standardization!")
        return df

    # --- Step 3: Deduplication ---------------------------------------------
    # Group by (inchi_key, standard_type) and aggregate
    logger.info("Deduplicating by (InChI Key, standard_type) with median aggregation …")

    # Group by (InChI Key, assay type, target) — "same endpoint" means
    # the same compound measured against the same target with the same assay type.
    # Prefer target_chembl_id if available, fall back to target_name.
    target_col = "target_chembl_id" if "target_chembl_id" in df.columns else "target_name"
    group_cols = ["inchi_key", "standard_type", target_col]

    # Build aggregation dict for all relevant columns
    agg_dict: dict = {
        "pchembl_value": "median",
        "standardized_smiles": "first",
        "canonical_smiles": "first",
        "molregno": "first",
        "standard_relation": lambda x: list(x),
        "standard_value": lambda x: list(x),
        "standard_units": lambda x: list(x),
        "assay_description": "first",
        "target_name": "first",
        "target_type": "first",
        "target_organism": "first",
        "confidence_score": "max",
    }

    # Only include columns that actually exist
    existing_cols = [c for c in agg_dict if c in df.columns and c not in group_cols]
    actual_agg = {k: agg_dict[k] for k in existing_cols}

    deduped = df.groupby(group_cols, as_index=False).agg(actual_agg)

    # Count the number of source rows per group for transparency
    row_counts = df.groupby(group_cols).size().reset_index(name="n_measurements")
    deduped = deduped.merge(row_counts, on=group_cols, how="left")

    # Sort by pChEMBL descending (most potent first) then by InChI Key
    deduped = deduped.sort_values(
        ["inchi_key", "pchembl_value"],
        ascending=[True, False],
    ).reset_index(drop=True)

    logger.info(
        "Deduplication: %d → %d unique (inchi_key, standard_type) pairs",
        len(df),
        len(deduped),
    )

    return deduped


def load(df: pd.DataFrame, output_path: str | Path) -> Path:
    """
    Persist the cleaned DataFrame as CSV.

    Args:
        df: Cleaned, deduplicated DataFrame.
        output_path: Destination path (absolute or relative).

    Returns:
        Resolved Path to the written file.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)
    return output_path


def run_etl(
    db_path: str | Path,
    output_path: str | Path = "data/processed/chembl_cleaned_sample.csv",
) -> pd.DataFrame:
    """
    End-to-end ETL pipeline: extract → transform → load.

    Args:
        db_path: Path to the ChEMBL SQLite database.
        output_path: Destination path for the cleaned CSV file.

    Returns:
        The cleaned, deduplicated DataFrame.
    """
    logger.info("=" * 60)
    logger.info("ADMET-Engine ChEMBL ETL Pipeline — STARTING")
    logger.info("=" * 60)

    df_raw = extract_raw(db_path)
    if df_raw.empty:
        logger.warning("No data extracted; check DB contents and filter criteria.")
        return pd.DataFrame()

    df_clean = transform(df_raw)
    if df_clean.empty:
        logger.warning("No data survived transformation.")
        return df_clean

    load(df_clean, output_path)

    # Summary statistics
    n_mols = df_clean["inchi_key"].nunique()
    n_endpoints = df_clean["standard_type"].nunique()
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE — %d unique molecules across %d assay types", n_mols, n_endpoints)
    logger.info("Output: %s", Path(output_path).resolve())
    logger.info("=" * 60)

    return df_clean


# ---------------------------------------------------------------------------
# Convenience: run as script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ADMET-Engine ChEMBL ETL Pipeline")
    parser.add_argument("db_path", help="Path to the ChEMBL SQLite database")
    parser.add_argument(
        "-o",
        "--output",
        default="data/processed/chembl_cleaned_sample.csv",
        help="Output CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    run_etl(args.db_path, args.output)

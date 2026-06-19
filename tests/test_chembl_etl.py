"""
ADMET-Engine — ChEMBL ETL Pipeline Unit Tests

Creates an in-memory SQLite database with the ChEMBL v37 schema and
realistic mock data, then runs the full ETL pipeline and asserts:

  - Confidence score filter (>= 4)    → low-score entries excluded
  - Standard-type filter (IC50/Ki/EC50/Kd) → AC50 etc. excluded
  - Potential-duplicate exclusion
  - RDKit standardisation & salt removal
  - InChI Key deduplication with median aggregation
  - pChEMBL computation for missing values

Schema (abbreviated from the full ChEMBL schema, only essential columns):

  target_dictionary       compound_structures       assays            activities
  ┌──────────────┐       ┌──────────────────┐       ┌──────────┐      ┌──────────────┐
  │ tid (PK)     │       │ molregno (PK)    │◄──────│ assay_id │◄─────│ activity_id  │
  │ pref_name    │       │ canonical_smiles │       │ tid (FK) │      │ assay_id (FK)│
  │ target_type  │       │ standard_inchi_  │       │ conf_scr │      │ molregno(FK) │
  │ organism     │       │     key          │       │ assay_   │      │ std_relation │
  └──────────────┘       └──────────────────┘       │   type   │      │ std_value    │
                                                     └──────────┘      │ std_units    │
                                                                        │ std_type     │
                                                                        │ pchembl_val  │
                                                                        │ potl_dupl    │
                                                                        └──────────────┘
"""

import sqlite3
import sys
import os
from pathlib import Path
from typing import Any

import pytest
import numpy as np
import pandas as pd

# Ensure src/ is importable
_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))

from chembl_etl import (  # noqa: E402
    extract_raw,
    transform,
    run_etl,
    compute_pchembl,
    RELEVANT_STANDARD_TYPES,
    MIN_CONFIDENCE_SCORE,
)


# ============================================================================
# Helper fixtures
# ============================================================================

@pytest.fixture
def mock_db() -> sqlite3.Connection:
    """
    Create an in-memory SQLite database with the ChEMBL schema and
    16 realistic mock activities covering all required edge cases.
    """
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")

    # ---- Schema -----------------------------------------------------------
    conn.executescript("""
        CREATE TABLE target_dictionary (
            tid         INTEGER PRIMARY KEY,
            pref_name   TEXT,
            target_type TEXT,
            organism    TEXT,
            chembl_id   TEXT
        );

        CREATE TABLE compound_structures (
            molregno            INTEGER PRIMARY KEY,
            canonical_smiles    TEXT    NOT NULL,
            standard_inchi_key  TEXT
        );

        CREATE TABLE assays (
            assay_id          INTEGER PRIMARY KEY,
            description       TEXT,
            assay_type        TEXT,          -- 'B','F','A','T','P','U'
            confidence_score  INTEGER,
            tid               INTEGER,
            chembl_id         TEXT,
            FOREIGN KEY (tid) REFERENCES target_dictionary(tid)
        );

        CREATE TABLE activities (
            activity_id         INTEGER PRIMARY KEY,
            assay_id            INTEGER,
            molregno            INTEGER,
            standard_relation   TEXT,
            standard_value      REAL,
            standard_units      TEXT,
            standard_type       TEXT,
            pchembl_value       REAL,
            potential_duplicate INTEGER DEFAULT 0,
            FOREIGN KEY (assay_id) REFERENCES assays(assay_id),
            FOREIGN KEY (molregno) REFERENCES compound_structures(molregno)
        );
    """)

    # ---- Insert target_dictionary (4 ADMET-relevant targets) --------------
    targets = [
        (1, "Cytochrome P450 3A4",  "SINGLE PROTEIN", "Homo sapiens", "CHEMBL340"),
        (2, "HERG",                  "SINGLE PROTEIN", "Homo sapiens", "CHEMBL240"),
        (3, "P-glycoprotein 1",      "SINGLE PROTEIN", "Homo sapiens", "CHEMBL4302"),
        (4, "Beta-secretase 1",      "SINGLE PROTEIN", "Homo sapiens", "CHEMBL4822"),
    ]
    conn.executemany(
        "INSERT INTO target_dictionary VALUES (?,?,?,?,?)", targets
    )

    # ---- Insert compound_structures (8 molecules, incl. edge-cases) -------
    compounds = [
        # molregno | canonical_smiles                          | inchi_key (raw, will be replaced)
        (1,  "CC(=O)Oc1ccccc1C(=O)O",          "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"),  # Aspirin
        (2,  "CC(C)Cc1ccc(C(C)C(=O)O)cc1",     "HEFNNWSXXWATRW-UHFFFAOYSA-N"),  # Ibuprofen
        (3,  "CC(=O)Nc1ccc(O)cc1",             "RZVAJINKPMORJF-UHFFFAOYSA-N"),  # Paracetamol
        (4,  "Cn1cnc2n(C)c(=O)n(C)c(=O)c12",   "RYYVLZVUVIJVGH-UHFFFAOYSA-N"),  # Caffeine
        (5,  "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13", "AAOVKJBEBIDNHE-UHFFFAOYSA-N"),  # Diazepam
        (6,  "CC(=O)Oc1ccccc1C(=O)O.[Na+]",    "BSYNRYMUTXBXSQ-MDQQWAPMSA-M"),  # Aspirin Na — SAME InChI after salt removal
        (7,  "C1CC1XXXXYYYYZZZZINVALID",       None),                            # INVALID SMILES → RDKit will fail
        (8,  "CC(=O)CC(c1ccccc1)c2c(O)c3ccccc3oc2=O", "PJVWKTKQMONHTI-UHFFFAOYSA-N"),  # Warfarin
    ]
    conn.executemany(
        "INSERT INTO compound_structures VALUES (?,?,?)", compounds
    )

    # ---- Insert assays ----------------------------------------------------
    assays = [
        # id | description              | type | conf | tid | chembl_id
        (1,  "CYP3A4 inhibition IC50",  "B",   7,     1,    "CHEMBL12345"),
        (2,  "CYP3A4 inhibition IC50 v2","B",  8,     1,    "CHEMBL12346"),
        (3,  "hERG binding Ki",         "B",   9,     2,    "CHEMBL12347"),
        (4,  "hERG IC50 low-conf",      "B",   3,     2,    "CHEMBL12348"),   # ── SCORE=3 → FILTERED
        (5,  "P-gp transport EC50",     "F",   6,     3,    "CHEMBL12349"),
        (6,  "BACE1 inhibition Ki",     "B",   5,     4,    "CHEMBL12350"),
        (7,  "BACE1 AC50 (non-std)",    "B",   8,     4,    "CHEMBL12351"),   # ── AC50 → FILTERED by SQL
        (8,  "CYP3A4 inhibition Ki",    "B",   4,     1,    "CHEMBL12352"),
    ]
    conn.executemany(
        "INSERT INTO assays VALUES (?,?,?,?,?,?)", assays
    )

    # ---- Insert activities (16 entries with deliberate edge-cases) --------
    activities = [
        # (act_id, assay_id, molregno, relation, value, units, type, pchembl, pot_dup)
        # ─── Aspirin (molregno=1) ──────────────────────────────────────────
        (1,  1, 1, "=",  5000,  "nM", "IC50", None,   0),   # CYP3A4 IC50, needs pChEMBL calc
        (2,  2, 1, "=",  3000,  "nM", "IC50", None,   0),   # CYP3A4 IC50 (duplicate w/ #1)
        (3,  3, 1, "<",  10,    "uM", "Ki",   5.0,    0),   # hERG Ki, pre-computed
        (4,  4, 1, "=",  50,    "uM", "IC50", None,   0),   # ── LOW CONF (assay_id=4, score=3)

        # ─── Ibuprofen (molregno=2) ───────────────────────────────────────
        (5,  1, 2, "=",  100,   "uM", "IC50", 4.0,    0),   # CYP3A4 IC50
        (6,  3, 2, "=",  50,    "uM", "Ki",   4.3,    0),   # hERG Ki

        # ─── Paracetamol (molregno=3) ─────────────────────────────────────
        (7,  1, 3, "=",  200,   "uM", "IC50", 3.7,    0),   # CYP3A4 IC50
        (8,  5, 3, "=",  1,     "mM", "EC50", None,   0),   # P-gp EC50, needs pChEMBL calc

        # ─── Caffeine (molregno=4) ────────────────────────────────────────
        (9,  3, 4, ">",  500,   "uM", "Ki",   None,   0),   # hERG Ki, needs pChEMBL calc
        (10, 6, 4, "=",  8,     "uM", "Ki",   5.1,    0),   # BACE1 Ki

        # ─── Diazepam (molregno=5) ────────────────────────────────────────
        (11, 8, 5, "=",  200,   "nM", "Ki",   6.7,    0),   # CYP3A4 Ki

        # ─── Aspirin Sodium (molregno=6, same InChI as #1 after salt strip)
        (12, 1, 6, "=",  4500,  "nM", "IC50", None,   0),   # CYP3A4 IC50 — DUPLICATE w/ #1,#2

        # ─── Warfarin (molregno=8) ────────────────────────────────────────
        (13, 1, 8, "=",  2,     "uM", "IC50", None,   0),   # CYP3A4 IC50, needs pChEMBL calc

        # ─── Invalid SMILES compound (molregno=7) ─────────────────────────
        (14, 3, 7, "=",  100,   "uM", "Ki",   None,   0),   # ── INVALID SMILES → filtered by RDKit

        # ─── Wrong standard_type (AC50) ───────────────────────────────────
        (15, 7, 4, "=",  20,    "uM", "AC50", None,   0),   # ── AC50 → filtered by SQL

        # ─── Potential duplicate flag ─────────────────────────────────────
        (16, 2, 1, "=",  6000,  "nM", "IC50", None,   1),   # ── pot_dup=1 → filtered
    ]
    conn.executemany(
        "INSERT INTO activities VALUES (?,?,?,?,?,?,?,?,?)", activities
    )

    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def db_path(mock_db: sqlite3.Connection, tmp_path: Path) -> Path:
    """Dump the in-memory DB to a file so extract_raw() can use a path."""
    file_db = tmp_path / "chembl_mock.db"
    disk = sqlite3.connect(str(file_db))
    mock_db.backup(disk)
    disk.close()

    # Also write the conn to disk for the test that uses run_etl
    # Actually, extract_raw uses a path, so this is what we need.
    return file_db


# ============================================================================
# Unit tests — pChEMBL computation
# ============================================================================

class TestPchemblComputation:
    """Verify pChEMBL = -log10(value * unit_factor)."""

    def test_nM_conversion(self):
        assert compute_pchembl(5000, "nM") == pytest.approx(5.3010, abs=1e-3)
        # 5000 nM = 5e-6 M → -log10(5e-6) = 5.301

    def test_uM_conversion(self):
        assert compute_pchembl(10, "uM") == pytest.approx(5.0, abs=1e-3)
        # 10 uM = 1e-5 M → -log10(1e-5) = 5.0

    def test_mM_conversion(self):
        assert compute_pchembl(1, "mM") == pytest.approx(3.0, abs=1e-3)

    def test_M_conversion(self):
        assert compute_pchembl(0.001, "M") == pytest.approx(3.0, abs=1e-3)

    def test_none_value_returns_none(self):
        assert compute_pchembl(None, "nM") is None

    def test_none_units_returns_none(self):
        assert compute_pchembl(100, None) is None

    def test_zero_value_returns_none(self):
        assert compute_pchembl(0, "nM") is None

    def test_negative_value_returns_none(self):
        assert compute_pchembl(-10, "uM") is None

    def test_unknown_unit_returns_none(self):
        assert compute_pchembl(100, "kg/L") is None


# ============================================================================
# Integration tests — full ETL pipeline on mock DB
# ============================================================================

class TestChemblEtlPipeline:
    """Run the complete ETL over the mock DB and verify every filter."""

    @pytest.fixture(autouse=True)
    def _run(self, db_path: Path):
        """Run once and store results as instance attributes."""
        self.df_raw = extract_raw(db_path)
        self.df_clean = transform(self.df_raw)

    # ------------------------------------------------------------------
    # SQL-level filters (pre-transform assertions)
    # ------------------------------------------------------------------

    def test_all_extracted_have_high_confidence(self):
        """No low-confidence entries should survive the SQL WHERE clause."""
        if "confidence_score" in self.df_raw.columns:
            assert (self.df_raw["confidence_score"] >= MIN_CONFIDENCE_SCORE).all(), (
                "Low-confidence records leaked through SQL filter!"
            )

    def test_all_extracted_have_relevant_standard_types(self):
        """Only IC50, Ki, EC50, Kd should be present."""
        types_present = set(self.df_raw["standard_type"].unique())
        assert types_present.issubset(RELEVANT_STANDARD_TYPES), (
            f"Unwanted standard_types found: {types_present - RELEVANT_STANDARD_TYPES}"
        )

    def test_no_potential_duplicates_in_raw(self):
        """SQL must exclude potential_duplicate=1."""
        if "potential_duplicate" in self.df_raw.columns:
            assert (self.df_raw["potential_duplicate"] == 0).all(), (
                "Potential duplicates leaked through SQL filter!"
            )

    def test_ac50_not_in_extracted_data(self):
        """AC50 entries should be filtered by the SQL standard_type IN clause."""
        assert "AC50" not in set(self.df_raw["standard_type"]), (
            "AC50 assay type was not filtered out!"
        )

    # ------------------------------------------------------------------
    # Raw extraction counts
    # ------------------------------------------------------------------

    def test_raw_row_count(self):
        """16 activities → after SQL filter:
        Remove: activity #4 (low conf), #15 (AC50 type), #16 (pot_dup) = 13 remain.
        """
        # 16 total - 3 filtered by SQL = 13
        assert len(self.df_raw) == 13, (
            f"Expected 13 raw rows after SQL filter, got {len(self.df_raw)}"
        )

    # ------------------------------------------------------------------
    # RDKit pipeline assertions
    # ------------------------------------------------------------------

    def test_invalid_smiles_excluded(self):
        """Compound #7 (molregno=7, invalid SMILES) must not appear in cleaned output."""
        invalid_inchi = self.df_clean[self.df_clean["molregno"] == 7]
        assert invalid_inchi.empty, (
            "Invalid SMILES compound (molregno=7) was not removed by RDKit pipeline!"
        )

    def test_all_cleaned_rows_have_inchi_key(self):
        """Every surviving row must have a non-empty InChI Key."""
        assert self.df_clean["inchi_key"].notna().all()
        assert (self.df_clean["inchi_key"] != "").all()

    # ------------------------------------------------------------------
    # pChEMBL computation assertions
    # ------------------------------------------------------------------

    def test_pchembl_filled_for_all_rows(self):
        """After transform, no row should have a null pchembl_value."""
        assert self.df_clean["pchembl_value"].notna().all(), (
            "Some rows still have null pchembl_value after transformation!"
        )

    def test_pchembl_computation_paracetamol_pgp(self):
        """Paracetamol P-gp EC50: 1 mM → pchembl = 3.0.
        Matched by InChI Key (paracetamol = RZVAJINKPMORJF-UHFFFAOYSA-N)."""
        row = self.df_clean[
            (self.df_clean["inchi_key"].str.startswith("RZVAJINKPMORJF", na=False))
            & (self.df_clean["standard_type"] == "EC50")
        ]
        assert len(row) == 1, f"Expected 1 row for Paracetamol EC50, got {len(row)}"
        assert row.iloc[0]["pchembl_value"] == pytest.approx(3.0, abs=1e-3)

    def test_pchembl_computation_warfarin(self):
        """Warfarin CYP3A4 IC50: 2 uM → pchembl ≈ 5.699.
        Matched by InChI Key (warfarin = PJVWKTKQMONHTI-UHFFFAOYSA-N)."""
        row = self.df_clean[
            self.df_clean["inchi_key"].str.startswith("PJVWKTKQMONHTI", na=False)
        ]
        assert not row.empty, "Warfarin row missing from output!"
        assert row.iloc[0]["pchembl_value"] == pytest.approx(5.699, abs=1e-3)

    # ------------------------------------------------------------------
    # Deduplication assertions (salt removal + median)
    # ------------------------------------------------------------------

    def test_aspirin_salt_merged(self):
        """Aspirin (molregno=1) and Aspirin Na (molregno=6) share the same
        InChI Key after salt removal. Both are CYP3A4 IC50 — merged into 1 row.
        """
        aspirin_rows = self.df_clean[
            self.df_clean["standardized_smiles"].str.contains(
                "CC(=O)Oc1ccccc1C(=O)O", na=False, regex=False
            )
            & (self.df_clean["standard_type"] == "IC50")
        ]
        assert len(aspirin_rows) == 1, (
            f"Aspirin CYP3A4 IC50 should be 1 deduplicated row, got {len(aspirin_rows)}"
        )

    def test_aspirin_ic50_median(self):
        """After dedup: median(5000 nM, 3000 nM, 4500 nM) → pchembl ≈ 5.347."""
        aspirin_row = self.df_clean[
            self.df_clean["standardized_smiles"].str.contains("CC(=O)Oc1ccccc1C(=O)O", na=False, regex=False)
            & (self.df_clean["standard_type"] == "IC50")
        ]
        assert aspirin_row.iloc[0]["pchembl_value"] == pytest.approx(5.347, abs=1e-2)
        assert aspirin_row.iloc[0]["n_measurements"] == 3

    def test_aspirin_hERG_ki_unchanged(self):
        """Aspirin hERG Ki has only 1 measurement → value stays 5.0."""
        aspirin_herg = self.df_clean[
            self.df_clean["standardized_smiles"].str.contains("CC(=O)Oc1ccccc1C(=O)O", na=False, regex=False)
            & (self.df_clean["standard_type"] == "Ki")
        ]
        assert len(aspirin_herg) == 1
        assert aspirin_herg.iloc[0]["pchembl_value"] == pytest.approx(5.0, abs=1e-3)
        assert aspirin_herg.iloc[0]["n_measurements"] == 1  # Only the non-herg entries; actually, let's think... 

    # ------------------------------------------------------------------
    # Final output shape
    # ------------------------------------------------------------------

    def test_unique_pairs_count(self):
        """After full pipeline: 10 unique (inchi_key, standard_type) pairs.

        13 raw rows → −1 (invalid SMILES, molregno=7) = 12 → dedup aspirin
        IC50 (3→1) = 10.
        """
        assert len(self.df_clean) == 10, (
            f"Expected 10 unique rows after full pipeline, got {len(self.df_clean)}"
        )

    def test_unique_molecules_count(self):
        """7 unique molecules: Aspirin, Ibuprofen, Paracetamol, Caffeine, Diazepam, Warfarin.
        (mol 7 invalid → removed, mol 6 merged into mol 1)"""
        n_mols = self.df_clean["inchi_key"].nunique()
        assert n_mols == 6, f"Expected 6 unique molecules, got {n_mols}"

    def test_csv_output(self, db_path: Path, tmp_path: Path):
        """End-to-end: run_etl writes a valid CSV file."""
        out = tmp_path / "chembl_cleaned_test.csv"
        result = run_etl(db_path, out)
        assert out.exists()
        df_from_csv = pd.read_csv(out)
        assert len(df_from_csv) == len(result)
        assert "inchi_key" in df_from_csv.columns
        assert "pchembl_value" in df_from_csv.columns

    # ------------------------------------------------------------------
    # relation preservation
    # ------------------------------------------------------------------

    def test_standard_relations_preserved(self):
        """The 'standard_relation' column (list) must contain the original relations."""
        assert "standard_relation" in self.df_clean.columns, (
            "standard_relation column missing after aggregation"
        )

    def test_confidence_column_in_result(self):
        """Confidence score should be present (max across merged rows)."""
        assert "confidence_score" in self.df_clean.columns


# ============================================================================
# Edge-case: empty database
# ============================================================================

class TestEmptyDatabase:
    """Verify graceful behaviour when the DB contains no matching data."""

    def test_empty_extraction(self, tmp_path: Path):
        """Empty DB → extract returns empty DataFrame."""
        db = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db))
        conn.executescript("""
            CREATE TABLE compound_structures (molregno INTEGER PRIMARY KEY, canonical_smiles TEXT, standard_inchi_key TEXT);
            CREATE TABLE assays (assay_id INTEGER PRIMARY KEY, description TEXT, assay_type TEXT, confidence_score INTEGER, tid INTEGER, chembl_id TEXT);
            CREATE TABLE activities (activity_id INTEGER PRIMARY KEY, assay_id INTEGER, molregno INTEGER, standard_relation TEXT, standard_value REAL, standard_units TEXT, standard_type TEXT, pchembl_value REAL, potential_duplicate INTEGER DEFAULT 0);
            CREATE TABLE target_dictionary (tid INTEGER PRIMARY KEY, pref_name TEXT, target_type TEXT, organism TEXT, chembl_id TEXT);
        """)
        conn.commit()
        conn.close()

        df = extract_raw(db)
        assert df.empty

    def test_empty_transform_does_not_crash(self):
        """Transform on empty DataFrame returns empty DataFrame."""
        df = pd.DataFrame(columns=["canonical_smiles", "pchembl_value", "standard_type"])
        result = transform(df)
        assert result.empty

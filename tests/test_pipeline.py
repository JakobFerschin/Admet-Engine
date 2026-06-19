"""
Unit tests for the ADMET-Engine RDKit Standardization Pipeline.

Tests the standardize_molecule function from src/data_pipeline.py,
verifying salt removal, canonicalization, and error-handling behavior.
"""

import sys
import os

# Ensure src/ is on the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pytest
from rdkit import Chem
from data_pipeline import standardize_molecule, smiles_to_canonical


class TestStandardizeMolecule:
    """Core tests for the molecular standardization pipeline (Section 4.9)."""

    def test_salt_removal_aspirin_sodium(self):
        """
        Verifies that standardize_molecule correctly strips the [Na+] counterion
        from aspirin sodium, yielding pure acetylsalicylic acid.
        """
        smiles_input = "CC(=O)Oc1ccccc1C(=O)O.[Na+]"
        mol = standardize_molecule(smiles_input)

        # Must return a valid Mol object
        assert mol is not None, "standardize_molecule returned None for valid input"

        # Convert back to SMILES and check salt is gone
        result_smiles = Chem.MolToSmiles(mol, canonical=True)
        assert "Na" not in result_smiles, f"Salt [Na+] was not removed: {result_smiles}"
        assert "." not in result_smiles, f"Dot separator (multi-component) still present: {result_smiles}"

    def test_salt_removal_multiple_counterions(self):
        """
        Test removal of multiple counterions (e.g., hydrochloride salt).
        """
        smiles_input = "CN(C)C.[Cl-]"
        mol = standardize_molecule(smiles_input)
        result = Chem.MolToSmiles(mol, canonical=True)
        assert "Cl" not in result or "." not in result, f"Counterion not stripped: {result}"

    def test_invalid_smiles_raises(self):
        """
        Invalid SMILES must raise a ValueError.
        """
        with pytest.raises(ValueError, match="Invalid SMILES string"):
            standardize_molecule("CXX")

    def test_empty_smiles_handled(self):
        """
        Empty string produces an empty Mol from RDKit. The pipeline should
        gracefully handle this edge case without crashing.
        """
        mol = standardize_molecule("")
        assert mol is not None, "Empty SMILES should return an empty Mol, not None/NULL"

    def test_smiles_to_canonical_wrapper(self):
        """
        The convenience wrapper returns canonical SMILES after standardization.
        """
        result = smiles_to_canonical("CC(=O)Oc1ccccc1C(=O)O.[Na+]")
        assert result, "Wrapper returned empty string for valid input"
        assert "Na" not in result
        assert "." not in result

    def test_smiles_to_canonical_invalid_returns_empty(self):
        """
        Invalid SMILES via wrapper returns empty string without crashing.
        """
        result = smiles_to_canonical("not_a_smiles!")
        assert result == ""

    def test_standardization_idempotent(self):
        """
        Running standardization twice on the same molecule should produce the same result.
        """
        smiles = "c1ccccc1C(=O)O"
        mol1 = standardize_molecule(smiles)
        smi1 = Chem.MolToSmiles(mol1, canonical=True)
        mol2 = standardize_molecule(smi1)
        smi2 = Chem.MolToSmiles(mol2, canonical=True)
        assert smi1 == smi2, f"Standardization is not idempotent: {smi1} != {smi2}"

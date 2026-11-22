"""
ADMET-Engine — Data Pipeline: RDKit Standardization Core

Implements the canonical molecular standardization pipeline as specified
in Section 4.9 of the ADMET-Engine Project Report.

Key steps:
  1. Salt/counterion removal
  2. Explicit hydrogen removal
  3. Functional group normalization
  4. Reionization to preferred charge state
  5. Tautomer canonicalization
  6. Stereochemistry assignment
  7. Kekulization
  8. Final sanitization

Author: Senior Cheminformatics Data Engineer (Persona 1)
Project: ADMET-Engine, Phase 1 — Data Infrastructure
"""

from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover, AllChem  # noqa: F401 — reserved for future 3D pipeline
from rdkit.Chem.MolStandardize import rdMolStandardize

# Silence noisy RDKit standardization info/warning logs during batch processing
RDLogger.DisableLog("rdApp.info")

# Module-level cached standardizers (avoids re-instantiating C++ objects per molecule)
_SALT_REMOVER = SaltRemover.SaltRemover()
_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_NORMALIZER = rdMolStandardize.Normalizer()
_REIONIZER = rdMolStandardize.Reionizer()
_TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()
_TAUTOMER_ENUMERATOR.SetMaxTautomers(64)
_TAUTOMER_ENUMERATOR.SetMaxTransforms(64)


def standardize_molecule(smiles: str) -> Chem.Mol:
    """
    Canonical standardization pipeline for ADMET ML.

    Applies the full RDKit MolStandardize workflow to ensure every molecule
    is represented in a consistent, canonical form before featurization and
    model training.

    Args:
        smiles: A SMILES string representing the input molecule.

    Returns:
        An RDKit Mol object in its canonical standardized form,
        or an empty Mol for an empty input string.

    Raises:
        ValueError: If the input SMILES cannot be parsed by RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: '{smiles}' — could not parse with RDKit.")

    if mol.GetNumAtoms() == 0:
        return mol

    # 1. Remove salts/counterions (e.g., [Na+], [Cl-], Br., etc.)
    mol = _SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)
    if mol.GetNumAtoms() > 0:
        mol = _LARGEST_FRAGMENT.choose(mol)

    # 2. Remove explicit hydrogens
    mol = Chem.RemoveHs(mol)

    # 3. Normalize functional groups (e.g., nitro, sulfoxide to canonical forms)
    mol = _NORMALIZER.normalize(mol)

    # 4. Reionize to preferred charge state
    mol = _REIONIZER.reionize(mol)

    # 5. Standardize tautomer
    mol = _TAUTOMER_ENUMERATOR.Canonicalize(mol)

    # 6. Assign stereochemistry
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

    # 7. Kekulize and re-sanitize to guarantee canonical aromaticity perception
    Chem.Kekulize(mol, clearAromaticFlags=True)
    Chem.SanitizeMol(mol)

    return mol


def smiles_to_canonical(smiles: str) -> str:
    """
    Convenience wrapper: standardize + return canonical SMILES.
    Useful for deduplication via InChI Key / canonical SMILES matching.

    Args:
        smiles: Input SMILES string.

    Returns:
        Canonical SMILES string after full standardization, or empty string on failure.
    """
    try:
        mol = standardize_molecule(smiles)
        return Chem.MolToSmiles(mol, canonical=True)
    except (ValueError, RuntimeError) as e:
        print(f"Warning: Failed to canonicalize SMILES '{smiles}': {e}")
        return ""

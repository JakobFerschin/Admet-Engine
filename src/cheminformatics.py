"""
ADMET-Engine — Academic Medicinal Chemistry & Cheminformatics Engine

Computes deterministic, literature-grounded cheminformatics profiles from
standardized RDKit Mol objects:
  • Canonical identifiers: Canonical SMILES, Molecular Formula, InChIKey,
    Bemis-Murcko generic/core scaffold
  • Physicochemical descriptors: MW, Wildman-Crippen WLOGP (cLogP), Ertl TPSA,
    HBD, HBA, Rotatable Bonds, Aromatic Rings, Fsp3 (Lovering et al., 2009),
    QED (Bickerton et al., Nature Chemistry 2012), Molar Refractivity, BertzCT
  • Medicinal chemistry compliance rules:
      - Lipinski Rule of 5 (Adv. Drug Deliv. Rev. 1997)
      - Veber Oral Bioavailability Rules (J. Med. Chem. 2002)
      - Pfizer 3/75 Promiscuity/Toxicity Rule (Hughes et al., BMCL 2008)
      - GSK 4/400 ADMET Rule (Gleeson, J. Med. Chem. 2008)
  • Substructure toxicity & interference alerts via RDKit FilterCatalog:
      - PAINS A/B/C (Baell & Holloway, J. Med. Chem. 2010)
      - Brenk unwanted/reactive functional groups (ChemMedChem 2008)
  • BOILED-Egg analytical HIA & BBB permeation model
    (Daina & Zoete, ChemMedChem 2016)
  • Publication-grade 2D chemical structure SVG rendering with optional
    structural-alert atom highlighting
  • Bemis-Murcko scaffold extraction & Morgan ECFP4 fingerprints for
    Applicability Domain (Tanimoto similarity) and hybrid ML featurization
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Crippen,
    Descriptors,
    FilterCatalog,
    GraphDescriptors,
    Lipinski,
    QED,
    rdDepictor,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold

# ---------------------------------------------------------------------------
# Module-level cached RDKit filter catalogs & fingerprint generator
# ---------------------------------------------------------------------------

def _build_catalog(catalog_flag) -> FilterCatalog.FilterCatalog:
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(catalog_flag)
    return FilterCatalog.FilterCatalog(params)


_PAINS_CATALOG = _build_catalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
_BRENK_CATALOG = _build_catalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# ---------------------------------------------------------------------------
# BOILED-Egg Analytical Ellipses (Daina & Zoete, ChemMedChem 2016, Table 1)
# ---------------------------------------------------------------------------

# White ellipse: Human Intestinal Absorption (HIA)
_HIA_ELLIPSE = {
    "cx": 71.051,
    "cy": 2.292,
    "a": 142.081 / 2.0,
    "b": 8.740 / 2.0,
    "angle_deg": -1.031325,
}

# Yolk ellipse: Blood-Brain Barrier (BBB) permeation
_BBB_ELLIPSE = {
    "cx": 38.117,
    "cy": 3.177,
    "a": 82.061 / 2.0,
    "b": 5.557 / 2.0,
    "angle_deg": 0.171887,
}


def _point_in_rotated_ellipse(x: float, y: float, ellipse: dict[str, float]) -> bool:
    """Evaluate exact membership in a rotated 2D ellipse."""
    theta = math.radians(ellipse["angle_deg"])
    dx = x - ellipse["cx"]
    dy = y - ellipse["cy"]
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    xr = dx * cos_t + dy * sin_t
    yr = -dx * sin_t + dy * cos_t
    val = (xr / ellipse["a"]) ** 2 + (yr / ellipse["b"]) ** 2
    return val <= 1.0


def evaluate_boiled_egg(tpsa: float, wlogp: float) -> dict[str, Any]:
    """
    Evaluate the BOILED-Egg model (Daina & Zoete, ChemMedChem 2016)
    for passive gastrointestinal absorption (HIA) and BBB permeation.
    """
    bbb = _point_in_rotated_ellipse(tpsa, wlogp, _BBB_ELLIPSE)
    hia = bbb or _point_in_rotated_ellipse(tpsa, wlogp, _HIA_ELLIPSE)
    if bbb:
        region = "yolk"  # BBB permeant + high HIA
        label = "BBB Permeant (CNS-accessible)"
    elif hia:
        region = "white"  # High HIA, non-BBB
        label = "High GI Absorption (HIA+)"
    else:
        region = "outside"
        label = "Low Passive Absorption"

    return {
        "tpsa": round(tpsa, 2),
        "wlogp": round(wlogp, 3),
        "hia_absorbed": hia,
        "bbb_permeant": bbb,
        "region": region,
        "classification": label,
    }


# ---------------------------------------------------------------------------
# Scaffold & Fingerprint utilities
# ---------------------------------------------------------------------------

def get_murcko_scaffold(mol: Chem.Mol) -> str:
    """Return canonical Bemis-Murcko scaffold SMILES (or 'Acyclic' if ringless)."""
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaf if scaf else "Acyclic"
    except Exception:
        return "Acyclic"


def compute_morgan_fp(mol: Chem.Mol):
    """Compute 2048-bit ECFP4 (Morgan radius=2) ExplicitBitVect."""
    return _MORGAN_GEN.GetFingerprint(mol)


def compute_morgan_numpy(mol: Chem.Mol) -> np.ndarray:
    """Compute 2048-bit ECFP4 as float32 numpy array."""
    return _MORGAN_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)


def compute_rdkit_ml_descriptors(mol: Chem.Mol) -> np.ndarray:
    """
    Compute 20 normalized, physically meaningful RDKit 2D descriptors
    used alongside Morgan ECFP4 and DMPNN graph embeddings.
    """
    mw = Descriptors.MolWt(mol)
    clogp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    aliph_rings = rdMolDescriptors.CalcNumAliphaticRings(mol)
    het_rings = rdMolDescriptors.CalcNumHeterocycles(mol)
    heavy = mol.GetNumHeavyAtoms()
    fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    mr = Crippen.MolMR(mol)
    formal_q = Chem.GetFormalCharge(mol)
    qed_val = QED.qed(mol)
    n_basic_n = sum(
        1 for a in mol.GetAtoms()
        if a.GetAtomicNum() == 7 and not a.GetIsAromatic() and a.GetFormalCharge() >= 0
    )
    n_halogens = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9, 17, 35, 53))
    n_hetero = rdMolDescriptors.CalcNumHeteroatoms(mol)
    labute_asa = rdMolDescriptors.CalcLabuteASA(mol)
    balaban = GraphDescriptors.BalabanJ(mol) if mol.GetNumBonds() > 0 else 0.0
    bertz = GraphDescriptors.BertzCT(mol) if heavy > 1 else 0.0

    feats = np.array(
        [
            mw / 500.0,
            clogp / 5.0,
            tpsa / 140.0,
            hbd / 5.0,
            hba / 10.0,
            rotb / 10.0,
            arom_rings / 5.0,
            aliph_rings / 5.0,
            het_rings / 5.0,
            heavy / 40.0,
            fsp3,
            mr / 130.0,
            float(formal_q),
            qed_val,
            n_basic_n / 4.0,
            n_halogens / 4.0,
            n_hetero / 10.0,
            labute_asa / 200.0,
            balaban / 5.0,
            math.log1p(max(bertz, 0.0)) / 8.0,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Structural Alerts (PAINS & Brenk)
# ---------------------------------------------------------------------------

def screen_structural_alerts(mol: Chem.Mol) -> dict[str, Any]:
    """
    Run RDKit PAINS (A/B/C) and Brenk substructure filters.
    Returns matched alert names and matching atom indices for 2D highlighting.
    """
    pains_alerts: list[str] = []
    brenk_alerts: list[str] = []
    highlight_atoms: set[int] = set()

    for entry in _PAINS_CATALOG.GetMatches(mol):
        desc = entry.GetDescription()
        if desc not in pains_alerts:
            pains_alerts.append(desc)
        for match in entry.GetFilterMatches(mol):
            for _, mol_atom_idx in match.atomPairs:
                highlight_atoms.add(int(mol_atom_idx))

    for entry in _BRENK_CATALOG.GetMatches(mol):
        desc = entry.GetDescription()
        if desc not in brenk_alerts:
            brenk_alerts.append(desc)
        for match in entry.GetFilterMatches(mol):
            for _, mol_atom_idx in match.atomPairs:
                highlight_atoms.add(int(mol_atom_idx))

    return {
        "pains_count": len(pains_alerts),
        "pains_alerts": pains_alerts,
        "brenk_count": len(brenk_alerts),
        "brenk_alerts": brenk_alerts,
        "total_alerts": len(pains_alerts) + len(brenk_alerts),
        "highlight_atoms": sorted(highlight_atoms),
    }


# ---------------------------------------------------------------------------
# 2D Publication-Grade SVG Rendering
# ---------------------------------------------------------------------------

def mol_to_svg(
    mol: Chem.Mol,
    highlight_atoms: Optional[list[int]] = None,
    width: int = 340,
    height: int = 230,
) -> str:
    """
    Render a clean, ACS-style 2D molecular structure as an inline SVG string.
    Optionally highlights toxicophore/alert atoms in translucent crimson.
    """
    draw_mol = Chem.Mol(mol)
    try:
        rdDepictor.Compute2DCoords(draw_mol)
    except Exception:
        pass

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 1.6
    opts.padding = 0.08
    opts.minFontSize = 11
    opts.maxFontSize = 15
    opts.addStereoAnnotation = True
    opts.setHighlightColour((0.647, 0.110, 0.188, 0.28))  # Harvard Crimson tint

    hl_atoms = list(highlight_atoms) if highlight_atoms else []
    hl_bonds: list[int] = []
    if hl_atoms:
        atom_set = set(hl_atoms)
        for bond in draw_mol.GetBonds():
            if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set:
                hl_bonds.append(bond.GetIdx())

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        draw_mol,
        highlightAtoms=hl_atoms,
        highlightBonds=hl_bonds,
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # Strip XML declaration for clean inline HTML/React embedding
    if svg.startswith("<?xml"):
        svg = svg.split("?>", 1)[1].lstrip()
    return svg


# ---------------------------------------------------------------------------
# Full Molecular Profile
# ---------------------------------------------------------------------------

def analyze_molecule(mol: Chem.Mol, include_svg: bool = True) -> dict[str, Any]:
    """
    Compute the complete academic medicinal chemistry profile for a
    standardized RDKit Mol.
    """
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    try:
        inchi_key = Chem.MolToInchiKey(mol)
    except Exception:
        inchi_key = "N/A"
    scaffold = get_murcko_scaffold(mol)

    mw = float(Descriptors.MolWt(mol))
    clogp = float(Crippen.MolLogP(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    rotb = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    arom_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))
    heavy_atoms = int(mol.GetNumHeavyAtoms())
    formal_charge = int(Chem.GetFormalCharge(mol))
    mr = float(Crippen.MolMR(mol))
    fsp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
    qed_score = float(QED.qed(mol))
    bertz_ct = float(GraphDescriptors.BertzCT(mol)) if heavy_atoms > 1 else 0.0

    # --- Medicinal chemistry rule evaluations ---
    lipinski_violations: list[str] = []
    if mw > 500.0:
        lipinski_violations.append(f"MW {mw:.1f} > 500 Da")
    if clogp > 5.0:
        lipinski_violations.append(f"cLogP {clogp:.2f} > 5.0")
    if hbd > 5:
        lipinski_violations.append(f"HBD {hbd} > 5")
    if hba > 10:
        lipinski_violations.append(f"HBA {hba} > 10")

    veber_pass = (tpsa <= 140.0) and (rotb <= 10)
    pfizer_3_75_flag = (clogp > 3.0) and (tpsa < 75.0)
    gsk_4_400_pass = (mw <= 400.0) and (clogp <= 4.0)

    alerts = screen_structural_alerts(mol)
    boiled_egg = evaluate_boiled_egg(tpsa=tpsa, wlogp=clogp)

    svg = (
        mol_to_svg(mol, highlight_atoms=alerts["highlight_atoms"])
        if include_svg
        else None
    )

    return {
        "identifiers": {
            "canonical_smiles": canonical_smiles,
            "molecular_formula": formula,
            "inchi_key": inchi_key,
            "murcko_scaffold": scaffold,
        },
        "physchem": {
            "mw": round(mw, 2),
            "clogp": round(clogp, 3),
            "tpsa": round(tpsa, 2),
            "hbd": hbd,
            "hba": hba,
            "rotatable_bonds": rotb,
            "aromatic_rings": arom_rings,
            "heavy_atoms": heavy_atoms,
            "formal_charge": formal_charge,
            "molar_refractivity": round(mr, 2),
            "fsp3": round(fsp3, 3),
            "qed": round(qed_score, 3),
            "bertz_ct": round(bertz_ct, 1),
        },
        "rules": {
            "lipinski_pass": len(lipinski_violations) <= 1,
            "lipinski_violations": len(lipinski_violations),
            "lipinski_details": lipinski_violations,
            "veber_pass": veber_pass,
            "pfizer_3_75_warning": pfizer_3_75_flag,
            "gsk_4_400_pass": gsk_4_400_pass,
        },
        "alerts": {
            "pains_count": alerts["pains_count"],
            "pains_alerts": alerts["pains_alerts"],
            "brenk_count": alerts["brenk_count"],
            "brenk_alerts": alerts["brenk_alerts"],
            "total_alerts": alerts["total_alerts"],
        },
        "boiled_egg": boiled_egg,
        "svg_2d": svg,
    }


def max_tanimoto_similarity(query_fp, ref_fps: list) -> tuple[float, int]:
    """
    Compute maximum Tanimoto similarity against a corpus of ECFP4 fingerprints
    (Applicability Domain assessment). Returns (max_sim, best_idx).
    """
    if not ref_fps:
        return 0.0, -1
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, ref_fps)
    best_idx = int(np.argmax(sims))
    return float(sims[best_idx]), best_idx

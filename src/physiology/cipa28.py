"""
ADMET-Engine v3.0 (CardioPhys-3D) — Official FDA-CiPA 28 Reference Drugs
========================================================================
Curated clinical validation benchmark of the 28 drugs selected by the FDA
Comprehensive in vitro Proarrhythmia Assay (CiPA) initiative across three
clinical Torsades de Pointes (TdP) risk categories:
  - High TdP Risk (8 compounds)
  - Intermediate TdP Risk (11 compounds)
  - Low / No TdP Risk (9 compounds)

Pharmacological parameters:
  - SMILES: IUPAC / PubChem verified canonical SMILES
  - cmax_free_uM: Effective Free Therapeutic Plasma Concentration (EFTPC, uM)
  - ic50_herg_uM, h_herg: I_Kr (hERG / KCNH2) patch-clamp IC50 (uM) & Hill slope
  - ic50_cav12_uM, h_cav12: I_CaL (Cav1.2 / CACNA1C) IC50 (uM) & Hill slope
  - ic50_nav15_late_uM, h_nav15_late: I_NaL (Nav1.5 late / SCN5A) IC50 (uM) & Hill slope
  - ic50_nav15_peak_uM, h_nav15_peak: I_Na (Nav1.5 peak / SCN5A) IC50 (uM) & Hill slope

Primary literature sources:
  - Colatsky, T. et al. J. Pharmacol. Toxicol. Methods 2016, 81, 15-20.
  - Crumb, W. J. et al. J. Pharmacol. Toxicol. Methods 2016, 81, 251-262.
  - Li, Z. et al. Clin. Pharmacol. Ther. 2019, 105 (2), 466-475.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional

import pandas as pd

TdPRiskCategory = Literal["High", "Intermediate", "Low"]


@dataclass(frozen=True)
class CiPADrug:
    name: str
    tdp_risk: TdPRiskCategory
    cipa_split: Literal["Training-12", "Validation-16"]
    smiles: str
    cmax_free_uM: float
    ic50_herg_uM: Optional[float]
    h_herg: float = 1.0
    ic50_cav12_uM: Optional[float] = None
    h_cav12: float = 1.0
    ic50_nav15_late_uM: Optional[float] = None
    h_nav15_late: float = 1.0
    ic50_nav15_peak_uM: Optional[float] = None
    h_nav15_peak: float = 1.0
    clinical_note: str = ""


# Official 28 FDA-CiPA drugs with published multi-channel patch-clamp parameters
# (None = no significant block up to highest tested concentration -> treated as >1e6 uM)
CIPA_28_DRUGS: list[CiPADrug] = [
    # -----------------------------------------------------------------------
    # HIGH TdP RISK (8)
    # -----------------------------------------------------------------------
    CiPADrug(
        name="Azimilide",
        tdp_risk="High",
        cipa_split="Validation-16",
        smiles="CN1CCN(Cc2ccc(-c3ccc(Cl)cc3)o2)CC1=NN1CC(=O)NC1=O",
        cmax_free_uM=0.070,
        ic50_herg_uM=0.35, h_herg=0.95,
        ic50_cav12_uM=18.0, h_cav12=1.0,
        ic50_nav15_late_uM=None,
        ic50_nav15_peak_uM=None,
        clinical_note="Class III antiarrhythmic; potent I_Kr blocker with high TdP incidence.",
    ),
    CiPADrug(
        name="Bepridil",
        tdp_risk="High",
        cipa_split="Training-12",
        smiles="CC(C)COCC(CN(Cc1ccccc1)c1ccccc1)N1CCCC1",
        cmax_free_uM=0.033,
        ic50_herg_uM=0.033, h_herg=0.90,
        ic50_cav12_uM=2.81, h_cav12=0.65,
        ic50_nav15_late_uM=1.81, h_nav15_late=1.42,
        ic50_nav15_peak_uM=2.93, h_nav15_peak=1.20,
        clinical_note="Withdrawn calcium antagonist; extreme hERG potency outweighs Cav1.2 block.",
    ),
    CiPADrug(
        name="Disopyramide",
        tdp_risk="High",
        cipa_split="Validation-16",
        smiles="CC(C)N(CCC(C(N)=O)(c1ccccc1)c1ccccn1)C(C)C",
        cmax_free_uM=0.742,
        ic50_herg_uM=7.23, h_herg=0.92,
        ic50_cav12_uM=1036.0, h_cav12=1.0,
        ic50_nav15_late_uM=None,
        ic50_nav15_peak_uM=168.0, h_nav15_peak=1.0,
        clinical_note="Class Ia antiarrhythmic; high free Cmax yields strong I_Kr inhibition.",
    ),
    CiPADrug(
        name="Dofetilide",
        tdp_risk="High",
        cipa_split="Training-12",
        smiles="CN(CCOc1ccc(NS(C)(=O)=O)cc1)CCc1ccc(NS(C)(=O)=O)cc1",
        cmax_free_uM=0.0020,
        ic50_herg_uM=0.0049, h_herg=0.94,
        ic50_cav12_uM=None,
        ic50_nav15_late_uM=None,
        ic50_nav15_peak_uM=None,
        clinical_note="Selective pure I_Kr blocker; benchmark high-TdP positive control.",
    ),
    CiPADrug(
        name="Ibutilide",
        tdp_risk="High",
        cipa_split="Validation-16",
        smiles="CCCCCCC(CC)N(CC)CCCC(O)c1ccc(NS(C)(=O)=O)cc1",
        cmax_free_uM=0.140,
        ic50_herg_uM=0.018, h_herg=0.90,
        ic50_cav12_uM=None,
        ic50_nav15_late_uM=None,
        ic50_nav15_peak_uM=None,
        clinical_note="Class III antiarrhythmic; nanomolar I_Kr block + late Na activation.",
    ),
    CiPADrug(
        name="Quinidine",
        tdp_risk="High",
        cipa_split="Training-12",
        smiles="C=C[C@H]1CN2CC[C@H]1C[C@@H]2[C@@H](O)c1ccnc2ccc(OC)cc12",
        cmax_free_uM=3.237,
        ic50_herg_uM=0.992, h_herg=0.80,
        ic50_cav12_uM=51.59, h_cav12=0.60,
        ic50_nav15_late_uM=9.42, h_nav15_late=1.30,
        ic50_nav15_peak_uM=12.33, h_nav15_peak=1.50,
        clinical_note="Class Ia antiarrhythmic; high free Cmax drives >75% I_Kr block.",
    ),
    CiPADrug(
        name="Sotalol",
        tdp_risk="High",
        cipa_split="Training-12",
        smiles="CC(C)NCC(O)c1ccc(NS(C)(=O)=O)cc1",
        cmax_free_uM=14.69,
        ic50_herg_uM=81.3, h_herg=0.80,
        ic50_cav12_uM=None,
        ic50_nav15_late_uM=None,
        ic50_nav15_peak_uM=None,
        clinical_note="Class III beta-blocker; very high free Cmax causes selective I_Kr block.",
    ),
    CiPADrug(
        name="Vandetanib",
        tdp_risk="High",
        cipa_split="Validation-16",
        smiles="COc1cc2c(Nc3ccc(Br)cc3F)ncnc2cc1OCC1CCN(C)CC1",
        cmax_free_uM=0.255,
        ic50_herg_uM=0.40, h_herg=0.95,
        ic50_cav12_uM=12.0, h_cav12=1.0,
        ic50_nav15_late_uM=None,
        ic50_nav15_peak_uM=None,
        clinical_note="Kinase inhibitor with FDA black-box QT/TdP warning.",
    ),

    # -----------------------------------------------------------------------
    # INTERMEDIATE TdP RISK (11)
    # -----------------------------------------------------------------------
    CiPADrug(
        name="Astemizole",
        tdp_risk="Intermediate",
        cipa_split="Validation-16",
        smiles="COc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc3ccc(F)cc3)CC2)cc1",
        cmax_free_uM=0.00026,
        ic50_herg_uM=0.0011, h_herg=1.0,
        ic50_cav12_uM=1.10, h_cav12=1.0,
        ic50_nav15_late_uM=None,
        ic50_nav15_peak_uM=3.0, h_nav15_peak=1.0,
        clinical_note="Withdrawn antihistamine; extreme hERG potency (1 nM), low free Cmax.",
    ),
    CiPADrug(
        name="Chlorpromazine",
        tdp_risk="Intermediate",
        cipa_split="Training-12",
        smiles="CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21",
        cmax_free_uM=0.038,
        ic50_herg_uM=1.12, h_herg=0.90,
        ic50_cav12_uM=8.19, h_cav12=0.80,
        ic50_nav15_late_uM=4.56, h_nav15_late=1.0,
        ic50_nav15_peak_uM=4.54, h_nav15_peak=2.0,
        clinical_note="Typical antipsychotic; moderate hERG block with modest Cav1.2/NaL offset.",
    ),
    CiPADrug(
        name="Cisapride",
        tdp_risk="Intermediate",
        cipa_split="Training-12",
        smiles="COc1cc(N)c(Cl)cc1C(=O)NC1CCN(CCCOc2ccc(F)cc2)CC1OC",
        cmax_free_uM=0.0026,
        ic50_herg_uM=0.012, h_herg=0.90,
        ic50_cav12_uM=11.8, h_cav12=1.0,
        ic50_nav15_late_uM=None,
        ic50_nav15_peak_uM=None,
        clinical_note="Withdrawn prokinetic; selective hERG blocker at nanomolar concentrations.",
    ),
    CiPADrug(
        name="Clarithromycin",
        tdp_risk="Intermediate",
        cipa_split="Validation-16",
        smiles="CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@@H](N(C)C)[C@H]2O)[C@](C)(OC)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O",
        cmax_free_uM=1.206,
        ic50_herg_uM=45.7, h_herg=0.90,
        ic50_cav12_uM=None,
        ic50_nav15_late_uM=None,
        ic50_nav15_peak_uM=None,
        clinical_note="Macrolide antibiotic; clinical QT prolongation at high free exposure.",
    ),
    CiPADrug(
        name="Clozapine",
        tdp_risk="Intermediate",
        cipa_split="Validation-16",
        smiles="CN1CCN(C2=Nc3cc(Cl)ccc3Nc3ccccc32)CC1",
        cmax_free_uM=0.071,
        ic50_herg_uM=2.29, h_herg=0.95,
        ic50_cav12_uM=3.60, h_cav12=1.0,
        ic50_nav15_late_uM=None,
        ic50_nav15_peak_uM=15.0, h_nav15_peak=1.0,
        clinical_note="Atypical antipsychotic; mixed hERG and Cav1.2 blockade.",
    ),
    CiPADrug(
        name="Domperidone",
        tdp_risk="Intermediate",
        cipa_split="Validation-16",
        smiles="O=c1[nH]c2ccc(Cl)cc2n1CCCN1CCC(n2c(=O)[nH]c3ccccc32)CC1",
        cmax_free_uM=0.019,
        ic50_herg_uM=0.057, h_herg=0.90,
        ic50_cav12_uM=4.0, h_cav12=1.0,
        ic50_nav15_late_uM=None,
        ic50_nav15_peak_uM=None,
        clinical_note="Peripheral D2 antagonist; strong selective hERG block.",
    ),
    CiPADrug(
        name="Droperidol",
        tdp_risk="Intermediate",
        cipa_split="Validation-16",
        smiles="O=C(CCCN1CC=C(n2c(=O)[nH]c3ccccc32)CC1)c1ccc(F)cc1",
        cmax_free_uM=0.016,
        ic50_herg_uM=0.060, h_herg=0.90,
        ic50_cav12_uM=7.6, h_cav12=1.0,
        ic50_nav15_late_uM=None,
        ic50_nav15_peak_uM=None,
        clinical_note="Butyrophenone antiemetic; potent hERG inhibition.",
    ),
    CiPADrug(
        name="Ondansetron",
        tdp_risk="Intermediate",
        cipa_split="Training-12",
        smiles="Cc1nccn1CC1CCc2c(c3ccccc3n2C)C1=O",
        cmax_free_uM=0.139,
        ic50_herg_uM=1.49, h_herg=0.90,
        ic50_cav12_uM=22.55, h_cav12=0.80,
        ic50_nav15_late_uM=19.18, h_nav15_late=1.0,
        ic50_nav15_peak_uM=57.67, h_nav15_peak=1.0,
        clinical_note="5-HT3 antagonist; dose-dependent QT prolongation.",
    ),
    CiPADrug(
        name="Pimozide",
        tdp_risk="Intermediate",
        cipa_split="Validation-16",
        smiles="O=c1[nH]c2ccccc2n1C1CCN(CCCC(c2ccc(F)cc2)c2ccc(F)cc2)CC1",
        cmax_free_uM=0.00043,
        ic50_herg_uM=0.0042, h_herg=0.90,
        ic50_cav12_uM=0.16, h_cav12=1.0,
        ic50_nav15_late_uM=None,
        ic50_nav15_peak_uM=None,
        clinical_note="Antipsychotic; potent hERG blocker partially tempered by Cav1.2.",
    ),
    CiPADrug(
        name="Risperidone",
        tdp_risk="Intermediate",
        cipa_split="Validation-16",
        smiles="Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2",
        cmax_free_uM=0.0018,
        ic50_herg_uM=0.148, h_herg=0.90,
        ic50_cav12_uM=30.0, h_cav12=1.0,
        ic50_nav15_late_uM=None,
        ic50_nav15_peak_uM=None,
        clinical_note="Atypical antipsychotic; moderate-to-strong hERG liability.",
    ),
    CiPADrug(
        name="Terfenadine",
        tdp_risk="Intermediate",
        cipa_split="Training-12",
        smiles="CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1",
        cmax_free_uM=0.00029,
        ic50_herg_uM=0.050, h_herg=0.90,
        ic50_cav12_uM=0.70, h_cav12=0.70,
        ic50_nav15_late_uM=20.06, h_nav15_late=0.60,
        ic50_nav15_peak_uM=4.80, h_nav15_peak=1.0,
        clinical_note="Withdrawn antihistamine; TdP triggered especially upon CYP3A4 DDI (10-50x Cmax).",
    ),

    # -----------------------------------------------------------------------
    # LOW / NO TdP RISK (9) — Includes Verapamil & Ranolazine!
    # -----------------------------------------------------------------------
    CiPADrug(
        name="Diltiazem",
        tdp_risk="Low",
        cipa_split="Training-12",
        smiles="COc1ccc([C@@H]2Sc3ccccc3N(CCN(C)C)C(=O)[C@@H]2OC(C)=O)cc1",
        cmax_free_uM=0.128,
        ic50_herg_uM=13.2, h_herg=0.80,
        ic50_cav12_uM=0.112, h_cav12=0.70,
        ic50_nav15_late_uM=21.9, h_nav15_late=0.70,
        ic50_nav15_peak_uM=111.0, h_nav15_peak=0.70,
        clinical_note="Calcium channel blocker; dominant I_CaL block protects against EADs.",
    ),
    CiPADrug(
        name="Loratadine",
        tdp_risk="Low",
        cipa_split="Validation-16",
        smiles="CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc32)CC1",
        cmax_free_uM=0.00045,
        ic50_herg_uM=6.10, h_herg=1.0,
        ic50_cav12_uM=11.4, h_cav12=1.0,
        ic50_nav15_late_uM=None,
        ic50_nav15_peak_uM=None,
        clinical_note="Second-generation antihistamine; wide hERG safety margin (>10,000x).",
    ),
    CiPADrug(
        name="Metoprolol",
        tdp_risk="Low",
        cipa_split="Validation-16",
        smiles="COCCc1ccc(OCC(O)CNC(C)C)cc1",
        cmax_free_uM=1.80,
        ic50_herg_uM=145.0, h_herg=0.90,
        ic50_cav12_uM=None,
        ic50_nav15_late_uM=None,
        ic50_nav15_peak_uM=None,
        clinical_note="Beta-1 blocker; negligible I_Kr block at therapeutic free exposure.",
    ),
    CiPADrug(
        name="Mexiletine",
        tdp_risk="Low",
        cipa_split="Training-12",
        smiles="Cc1cccc(C)c1OCC(C)N",
        cmax_free_uM=2.50,
        ic50_herg_uM=50.0, h_herg=0.90,
        ic50_cav12_uM=38.2, h_cav12=1.0,
        ic50_nav15_late_uM=8.96, h_nav15_late=1.40,
        ic50_nav15_peak_uM=43.0, h_nav15_peak=1.0,
        clinical_note="Class Ib antiarrhythmic; I_NaL inhibition shortens APD and prevents TdP.",
    ),
    CiPADrug(
        name="Nifedipine",
        tdp_risk="Low",
        cipa_split="Validation-16",
        smiles="COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]",
        cmax_free_uM=0.0077,
        ic50_herg_uM=440.0, h_herg=1.0,
        ic50_cav12_uM=0.012, h_cav12=1.0,
        ic50_nav15_late_uM=None,
        ic50_nav15_peak_uM=None,
        clinical_note="Dihydropyridine CCB; ultra-potent Cav1.2 block, no TdP risk.",
    ),
    CiPADrug(
        name="Nitrendipine",
        tdp_risk="Low",
        cipa_split="Validation-16",
        smiles="CCOC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1cccc([N+](=O)[O-])c1",
        cmax_free_uM=0.0030,
        ic50_herg_uM=25.0, h_herg=1.0,
        ic50_cav12_uM=0.015, h_cav12=1.0,
        ic50_nav15_late_uM=None,
        ic50_nav15_peak_uM=None,
        clinical_note="Dihydropyridine CCB; potent Cav1.2 block prevents APD prolongation.",
    ),
    CiPADrug(
        name="Ranolazine",
        tdp_risk="Low",
        cipa_split="Training-12",
        smiles="COc1ccccc1OCC(O)CN1CCN(CC(=O)Nc2c(C)cccc2C)CC1",
        cmax_free_uM=1.948,
        ic50_herg_uM=8.27, h_herg=0.95,
        ic50_cav12_uM=296.0, h_cav12=1.0,
        ic50_nav15_late_uM=7.88, h_nav15_late=0.95,
        ic50_nav15_peak_uM=68.8, h_nav15_peak=1.40,
        clinical_note="Canonical hERG false-positive rescued by balanced I_NaL (Nav1.5 late) block.",
    ),
    CiPADrug(
        name="Tamoxifen",
        tdp_risk="Low",
        cipa_split="Validation-16",
        smiles="CC/C(=C(\\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
        cmax_free_uM=0.021,
        ic50_herg_uM=1.00, h_herg=1.0,
        ic50_cav12_uM=1.20, h_cav12=1.0,
        ic50_nav15_late_uM=None,
        ic50_nav15_peak_uM=None,
        clinical_note="SERM; modest hERG block offset by equi-potent Cav1.2 inhibition.",
    ),
    CiPADrug(
        name="Verapamil",
        tdp_risk="Low",
        cipa_split="Training-12",
        smiles="COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC",
        cmax_free_uM=0.081,
        ic50_herg_uM=0.499, h_herg=1.0,
        ic50_cav12_uM=0.202, h_cav12=1.10,
        ic50_nav15_late_uM=None,
        ic50_nav15_peak_uM=41.5, h_nav15_peak=1.0,
        clinical_note="Canonical hERG false-positive (IC50=0.5 uM) rescued by potent Cav1.2 (0.2 uM) block.",
    ),
]


def get_cipa28_dataframe() -> pd.DataFrame:
    """Return the official 28 FDA-CiPA drugs as a structured DataFrame."""
    return pd.DataFrame([asdict(d) for d in CIPA_28_DRUGS])


def get_cipa28_drug(name: str) -> CiPADrug:
    """Case-insensitive lookup of a reference drug in the FDA-CiPA 28 panel."""
    key = name.strip().lower()
    for drug in CIPA_28_DRUGS:
        if drug.name.lower() == key:
            return drug
    raise KeyError(f"Drug '{name}' not found in FDA-CiPA 28 panel.")


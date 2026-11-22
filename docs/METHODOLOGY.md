# ADMET-Engine v2.0 — Computational Methodology & Validation Protocol

This document specifies the cheminformatics standardization, graph/descriptor featurization, Bemis-Murcko scaffold partitioning, multi-modal ensemble calibration, and analytical medicinal chemistry modules implemented in **ADMET-Engine v2.0**.

---

## 1. Molecular Standardization (`src/data_pipeline.py`)

Raw chemical databases frequently contain counterions, tautomeric variants, non-canonical charge representations, and salt mixtures that introduce label conflicts and duplicate splits. Every molecule entering training or inference passes through a deterministic 8-stage RDKit `MolStandardize` pipeline:

1. **SMILES Parsing & Validation** (`Chem.MolFromSmiles`)
2. **Salt & Fragment Stripping** (`SaltRemover` followed by `LargestFragmentChooser(preferOrganic=True)` to remove unbound counterions such as `Br.`, `[Na+]`, `[Cl-]`)
3. **Explicit Hydrogen Removal** (`Chem.RemoveHs`)
4. **Functional Group Normalization** (`rdMolStandardize.Normalizer`)
5. **Preferred Charge State Reionization** (`rdMolStandardize.Reionizer`)
6. **Canonical Tautomer Enumeration** (`rdMolStandardize.TautomerEnumerator.Canonicalize`)
7. **Stereochemistry Assignment** (`Chem.AssignStereochemistry`)
8. **Kekulization & Re-Sanitization** (`Chem.Kekulize` + `Chem.SanitizeMol`)

All RDKit standardizer instances are cached at module scope for batch throughput.

---

## 2. Strict Bemis-Murcko Scaffold Splitting (`src/dataset.py`, `src/training/train_chembl_ensemble.py`)

In QSAR and ADMET modeling, random train/test splitting leaks congeneric chemical series (analogs sharing a core ring scaffold) across partitions, substantially inflating apparent ROC-AUC.

ADMET-Engine extracts the canonical **Bemis-Murcko scaffold** (`MurckoScaffoldSmiles`) for every standardized compound and partitions scaffold clusters deterministically into **80% train / 10% calibration-validation / 10% held-out test** following the MoleculeNet protocol (Wu et al., *Chem. Sci.* 2018):

- **Zero Core Overlap:** Assert-verified `train_scaffolds ∩ val_scaffolds = ∅` and `train_scaffolds ∩ test_scaffolds = ∅`.
- **Leakage-Free Preprocessing:** Continuous physicochemical descriptor standardization ($\mu, \sigma$), temperature calibration ($T^*$), and convex ensemble weights ($w^*$) are fitted exclusively on train/validation partitions; the held-out scaffold test set is evaluated strictly once.

---

## 3. Multi-Modal Molecular Representations & Architecture (`src/featurizer.py`, `src/models/`)

ADMET-Engine v2.0 combines four structurally distinct predictive model families over two complementary molecular representations:

### 3.1 Chemprop-Grade One-Hot Directed Message-Passing Neural Network (`DMPNNEncoder` + `ADMETPredictor`)
- **Atom Features (45 dims):** One-hot atomic symbol (13), total degree (7), formal charge (6), hybridization (6), chirality (4), total hydrogens (6), aromaticity (1), ring membership (1), and scaled atomic mass (`mass / 100.0`, 1).
- **Bond Features (10 dims):** One-hot bond order (`SINGLE`, `DOUBLE`, `TRIPLE`, `AROMATIC`, 4), conjugation (1), ring membership (1), and bond stereochemistry (4).
- **Directed Bond Message Passing ($T = 3$):**
  $$m_{vw}^{(t+1)} = \sum_{k \in \mathcal{N}(v) \setminus \{w\}} h_{kv}^{(t)}, \qquad h_{vw}^{(t+1)} = \operatorname{ReLU}\!\left(h_{vw}^{(0)} + W_m m_{vw}^{(t+1)}\right)$$
  Reverse edge lookup uses an $O(E)$ adjacent-pair fast-path with an $O(E \log E)$ sorted fallback (eliminating $O(N^2)$ batch memory allocation) and explicit support for edge-free single-atom molecules.
- **Readout:** `Set2Set` attention pooling followed by task-specific MLP heads trained with NaN-masked `BCEWithLogitsLoss`.

### 3.2 Fingerprint + Physicochemical Hybrid Models
Every standardized compound is also encoded as a 2,068-dimensional vector concatenating a **2048-bit Morgan ECFP4 fingerprint** (radius 2) with **20 standardized RDKit 2D physicochemical descriptors** (MW, WLOGP, TPSA, HBD, HBA, RotB, aromatic/aliphatic/hetero rings, heavy atoms, Fsp³, molar refractivity, formal charge, QED, basic nitrogens, halogens, heteroatoms, Labute ASA, Balaban J, and $\log(1 + \text{BertzCT})$):
- **Residual ECFP4 + PhysChem Deep Network (`FingerprintPhysChemNet`):** LayerNorm + GELU residual MLP trained with multi-task NaN-masked BCE.
- **ExtraTrees Non-Linear QSAR Classifier:** 220 randomized trees capturing non-linear pharmacophore interactions.
- **L2-Regularized Logistic QSAR Classifier:** High-dimensional regularized linear baseline in ECFP4 + descriptor space.

---

## 4. Calibration, Epistemic Uncertainty & Applicability Domain

- **Post-Hoc Temperature Scaling:** For each model family $m$ and endpoint $t$, a scalar temperature $T_{m,t} \in [0.25, 4.0]$ is optimized on the scaffold validation set to minimize negative log-likelihood (NLL):
  $$\hat{p}_{m,t} = \sigma\!\left(\frac{\operatorname{logit}(p_{m,t})}{T_{m,t}}\right)$$
- **Consensus Probability & Epistemic Uncertainty:**
  $$\bar{p}_t = \sum_m w_{m,t}\,\hat{p}_{m,t}, \qquad \sigma_{\text{epi},t} = \sqrt{\sum_m w_{m,t}\left(\hat{p}_{m,t} - \bar{p}_t\right)^2}$$
  with 95% confidence intervals clipped to $[0, 1]$.
- **Applicability Domain (AD):** Exact vectorized bitwise Tanimoto similarity against the packed uint8 ECFP4 matrix of the ChEMBL v37 training corpus:
  - **In-Domain:** $T_{\max} \ge 0.40$
  - **Borderline:** $0.25 \le T_{\max} < 0.40$
  - **Out-of-Domain:** $T_{\max} < 0.25$

---

## 5. Deterministic Medicinal Chemistry Modules (`src/cheminformatics.py`)

Alongside ML predictions, every query molecule receives deterministic literature-grounded profiling:
- **BOILED-Egg Permeation Model** (Daina & Zoete, *ChemMedChem* 2016): Exact rotated ellipse membership in `(Ertl TPSA, Wildman-Crippen WLOGP)` space for Human Intestinal Absorption (White) and Blood-Brain Barrier Permeation (Yolk).
- **Drug-Likeness & Safety Filters:** Lipinski Rule of 5 (1997), Veber Oral Bioavailability (2002), Pfizer 3/75 In Vivo Toxicity Rule (Hughes et al., *BMCL* 2008), GSK 4/400 (Gleeson, *J. Med. Chem.* 2008), QED (Bickerton et al., *Nature Chemistry* 2012), and Fsp³ (Lovering et al., *J. Med. Chem.* 2009).
- **Substructure Toxicophore Screening:** RDKit `FilterCatalog` screening against **PAINS A/B/C** (Baell & Holloway, *J. Med. Chem.* 2010) and **Brenk** (2008) alerts, with matching atoms highlighted directly on the 2D RDKit SVG structure rendering.

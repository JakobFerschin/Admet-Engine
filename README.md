# ADMET-Engine

**Computational Medicinal Chemistry & Preclinical Safety Workbench** — a full-stack ADMET prediction platform combining a scaffold-validated **ChEMBL v37 Multi-Modal Ensemble** (45-dim one-hot Chemprop DMPNN + Residual ECFP4/PhysChem Deep Network + ExtraTrees QSAR + L2 Logistic QSAR) with deterministic RDKit cheminformatics, Tanimoto Applicability Domain assessment, and an academic React/TypeScript workbench.

---

## Scientific Highlights

- **Strict Bemis-Murcko Scaffold Validation:** Trained and evaluated on **16,270 unique standardized ChEMBL v37 compounds** partitioned with **zero core scaffold overlap** (`13,016` train / `1,627` calibration-val / `1,627` held-out test across `8,252` unique scaffolds).
- **Calibrated Multi-Modal Consensus + Epistemic Uncertainty:** Post-hoc validation-set temperature scaling minimizes negative log-likelihood; disagreement across the four distinct model families yields calibrated probabilities $\bar{p} \pm \sigma_{\text{epi}}$ with 95% confidence intervals.
- **Vectorized Applicability Domain (AD):** Exact sub-millisecond ECFP4 Tanimoto similarity ($T_{\max}$ and top-5 mean) against the ChEMBL v37 training corpus, surfacing the closest experimental training analog and flagging `In-Domain` ($T_{\max} \ge 0.40$), `Borderline` ($0.25$–$0.40$), or `Out-of-Domain` ($< 0.25$).
- **Deterministic Medicinal Chemistry Suite:**
  - Publication-grade **2D RDKit SVG rendering** with crimson highlighting of **PAINS A/B/C** (Baell & Holloway, *J. Med. Chem.* 2010) and **Brenk** (2008) toxicophores.
  - Exact analytical **BOILED-Egg Permeation Map** (Daina & Zoete, *ChemMedChem* 2016) predicting passive GI absorption (White) and BBB permeation (Yolk).
  - **Physicochemical & Compliance Profiling:** MW, Wildman-Crippen WLOGP, Ertl TPSA, HBD/HBA, RotB, **QED** (Bickerton et al., *Nature Chemistry* 2012), **Fsp³** (Lovering et al., *J. Med. Chem.* 2009), Lipinski Rule of 5, Veber, Pfizer 3/75 (Hughes et al., *BMCL* 2008), and GSK 4/400.

---

## Held-Out Bemis-Murcko Scaffold Benchmark (Zero Core Overlap)

Evaluated strictly on the untouched scaffold holdout test split (`n_test = 1,627` compounds across `1,627` unseen core scaffolds):

### hERG Channel Blockade (`KCNH2 / CHEMBL240`, `n = 1,177`)

| Model Architecture | ROC-AUC | PR-AUC | Brier ↓ | ECE ↓ | F1 |
|---|---:|---:|---:|---:|---:|
| Chemprop One-Hot DMPNN (45-dim atom / 10-dim bond) | 0.7330 | 0.8595 | 0.1787 | **0.0219** | 0.8260 |
| Residual ECFP4 + 20 PhysChem MLP | 0.7685 | 0.8794 | 0.1689 | 0.0365 | **0.8405** |
| ExtraTrees Non-Linear QSAR | 0.7739 | 0.8909 | 0.1705 | 0.0455 | 0.8304 |
| L2-Regularized Logistic QSAR | 0.7447 | 0.8686 | 0.1780 | 0.0410 | 0.8251 |
| **Calibrated Consensus Ensemble (v2.0)** | **0.7836** | **0.8955** | **0.1656** | 0.0328 | 0.8385 |

### CYP3A4 Metabolic Inhibition (`CYP3A4 / CHEMBL340`, `n = 516`)

| Model Architecture | ROC-AUC | PR-AUC | Brier ↓ | ECE ↓ | F1 |
|---|---:|---:|---:|---:|---:|
| Chemprop One-Hot DMPNN (45-dim atom / 10-dim bond) | 0.6464 | 0.7685 | 0.2183 | 0.0638 | 0.7365 |
| Residual ECFP4 + 20 PhysChem MLP | 0.7475 | 0.8315 | 0.1924 | **0.0430** | 0.7719 |
| ExtraTrees Non-Linear QSAR | **0.7685** | **0.8538** | 0.1878 | 0.0657 | **0.8037** |
| L2-Regularized Logistic QSAR | 0.7327 | 0.8162 | 0.1989 | 0.0566 | 0.7821 |
| **Calibrated Consensus Ensemble (v2.0)** | 0.7664 | 0.8506 | **0.1875** | 0.0541 | 0.7911 |

### Clinical Pharmacology Reference Check

| Compound | Clinical Profile | hERG $P(\text{active})$ | CYP3A4 $P(\text{active})$ | Applicability Domain |
|---|---|---:|---:|---|
| Astemizole | Withdrawn — Potent hERG blocker ($\text{IC}_{50} \approx 1\text{ nM}$) | **96.7%** ± 2.6% | 71.3% ± 9.3% | In-Domain ($T = 1.00$) |
| Cisapride | Withdrawn — hERG QT prolongation | **91.5%** ± 6.0% | 78.6% ± 4.0% | In-Domain ($T = 1.00$) |
| Haloperidol | FDA QT warning — Known hERG liability | **87.1%** ± 6.0% | 44.4% ± 9.2% | In-Domain ($T = 1.00$) |
| Terfenadine | Withdrawn — hERG / Torsades de Pointes | **84.6%** ± 9.7% | 72.2% ± 11.7% | In-Domain ($T = 1.00$) |
| Ketoconazole | FDA index strong CYP3A4 inhibitor ($K_i \approx 15\text{ nM}$) | 74.0% ± 11.8% | **93.3%** ± 8.2% | In-Domain ($T = 1.00$) |
| Aspirin | Safe clinical control | 26.8% ± 10.7% | 25.3% ± 7.3% | Borderline ($T = 0.34$) |
| Caffeine | Safe clinical control (CYP1A2 substrate) | 30.6% ± 16.8% | 27.0% ± 8.7% | Out-of-Domain ($T = 0.22$) |
| Acetaminophen | Safe clinical control | 31.0% ± 10.7% | 31.3% ± 9.7% | In-Domain ($T = 0.42$) |

---

## Quickstart

### 1. Python Backend (FastAPI)

```bash
# From the repository root:
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Start the scientific inference API (port 8000)
uvicorn src.api.main:app --reload --port 8000
```

Interactive OpenAPI docs are available at `http://localhost:8000/docs`. Key endpoints:
- `GET /health` — readiness check & active model version
- `GET /model/metadata` — full Bemis-Murcko scaffold ablation table, calibration parameters & provenance
- `POST /predict/admet` — batch ADMET predictions (1–1,000 SMILES), epistemic uncertainty, Applicability Domain, BOILED-Egg, PAINS/Brenk alerts & 2D SVG structures

### 2. Academic Frontend Workbench (React + TypeScript + Vite)

```bash
cd frontend
npm install
npm run dev
```

### 3. Retraining & Verification

```bash
# Retrain the 4-model ChEMBL v37 ensemble under Bemis-Murcko scaffold splitting
python src/training/train_chembl_ensemble.py

# Run the clinical drug backtest
python scripts/backtest_failures.py

# Run the complete pytest suite
pytest tests/ -v
```

---

## Repository Structure

```text
src/
├── api/main.py                    # FastAPI server (/predict/admet, /model/metadata, /health)
├── cheminformatics.py             # RDKit descriptors, BOILED-Egg, PAINS/Brenk, QED, 2D SVG
├── data_pipeline.py               # 8-stage cached RDKit MolStandardize pipeline
├── featurizer.py                  # 45-dim/10-dim Chemprop one-hot + legacy 9/4 featurizer
├── dataset.py                     # PyG Dataset with Bemis-Murcko scaffold splitting
├── models/
│   ├── dmpnn.py                   # Directed Message-Passing Neural Network (O(E) reverse map)
│   └── chembl_ensemble.py         # 4-family calibrated ensemble + packed-bit Tanimoto AD
└── training/
    └── train_chembl_ensemble.py   # End-to-end scaffold training, calibration & benchmarking
frontend/                          # Academic Medicinal Chemistry Workbench (React 19, Vite, TS)
scripts/                           # Backtesting & experimental utilities
tests/                             # Unit, cheminformatics, DMPNN, training & API test suite
docs/                              # Computational methodology & project status reports
```

---

## Documentation

- **Computational Methodology & Literature Basis:** [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)
- **Historical Status Report:** [`docs/STATUS_REPORT.md`](docs/STATUS_REPORT.md)

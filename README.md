# ADMET-Engine

Full-stack system that predicts 12 toxicity endpoints for molecules from SMILES strings, using a DMPNN (graph neural network) behind a FastAPI backend and a React dashboard.

> **Status:** the pipeline works end to end, but the included model was trained on a small mock dataset to validate the setup. Its predictions are not meaningful yet. Training on the full ChEMBL data is the next step.

## Quickstart

```bash
# 1. Start the backend
cd src
uvicorn main:app --reload

# 2. Start the frontend (second terminal)
cd frontend
npm install
npm run dev
```

## How it works

1. Paste one or more SMILES strings into the frontend.
2. The frontend sends them to `POST /predict/admet`.
3. The DMPNN (PyTorch Geometric, RDKit featurization) returns a score per endpoint.
4. The dashboard shows a radar chart and a triage board for all 12 endpoints.

## Stack

- ML: PyTorch, PyTorch Geometric, RDKit
- Backend: FastAPI, Pydantic
- Frontend: React, TypeScript, Vite, Tailwind, Recharts
- Data: ChEMBL ETL with RDKit standardization

## More

Architecture details, test overview, backtesting setup and roadmap: [docs/STATUS_REPORT.md](docs/STATUS_REPORT.md)

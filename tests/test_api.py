"""
ADMET-Engine — API Integration Tests
Phase 4, Month 7–9: Backend & API Testing

Uses FastAPI's TestClient as a context manager for proper lifespan handling
(model loaded on startup).
"""

import math
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure src/ is importable
_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))

import api.main as api_main  # noqa: E402

# Override MODEL_PATH for tests — use the actual saved dummy model
api_main.MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "dmpnn_v1.pt"

from api.main import app  # noqa: E402

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

VALID_ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
VALID_CAFFEINE = "Cn1cnc2n(C)c(=O)n(C)c(=O)c12"
VALID_IBUPROFEN = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
INVALID_SMILES = "C1CC1XXXXYYYYZZZZINVALID"


# ---------------------------------------------------------------------------
# Fixture — TestClient with lifespan
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Enter lifespan (loads model) before tests, tears down after."""
    with TestClient(app) as c:
        yield c


# ============================================================================
# Health check
# ============================================================================

class TestHealthEndpoint:

    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert "device" in body


# ============================================================================
# Prediction endpoint
# ============================================================================

class TestPredictEndpoint:

    def test_single_valid_smiles(self, client):
        resp = client.post("/predict/admet", json={"smiles": [VALID_ASPIRIN]})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["results"]) == 1
        mol = body["results"][0]
        assert mol["status"] == "success"
        assert mol["smiles"] == VALID_ASPIRIN
        assert mol["error"] is None
        assert isinstance(mol["predictions"], dict)
        assert len(mol["predictions"]) == 12

        expected = {
            "hepatotoxicity", "hERG_inhibition", "AMES_mutagenicity",
            "CYP3A4_inhibition", "nephrotoxicity", "CYP2D6_inhibition",
            "CYP2C9_inhibition", "blood_brain_barrier", "plasma_protein_binding",
            "caco2_permeability", "logD_logP", "P_gp_substrate_inhibitor",
        }
        assert set(mol["predictions"].keys()) == expected

    def test_multiple_valid_smiles(self, client):
        resp = client.post("/predict/admet", json={
            "smiles": [VALID_ASPIRIN, VALID_CAFFEINE, VALID_IBUPROFEN],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["results"]) == 3
        for r in body["results"]:
            assert r["status"] == "success"
            assert len(r["predictions"]) == 12

    def test_invalid_smiles_returns_error(self, client):
        resp = client.post("/predict/admet", json={"smiles": [INVALID_SMILES]})
        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert len(body["results"]) == 1
        assert body["results"][0]["status"] == "error"
        assert body["results"][0]["predictions"] is None
        assert "Invalid" in body["results"][0]["error"]

    def test_mixed_batch_valid_and_invalid(self, client):
        smi_list = [VALID_ASPIRIN, INVALID_SMILES, VALID_CAFFEINE]
        resp = client.post("/predict/admet", json={"smiles": smi_list})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 3
        assert results[0]["status"] == "success"
        assert results[1]["status"] == "error"
        assert results[2]["status"] == "success"

    def test_empty_smiles_list_rejected(self, client):
        resp = client.post("/predict/admet", json={"smiles": []})
        assert resp.status_code == 422

    def test_missing_smiles_field_rejected(self, client):
        resp = client.post("/predict/admet", json={})
        assert resp.status_code == 422

    def test_oversized_batch_rejected(self, client):
        resp = client.post("/predict/admet", json={"smiles": ["C"] * 1001})
        assert resp.status_code == 422

    def test_response_contains_timing(self, client):
        resp = client.post("/predict/admet", json={"smiles": [VALID_ASPIRIN]})
        body = resp.json()
        assert body["model_version"] == "dmpnn_v1"
        assert body["prediction_time_ms"] > 0

    def test_predictions_are_finite_numbers(self, client):
        resp = client.post("/predict/admet", json={
            "smiles": [VALID_ASPIRIN, VALID_CAFFEINE],
        })
        for mol in resp.json()["results"]:
            if mol["status"] == "success":
                for val in mol["predictions"].values():
                    assert isinstance(val, (int, float))
                    assert math.isfinite(val), f"Non-finite: {val}"

    def test_salt_form_handled(self, client):
        resp = client.post("/predict/admet", json={
            "smiles": ["CC(=O)Oc1ccccc1C(=O)O.[Na+]"],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["results"][0]["status"] == "success"
        assert len(body["results"][0]["predictions"]) == 12

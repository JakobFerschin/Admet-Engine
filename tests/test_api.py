"""
ADMET-Engine — FastAPI Scientific API Integration Tests
Tests both the ChEMBL v37 Multi-Modal Ensemble mode (calibrated probabilities,
epistemic uncertainty, Applicability Domain, RDKit cheminformatics, 2D SVG,
and /model/metadata) and request validation / error isolation.
"""

import math
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_src = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_src))

import api.main as api_main  # noqa: E402
from api.main import app  # noqa: E402

VALID_ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
VALID_CAFFEINE = "Cn1cnc2n(C)c(=O)n(C)c(=O)c12"
VALID_TERFENADINE = "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1"
VALID_KETOCONAZOLE = "CC(=O)N1CCN(c2ccc(OC[C@H]3CO[C@](Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1"
INVALID_SMILES = "C1CC1XXXXYYYYZZZZINVALID"


@pytest.fixture
def client():
    """Enter lifespan (loads model) before tests, tears down after."""
    with TestClient(app) as c:
        yield c


class TestHealthAndMetadata:

    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert "device" in body
        assert "model_version" in body

    def test_metadata_endpoint(self, client):
        response = client.get("/model/metadata")
        assert response.status_code == 200
        body = response.json()
        assert "model_version" in body
        if body.get("model_version") == "chembl_v37_ensemble_v2":
            assert "scaffold_stats" in body
            assert "ablation_table" in body
            assert "hERG_inhibition" in body["endpoints"]
            assert "CYP3A4_inhibition" in body["endpoints"]


class TestPredictEndpoint:

    def test_single_valid_smiles_calibrated_probs_and_chem(self, client):
        resp = client.post("/predict/admet", json={"smiles": [VALID_ASPIRIN]})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["results"]) == 1
        mol = body["results"][0]
        assert mol["status"] == "success"
        assert mol["smiles"] == VALID_ASPIRIN
        assert mol["error"] is None
        assert isinstance(mol["predictions"], dict)
        assert "hERG_inhibition" in mol["predictions"]
        assert "CYP3A4_inhibition" in mol["predictions"]

        # All predicted probabilities must be strictly calibrated within [0, 1]
        for val in mol["predictions"].values():
            assert 0.0 <= val <= 1.0, f"Prediction outside [0, 1]: {val}"

        # Deterministic RDKit cheminformatics profile must be present
        chem = mol["cheminformatics"]
        assert chem is not None
        assert chem["identifiers"]["molecular_formula"] == "C9H8O4"
        assert chem["rules"]["lipinski_pass"] is True
        assert chem["boiled_egg"]["hia_absorbed"] is True
        assert "<svg" in chem["svg_2d"]

    def test_ensemble_uncertainty_and_applicability_domain(self, client):
        resp = client.post(
            "/predict/admet",
            json={"smiles": [VALID_TERFENADINE, VALID_ASPIRIN]},
        )
        assert resp.status_code == 200
        body = resp.json()
        if body["model_version"] == "chembl_v37_ensemble_v2":
            terf, asp = body["results"][0], body["results"][1]
            # Epistemic uncertainty & 4-model consensus must be present
            herg_terf = terf["ml_endpoints"]["hERG_inhibition"]
            herg_asp = asp["ml_endpoints"]["hERG_inhibition"]
            assert 0.0 <= herg_terf["epistemic_std"] <= 0.5
            assert len(herg_terf["member_probabilities"]) == 4
            assert terf["applicability_domain"]["max_tanimoto"] >= 0.0
            # Clinical sanity: Terfenadine (withdrawn hERG blocker) > Aspirin
            assert herg_terf["probability"] > herg_asp["probability"]

    def test_multiple_valid_smiles(self, client):
        resp = client.post(
            "/predict/admet",
            json={"smiles": [VALID_ASPIRIN, VALID_CAFFEINE, VALID_KETOCONAZOLE]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["results"]) == 3
        for r in body["results"]:
            assert r["status"] == "success"
            assert len(r["predictions"]) >= 2

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
        assert body["model_version"]
        assert body["prediction_time_ms"] > 0

    def test_predictions_are_finite_probabilities(self, client):
        resp = client.post(
            "/predict/admet",
            json={"smiles": [VALID_ASPIRIN, VALID_CAFFEINE], "include_svg": False},
        )
        for mol in resp.json()["results"]:
            if mol["status"] == "success":
                assert mol["cheminformatics"]["svg_2d"] is None
                for val in mol["predictions"].values():
                    assert isinstance(val, (int, float))
                    assert math.isfinite(val)
                    assert 0.0 <= val <= 1.0

    def test_salt_form_handled(self, client):
        resp = client.post(
            "/predict/admet",
            json={"smiles": ["CC(=O)Oc1ccccc1C(=O)O.[Na+]"]},
        )
        assert resp.status_code == 200
        body = resp.json()
        mol = body["results"][0]
        assert mol["status"] == "success"
        assert "." not in mol["cheminformatics"]["identifiers"]["canonical_smiles"]

"""
ADMET-Engine v3.0 (CardioPhys-3D) — FastAPI v3.0 Endpoint Integration Tests
Tests:
  - POST /api/v3/simulate-cipa (FDA-CiPA 28 lookup including Verapamil rescue,
    Dofetilide high-risk comparison, custom pIC50 mode, waveform downsampling,
    and input validation)
  - POST /api/v3/repair (WP5 closed-loop MMP repair on Terfenadine using the
    loaded ChEMBL v37 ensemble oracle, constraint verification, and validation)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from api.main import app  # noqa: E402

TERFENADINE_SMILES = "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1"


@pytest.fixture(scope="module")
def client():
    """Enter lifespan once per module to load the ChEMBL v37 ensemble."""
    with TestClient(app) as c:
        yield c


class TestSimulateCiPAEndpoint:

    def test_verapamil_lookup_and_multichannel_rescue(self, client):
        """
        Verapamil (IC50_hERG = 0.499 uM -> pIC50 ~ 6.30, IC50_Cav1.2 = 0.202 uM)
        at 4x Cmax (0.324 uM) must be rescued by its Cav1.2 co-block compared to
        an equivalent hERG-only blocker at the exact same concentration.
        """
        resp_ver = client.post(
            "/api/v3/simulate-cipa",
            json={"drug_name": "Verapamil", "cmax_multiple": 4.0, "n_beats": 1},
        )
        assert resp_ver.status_code == 200, resp_ver.text
        ver = resp_ver.json()

        assert ver["mode"] == "cipa28_lookup"
        assert ver["drug_name"] == "Verapamil"
        assert ver["cipa_clinical_risk"] == "Low"
        assert ver["n_beats"] == 1
        assert ver["risk_tier"] in ("Low", "Intermediate", "High")
        assert 0.0 <= ver["tdp_risk_score"] <= 1.0
        assert ver["apd90_ms"] > ver["apd50_ms"] > 0.0
        assert isinstance(ver["has_ead"], bool)
        assert ver["simulation_time_ms"] > 0.0

        # Waveform should be downsampled to ~200 points
        wf = ver["waveform"]
        assert len(wf["t_ms"]) == len(wf["v_mV"])
        assert 150 <= len(wf["t_ms"]) <= 205
        assert wf["t_ms"][0] == 0.0
        assert wf["t_ms"][-1] > 900.0

        # Compare against hERG-only block at Verapamil's hERG pIC50 (6.302) & 4x Cmax (0.324 uM)
        resp_herg_only = client.post(
            "/api/v3/simulate-cipa",
            json={
                "pic50_herg": 6.302,
                "conc_uM": 0.324,
                "n_beats": 1,
            },
        )
        assert resp_herg_only.status_code == 200, resp_herg_only.text
        herg_only = resp_herg_only.json()

        # Multi-channel Cav1.2 rescue: Verapamil has HIGHER qNet & SHORTER APD90 than hERG-only
        assert ver["qnet_uC_uF"] > herg_only["qnet_uC_uF"]
        assert ver["apd90_ms"] < herg_only["apd90_ms"]
        assert ver["tdp_risk_score"] < herg_only["tdp_risk_score"]

    def test_dofetilide_high_risk_vs_verapamil(self, client):
        resp_dof = client.post(
            "/api/v3/simulate-cipa",
            json={"drug_name": "dofetilide", "cmax_multiple": 4.0, "n_beats": 1},
        )
        resp_ver = client.post(
            "/api/v3/simulate-cipa",
            json={"drug_name": "verapamil", "cmax_multiple": 4.0, "n_beats": 1},
        )
        assert resp_dof.status_code == 200
        assert resp_ver.status_code == 200
        dof = resp_dof.json()
        ver = resp_ver.json()

        assert dof["drug_name"] == "Dofetilide"
        assert dof["cipa_clinical_risk"] == "High"
        assert dof["qnet_uC_uF"] < ver["qnet_uC_uF"]
        assert dof["tdp_risk_score"] > ver["tdp_risk_score"]

    def test_custom_pic50_multichannel_simulation(self, client):
        payload = {
            "pic50_herg": 6.8,
            "pic50_cav12": 6.5,
            "pic50_nav15_late": 5.5,
            "conc_uM": 0.5,
            "n_beats": 1,
        }
        resp = client.post("/api/v3/simulate-cipa", json=payload)
        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert body["mode"] == "custom_pic50"
        assert body["drug_name"] is None
        assert body["conc_uM"] == pytest.approx(0.5, rel=1e-4)
        assert math.isfinite(body["qnet_uC_uF"])
        assert 50.0 < body["apd90_ms"] <= 1000.0
        assert 0.0 < body["channel_conductances"]["g_kr"] < 1.0
        assert 0.0 < body["channel_conductances"]["g_cal"] < 1.0
        assert 0.0 < body["channel_conductances"]["g_nal"] < 1.0
        assert len(body["waveform"]["t_ms"]) == 200

    def test_unknown_drug_returns_404(self, client):
        resp = client.post("/api/v3/simulate-cipa", json={"drug_name": "NonExistentDrugXYZ"})
        assert resp.status_code == 404

    def test_empty_payload_returns_422(self, client):
        resp = client.post("/api/v3/simulate-cipa", json={})
        assert resp.status_code == 422


class TestRepairEndpoint:

    def test_terfenadine_closed_loop_repair(self, client):
        resp = client.post(
            "/api/v3/repair",
            json={
                "smiles": TERFENADINE_SMILES,
                "n_rounds": 2,
                "beam_width": 4,
                "min_tanimoto": 0.55,
                "top_k": 5,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert body["oracle"] == "ChEMBLADMETEnsemble.hERG_inhibition"
        assert body["repair_time_ms"] > 0.0
        assert body["total_evaluated"] >= 5

        seed = body["seed"]
        assert seed["risk_score"] > 0.70
        assert 0.0 <= seed["qed"] <= 1.0
        assert 1.0 <= seed["sa_score"] <= 10.0

        cands = body["repaired_candidates"]
        assert 1 <= len(cands) <= 5

        best = cands[0]
        assert best["delta_risk"] < -0.05
        assert best["risk_score"] == pytest.approx(seed["risk_score"] + best["delta_risk"], abs=1e-3)
        assert best["tanimoto"] >= 0.55
        assert best["pains_count"] == 0
        assert best["sa_score"] <= 5.0
        assert len(best["edit_path"]) >= 1
        assert isinstance(best["clogp"], float)
        assert isinstance(best["tpsa"], float)

    def test_invalid_smiles_returns_422(self, client):
        resp = client.post("/api/v3/repair", json={"smiles": "INVALID_NOT_A_MOLECULE_XXX"})
        assert resp.status_code == 422

    def test_out_of_bounds_hyperparams_rejected(self, client):
        resp = client.post(
            "/api/v3/repair",
            json={"smiles": TERFENADINE_SMILES, "n_rounds": 5, "beam_width": 10},
        )
        assert resp.status_code == 422

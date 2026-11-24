"""
ADMET-Engine v3.0 (CardioPhys-3D) — WP1 Tests
Verifies multi-channel Tobit censoring sign conventions, replicate aggregation,
unified Bemis-Murcko scaffold splitting across hERG/Cav1.2/Nav1.5, pH 7.4
physiological protonation, and MMFF94s Boltzmann-weighted 3D conformers.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from data.conformer_3d import (  # noqa: E402
    compute_boltzmann_weights,
    generate_3d_ensemble,
    protonate_at_ph74,
)
from data.multichannel_extractor import (  # noqa: E402
    aggregate_channel_measurements,
    build_multichannel_dataset,
    compute_pic50,
    relation_to_pic50_censor,
)


class TestTobitCensoringAndAggregation:

    def test_relation_sign_inversion_in_pic50_space(self):
        # IC50 = x  -> exact (0)
        assert relation_to_pic50_censor("=") == 0
        # IC50 > 30 uM -> pIC50 < 4.52 -> left-censored (-1)
        assert relation_to_pic50_censor(">") == -1
        assert relation_to_pic50_censor(">=") == -1
        # IC50 < 10 nM -> pIC50 > 8.0 -> right-censored (+1)
        assert relation_to_pic50_censor("<") == +1
        assert relation_to_pic50_censor("<=") == +1
        assert relation_to_pic50_censor("~") is None

    def test_pic50_unit_conversion(self):
        assert math.isclose(compute_pic50(1000.0, "nM"), 6.0, rel_tol=1e-5)
        assert math.isclose(compute_pic50(10.0, "uM"), 5.0, rel_tol=1e-5)
        assert math.isclose(compute_pic50(0.01, "mM"), 5.0, rel_tol=1e-5)
        assert compute_pic50(-5.0, "nM") is None

    def test_replicate_aggregation_prefers_exact(self):
        # Mix of exact and left-censored -> exact median wins
        val, cens, n_obs = aggregate_channel_measurements([(6.2, 0), (6.6, 0), (4.5, -1)])
        assert math.isclose(val, 6.4, rel_tol=1e-5)
        assert cens == 0
        assert n_obs == 3

        # Only left-censored -> left-censored median
        val2, cens2, _ = aggregate_channel_measurements([(4.5, -1), (4.7, -1)])
        assert math.isclose(val2, 4.6, rel_tol=1e-5)
        assert cens2 == -1

    def test_multichannel_builder_and_zero_scaffold_overlap(self):
        # 12 distinct scaffolds across hERG, Cav1.2, Nav1.5 with censored bounds
        scaffolds = [
            "c1ccccc1C(=O)O",
            "c1ccncc1C(=O)N",
            "c1ccc2ccccc2c1O",
            "c1ccc2[nH]ccc2c1C",
            "c1cnc2ccccc2n1",
            "c1ccc(-c2ccccc2)cc1",
            "O=C1CCCCC1",
            "C1CCNCC1",
            "C1CCOCC1",
            "c1ccsc1C(=O)O",
            "c1ccoc1CN",
            "c1ccc2occc2c1",
        ]
        raw = []
        for i, smi in enumerate(scaffolds):
            raw.append({
                "canonical_smiles": smi,
                "target_chembl_id": "CHEMBL240",
                "standard_relation": ">" if i % 3 == 0 else "=",
                "standard_value": 30.0 if i % 3 == 0 else 1.0,
                "standard_units": "uM",
                "pchembl_value": None,
            })
            if i % 2 == 0:
                raw.append({
                    "canonical_smiles": smi,
                    "target_chembl_id": "CHEMBL1940",
                    "standard_relation": "=",
                    "standard_value": 500.0,
                    "standard_units": "nM",
                    "pchembl_value": None,
                })
            if i % 4 == 0:
                raw.append({
                    "canonical_smiles": smi,
                    "target_chembl_id": "CHEMBL1980",
                    "standard_relation": "<",
                    "standard_value": 50.0,
                    "standard_units": "nM",
                    "pchembl_value": None,
                })

        df = build_multichannel_dataset(raw)
        assert len(df) == 12
        assert set(df["split"]) == {"train", "val", "test"}
        assert "hERG_censor" in df.columns
        assert "Cav1.2_pIC50" in df.columns
        assert "Nav1.5_censor" in df.columns

        tr_sc = set(df.loc[df["split"] == "train", "murcko_scaffold"])
        va_sc = set(df.loc[df["split"] == "val", "murcko_scaffold"])
        te_sc = set(df.loc[df["split"] == "test", "murcko_scaffold"])
        assert len(tr_sc & te_sc) == 0
        assert len(tr_sc & va_sc) == 0


class TestPH74And3DConformers:

    def test_ph74_protonation_medicinal_chemistry(self):
        # Aspirin (carboxylic acid, pKa ~3.5) -> net charge -1 at pH 7.4
        asp = protonate_at_ph74(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
        assert Chem.GetFormalCharge(asp) == -1

        # Verapamil (tertiary aliphatic amine, pKa ~8.9) -> net charge +1 at pH 7.4
        verap = protonate_at_ph74(
            Chem.MolFromSmiles("COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC")
        )
        assert Chem.GetFormalCharge(verap) == +1

        # Caffeine (neutral xanthine) -> net charge 0 at pH 7.4
        caf = protonate_at_ph74(Chem.MolFromSmiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C"))
        assert Chem.GetFormalCharge(caf) == 0

    def test_boltzmann_weights_valid_distribution(self):
        w = compute_boltzmann_weights(np.array([10.0, 10.616, 12.0], dtype=np.float32))
        assert math.isclose(float(w.sum()), 1.0, rel_tol=1e-5)
        assert w[0] > w[1] > w[2] > 0.0

    def test_3d_conformer_ensemble_generation(self):
        ens = generate_3d_ensemble(
            "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC",
            num_confs=3,
            protonate=True,
        )
        assert ens is not None
        assert ens.net_charge_ph74 == +1
        assert ens.positions.ndim == 3
        assert ens.positions.shape[0] >= 1
        assert ens.positions.shape[2] == 3
        assert np.isfinite(ens.positions).all()
        assert math.isclose(float(ens.boltzmann_weights.sum()), 1.0, rel_tol=1e-5)
        td = ens.to_torch_dict()
        assert td["pos"].shape == ens.positions.shape

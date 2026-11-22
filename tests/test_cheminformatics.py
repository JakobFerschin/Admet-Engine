"""
ADMET-Engine — Cheminformatics, DMPNN Edge-Case & Scaffold Split Tests
Verifies scientific accuracy of the medicinal chemistry module and
regression-tests the critical DMPNN & standardization fixes.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from cheminformatics import (  # noqa: E402
    analyze_molecule,
    evaluate_boiled_egg,
    get_murcko_scaffold,
    screen_structural_alerts,
)
from data_pipeline import standardize_molecule  # noqa: E402
from featurizer import RICH_EDGE_DIM, RICH_NODE_DIM, smiles_to_data  # noqa: E402
from models.dmpnn import ADMETPredictor, DMPNNEncoder  # noqa: E402


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
DIAZEPAM = "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"
TERFENADINE = "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1"
CURCUMIN = "COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O"
SUCROSE = "OC[C@H]1O[C@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O"


class TestCheminformaticsAccuracy:
    """Validate deterministic RDKit descriptors against textbook medicinal chemistry."""

    def test_aspirin_profile_and_clean_alerts(self):
        mol = standardize_molecule(ASPIRIN)
        profile = analyze_molecule(mol, include_svg=True)

        assert profile["identifiers"]["molecular_formula"] == "C9H8O4"
        assert 179.0 < profile["physchem"]["mw"] < 181.0
        assert profile["rules"]["lipinski_pass"] is True
        assert profile["rules"]["lipinski_violations"] == 0
        assert profile["rules"]["veber_pass"] is True
        assert profile["rules"]["pfizer_3_75_warning"] is False
        assert profile["alerts"]["pains_count"] == 0
        assert profile["svg_2d"] is not None
        assert "<svg" in profile["svg_2d"]

    def test_boiled_egg_cns_vs_non_permeant(self):
        # Diazepam is a classic CNS-active benzodiazepine -> Yolk (BBB+)
        diazepam_mol = standardize_molecule(DIAZEPAM)
        d_prof = analyze_molecule(diazepam_mol, include_svg=False)
        assert d_prof["boiled_egg"]["bbb_permeant"] is True
        assert d_prof["boiled_egg"]["hia_absorbed"] is True
        assert d_prof["boiled_egg"]["region"] == "yolk"

        # Sucrose (TPSA ~189 A^2, WLOGP ~-5.4) -> Outside passive absorption
        sucrose_mol = standardize_molecule(SUCROSE)
        s_prof = analyze_molecule(sucrose_mol, include_svg=False)
        assert s_prof["boiled_egg"]["bbb_permeant"] is False
        assert s_prof["boiled_egg"]["hia_absorbed"] is False
        assert s_prof["boiled_egg"]["region"] == "outside"

    def test_pfizer_3_75_lipophilic_promiscuity_flag(self):
        # Terfenadine: high cLogP (~6.5) and low TPSA (~43.7 A^2) -> Pfizer 3/75 warning
        mol = standardize_molecule(TERFENADINE)
        prof = analyze_molecule(mol, include_svg=False)
        assert prof["physchem"]["clogp"] > 3.0
        assert prof["physchem"]["tpsa"] < 75.0
        assert prof["rules"]["pfizer_3_75_warning"] is True

    def test_pains_and_brenk_detection(self):
        # Curcumin contains Michael acceptor / diketone alerts (Brenk) and/or PAINS
        mol = standardize_molecule(CURCUMIN)
        alerts = screen_structural_alerts(mol)
        assert alerts["total_alerts"] > 0
        assert len(alerts["highlight_atoms"]) > 0

    def test_counterion_fragment_stripping(self):
        # Bromide counterion prefix like in raw ChEMBL exports must be stripped
        mol = standardize_molecule("Br.CC(=O)Oc1ccccc1C(=O)O")
        prof = analyze_molecule(mol, include_svg=False)
        assert "." not in prof["identifiers"]["canonical_smiles"]
        assert "Br" not in prof["identifiers"]["molecular_formula"]

    def test_murcko_scaffold_extraction(self):
        mol = standardize_molecule(ASPIRIN)
        assert get_murcko_scaffold(mol) == "c1ccccc1"


class TestDMPNNRegressionFixes:
    """Verify fixes for O(N^2) reverse map, arbitrary edge ordering, and single-atom graphs."""

    def test_single_atom_molecule_no_crash(self):
        data = smiles_to_data("[Ar]", rich=True)
        assert data is not None
        assert data.x.shape == (1, RICH_NODE_DIM)
        assert data.edge_index.shape == (2, 0)
        assert data.edge_attr.shape == (0, RICH_EDGE_DIM)

        model = ADMETPredictor(
            node_dim=RICH_NODE_DIM,
            edge_dim=RICH_EDGE_DIM,
            hidden_dim=32,
            num_tasks=2,
            T=3,
            set2set_steps=2,
            dropout=0.0,
        )
        batch = Batch.from_data_list([data])
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        assert out.shape == (1, 2)
        assert torch.isfinite(out).all()

    def test_reverse_map_fast_and_shuffled_edge_order(self):
        enc = DMPNNEncoder(node_dim=9, edge_dim=4, hidden_dim=16, T=2)
        # Paired edges: (0,1), (1,0), (1,2), (2,1)
        ei_paired = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        rev_fast = enc._build_reverse_map(ei_paired)
        assert rev_fast.tolist() == [1, 0, 3, 2]

        # Shuffled arbitrary ordering: (0,1), (1,2), (1,0), (2,1)
        ei_shuffled = torch.tensor([[0, 1, 1, 2], [1, 2, 0, 1]], dtype=torch.long)
        rev_general = enc._build_reverse_map(ei_shuffled)
        assert rev_general.tolist() == [2, 3, 0, 1]

    def test_reverse_map_missing_reverse_raises(self):
        enc = DMPNNEncoder(node_dim=9, edge_dim=4, hidden_dim=16, T=2)
        ei_broken = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        with pytest.raises(RuntimeError, match="both directions"):
            enc._build_reverse_map(ei_broken)

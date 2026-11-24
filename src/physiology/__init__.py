"""
ADMET-Engine v3.0 (CardioPhys-3D) — Mechanistic Cardiac Electrophysiology Package.

Implements the FDA-CiPA aligned ventricular cardiomyocyte action potential
simulator (differentiable PyTorch Rush-Larsen integrator) and the official
28-drug CiPA Torsades de Pointes (TdP) validation benchmark.
"""

from physiology.cipa28 import CIPA_28_DRUGS, get_cipa28_dataframe
from physiology.ohara_rudy import (
    CardiomyocyteSimulator,
    ChannelBlockProfile,
    SimulationResult,
    hill_conductance_scale,
)

__all__ = [
    "CIPA_28_DRUGS",
    "get_cipa28_dataframe",
    "CardiomyocyteSimulator",
    "ChannelBlockProfile",
    "SimulationResult",
    "hill_conductance_scale",
]

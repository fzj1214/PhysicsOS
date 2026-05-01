from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SolverBackendInfo:
    name: str
    tier: int
    domains: tuple[str, ...]
    command: str | None
    python_integration: str
    role: str


DEFAULT_BACKENDS: tuple[SolverBackendInfo, ...] = (
    SolverBackendInfo("openfoam", 1, ("fluid", "thermal", "multiphysics"), "simpleFoam", "input/parser adapter", "heavy-duty CFD"),
    SolverBackendInfo("su2", 1, ("fluid", "thermal"), "SU2_CFD", "Python scripts/API", "lightweight CFD and adjoint design"),
    SolverBackendInfo("fenicsx", 1, ("solid", "thermal", "electromagnetic", "custom", "multiphysics"), None, "Python-first", "programmable FEM/PDE"),
    SolverBackendInfo("mfem", 1, ("solid", "thermal", "electromagnetic", "custom"), None, "PyMFEM", "high-order/HPC FEM"),
    SolverBackendInfo("quantum_espresso", 1, ("quantum",), "pw.x", "ASE/input parser", "periodic DFT"),
    SolverBackendInfo("cp2k", 1, ("quantum", "molecular"), "cp2k", "ASE/input parser", "DFT/MD/QM-MM"),
    SolverBackendInfo("lammps", 1, ("molecular",), "lmp", "Python module/ASE", "classical MD"),
    SolverBackendInfo("cantera", 1, ("thermal", "fluid", "custom"), None, "Python-first", "kinetics and reacting systems"),
)


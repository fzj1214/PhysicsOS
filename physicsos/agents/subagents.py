from __future__ import annotations

from physicsos.agents.prompts import (
    ANALYSIS_FILE_AGENT_PROMPT,
    GEOMETRY_EMBEDDING_AGENT_PROMPT,
    KS_DFT_ANALYSIS_AGENT_PROMPT,
    KS_DFT_TAPS_DERIVATION_AGENT_PROMPT,
    KS_DFT_TAPS_IMPLEMENTATION_AGENT_PROMPT,
    KS_DFT_VERIFICATION_AGENT_PROMPT,
    MATERIALS_PREPROCESS_AGENT_PROMPT,
    POSTPROCESS_AGENT_PROMPT,
    TAPS_DERIVATION_AGENT_PROMPT,
    TAPS_IMPLEMENTATION_AGENT_PROMPT,
    VERIFICATION_AGENT_PROMPT,
)
from physicsos.tools.registry import (
    DEEPAGENTS_SUBAGENT_TOOL_GROUPS,
)


SUBAGENTS: list[dict] = [
    {
        "name": "analysis-file-agent",
        "description": (
            "Extracts PDEs, fields, geometry sources, boundary/initial conditions, "
            "parameters, targets, and missing inputs from user analysis files."
        ),
        "system_prompt": ANALYSIS_FILE_AGENT_PROMPT,
        "tools": DEEPAGENTS_SUBAGENT_TOOL_GROUPS["analysis-file-agent"],
    },
    {
        "name": "geometry-embedding-agent",
        "description": (
            "Imports STL/CAD and prepares Gmsh/SDF/voxel/background-grid geometry "
            "embeddings for immersed-boundary or IFE TAPS."
        ),
        "system_prompt": GEOMETRY_EMBEDDING_AGENT_PROMPT,
        "tools": DEEPAGENTS_SUBAGENT_TOOL_GROUPS["geometry-embedding-agent"],
    },
    {
        "name": "materials-preprocess-agent",
        "description": (
            "Uses pymatgen, spglib, and seekpath tools to create deterministic "
            "materials artifacts for KS-DFT-TAPS."
        ),
        "system_prompt": MATERIALS_PREPROCESS_AGENT_PROMPT,
        "tools": DEEPAGENTS_SUBAGENT_TOOL_GROUPS["materials-preprocess-agent"],
    },
    {
        "name": "ks-dft-analysis-agent",
        "description": "Turns materials artifacts into a KS-DFT-TAPS problem statement and assumptions file.",
        "system_prompt": KS_DFT_ANALYSIS_AGENT_PROMPT,
        "tools": DEEPAGENTS_SUBAGENT_TOOL_GROUPS["ks-dft-analysis-agent"],
    },
    {
        "name": "ks-dft-taps-derivation-agent",
        "description": "Derives the case-local Kohn-Sham TAPS matrix, tensor-basis, SCF, and convergence formulation.",
        "system_prompt": KS_DFT_TAPS_DERIVATION_AGENT_PROMPT,
        "tools": DEEPAGENTS_SUBAGENT_TOOL_GROUPS["ks-dft-taps-derivation-agent"],
    },
    {
        "name": "ks-dft-taps-implementation-agent",
        "description": "Implements the case-local KS-DFT-TAPS kernel without external DFT engines.",
        "system_prompt": KS_DFT_TAPS_IMPLEMENTATION_AGENT_PROMPT,
        "tools": DEEPAGENTS_SUBAGENT_TOOL_GROUPS["ks-dft-taps-implementation-agent"],
    },
    {
        "name": "ks-dft-verification-agent",
        "description": "Checks KS-DFT-TAPS charge, orthonormality, SCF, Poisson, convergence, and material artifact usage evidence.",
        "system_prompt": KS_DFT_VERIFICATION_AGENT_PROMPT,
        "tools": DEEPAGENTS_SUBAGENT_TOOL_GROUPS["ks-dft-verification-agent"],
    },
    {
        "name": "taps-derivation-agent",
        "description": (
            "Creates paper-style TAPS derivation prompts, derivation Markdown, "
            "implementation notes, and geometry-coupled subspace iteration derivations."
        ),
        "system_prompt": TAPS_DERIVATION_AGENT_PROMPT,
        "tools": DEEPAGENTS_SUBAGENT_TOOL_GROUPS["taps-derivation-agent"],
    },
    {
        "name": "taps-implementation-agent",
        "description": (
            "Turns TAPS derivations and geometry embeddings into case-local TAPS "
            "kernels and execution plans."
        ),
        "system_prompt": TAPS_IMPLEMENTATION_AGENT_PROMPT,
        "tools": DEEPAGENTS_SUBAGENT_TOOL_GROUPS["taps-implementation-agent"],
    },
    {
        "name": "verification-agent",
        "description": (
            "Runs the paper-style verification chain: exact solution code, "
            "convergence study code, execution, plots, and verification reports."
        ),
        "system_prompt": VERIFICATION_AGENT_PROMPT,
        "tools": DEEPAGENTS_SUBAGENT_TOOL_GROUPS["verification-agent"],
    },
    {
        "name": "postprocess-agent",
        "description": "Writes final TAPS figures, engineering report, and summary artifacts.",
        "system_prompt": POSTPROCESS_AGENT_PROMPT,
        "tools": DEEPAGENTS_SUBAGENT_TOOL_GROUPS["postprocess-agent"],
    },
    {
        "name": "knowledge-agent",
        "description": "Retrieves local TAPS references, matrix definitions, verification patterns, and source-grounded notes.",
        "system_prompt": "Use the local knowledge/reference tools to support the paper-style TAPS workflow. Do not invent citations.",
        "tools": DEEPAGENTS_SUBAGENT_TOOL_GROUPS["knowledge-agent"],
    },
]

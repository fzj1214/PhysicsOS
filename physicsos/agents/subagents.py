from __future__ import annotations

from physicsos.agents.prompts import (
    ANALYSIS_FILE_AGENT_PROMPT,
    GEOMETRY_EMBEDDING_AGENT_PROMPT,
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

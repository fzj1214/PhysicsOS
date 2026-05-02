from __future__ import annotations

from physicsos.agents.prompts import (
    GEOMETRY_MESH_AGENT_PROMPT,
    KNOWLEDGE_AGENT_PROMPT,
    POSTPROCESS_AGENT_PROMPT,
    SOLVER_AGENT_PROMPT,
    TAPS_AGENT_PROMPT,
    VERIFICATION_AGENT_PROMPT,
)
from physicsos.tools.registry import SUBAGENT_TOOL_GROUPS


SUBAGENTS = [
    {
        "name": "geometry-mesh-agent",
        "description": "Build GeometrySpec and MeshSpec from CAD/STL/STEP/CIF/POSCAR/molecular/text inputs.",
        "system_prompt": GEOMETRY_MESH_AGENT_PROMPT,
        "tools": SUBAGENT_TOOL_GROUPS["geometry-mesh-agent"],
    },
    {
        "name": "taps-agent",
        "description": "Compile and run TAPS-first equation-driven surrogate solves for explicit parameterized PDEs.",
        "system_prompt": TAPS_AGENT_PROMPT,
        "tools": SUBAGENT_TOOL_GROUPS["taps-agent"],
    },
    {
        "name": "solver-agent",
        "description": "Select and run non-TAPS surrogate, full, or hybrid fallback solver backends for a validated PhysicsProblem.",
        "system_prompt": SOLVER_AGENT_PROMPT,
        "tools": SUBAGENT_TOOL_GROUPS["solver-agent"],
    },
    {
        "name": "verification-agent",
        "description": "Check residuals, uncertainty, conservation, OOD risk, and recommended next actions.",
        "system_prompt": VERIFICATION_AGENT_PROMPT,
        "tools": SUBAGENT_TOOL_GROUPS["verification-agent"],
    },
    {
        "name": "postprocess-agent",
        "description": "Extract KPIs, visualizations, reports, and optimization suggestions from solver results.",
        "system_prompt": POSTPROCESS_AGENT_PROMPT,
        "tools": SUBAGENT_TOOL_GROUPS["postprocess-agent"],
    },
    {
        "name": "knowledge-agent",
        "description": "Retrieve local knowledge, arXiv papers, DeepSearch reports, prior cases, materials, operator templates, validation rules, and paper notes.",
        "system_prompt": KNOWLEDGE_AGENT_PROMPT,
        "tools": SUBAGENT_TOOL_GROUPS["knowledge-agent"],
    },
]

export const CLAUDE_MD = `
# SciMate CFD Workspace

You are an expert CFD engineer. Execute the user's task completely using the tools available.

## Available Skills

Two skill packs are pre-installed in \`.claude/skills/\`:

- **hpc-openfoam** - OpenFOAM case setup, meshing, BCs, numerics, and error recovery
- **hpc-paraview** - ParaView post-processing for tasks that truly require ParaView

**Load a skill only when you actually need it for the task:**
- Before writing or editing OpenFOAM files -> read \`.claude/skills/hpc-openfoam/SKILL.md\`
- Before using ParaView-specific filters or a \`pvbatch\` script -> read \`.claude/skills/hpc-paraview/SKILL.md\`

Each SKILL.md will tell you which reference files to load next. Load only the references relevant to your current step.

## Environment

- OpenFOAM is pre-sourced. If a solver is not found: \`source /opt/openfoam*/etc/bashrc\`
- Gmsh binary: \`gmsh\`
- Post-processing default: use Python with \`matplotlib\` for plots, contours, line charts, residual charts, and simple field visualizations
- Use \`pvbatch\` only when ParaView-specific capabilities are clearly necessary, such as advanced VTK/VTU processing, volumetric streamline generation, or filters that are impractical to reproduce in Python
- Do not choose \`pvbatch\` when a direct Python script with \`matplotlib\` can produce the requested PNG faster and more simply
- Working directory: \`/workspace\`
- Output directory: \`/workspace/output\` (create if missing)

## Output Requirements

- Save result images to \`/workspace/output/\` as PNG
- Save logs, CSV, and force data to \`/workspace/output/\`
- Write \`/workspace/output/summary.txt\` with a brief results summary
`.trim()

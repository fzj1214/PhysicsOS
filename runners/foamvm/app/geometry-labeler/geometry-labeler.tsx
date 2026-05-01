'use client'

import { useMemo, useState } from 'react'

type BoundaryKind = 'inlet' | 'outlet' | 'wall' | 'symmetry' | 'periodic' | 'interface' | 'farfield' | 'surface' | 'custom'

interface SelectableGroup {
  id: string
  name: string
  dimension?: number
  edge_ids?: number[]
  face_ids?: number[]
  node_ids?: number[]
}

interface ViewerGeometry {
  points?: number[][]
  edges?: number[][]
  faces?: number[][]
}

interface BoundaryLabelingArtifact {
  schema_version: string
  geometry_id?: string
  viewer_geometry?: ViewerGeometry
  selectable_groups?: SelectableGroup[]
  suggested_boundary_labels?: Array<{
    target_ids: string[]
    boundary_id: string
    label: string
    kind: BoundaryKind
    confidence: number
    reason?: string
    requires_confirmation: boolean
  }>
  confirmed_boundary_labels?: Array<{
    target_ids: string[]
    boundary_id: string
    label: string
    kind: BoundaryKind
    confidence: number
    confirmed_by: string
  }>
}

interface ScreenPoint {
  x: number
  y: number
  z: number
}

const SAMPLE_ARTIFACT = {
  schema_version: 'physicsos.boundary_labeling.v1',
  geometry_id: 'geometry:demo_box',
  viewer_geometry: {
    points: [
      [0, 0, 0],
      [1, 0, 0],
      [1, 1, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 0, 1],
      [1, 1, 1],
      [0, 1, 1],
    ],
    edges: [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 0],
      [4, 5],
      [5, 6],
      [6, 7],
      [7, 4],
      [0, 4],
      [1, 5],
      [2, 6],
      [3, 7],
    ],
    faces: [
      [0, 3, 7, 4],
      [1, 2, 6, 5],
      [0, 1, 5, 4],
      [3, 2, 6, 7],
    ],
  },
  selectable_groups: [
    { id: 'mesh_graph:physical:1', name: 'inlet', dimension: 2, face_ids: [0], node_ids: [0, 3, 7, 4] },
    { id: 'mesh_graph:physical:2', name: 'outlet', dimension: 2, face_ids: [1], node_ids: [1, 2, 6, 5] },
    { id: 'mesh_graph:physical:3', name: 'wall', dimension: 2, face_ids: [2, 3], node_ids: [0, 1, 3, 4, 5, 6, 7] },
  ],
  suggested_boundary_labels: [
    {
      target_ids: ['mesh_graph:physical:1'],
      boundary_id: 'boundary:inlet',
      label: 'inlet',
      kind: 'inlet',
      confidence: 0.65,
      reason: 'Existing group name contains a common boundary-condition keyword.',
      requires_confirmation: true,
    },
  ],
  confirmed_boundary_labels: [],
}

function parseArtifact(text: string): { artifact: BoundaryLabelingArtifact | null; error: string | null } {
  try {
    const parsed = JSON.parse(text) as BoundaryLabelingArtifact
    if (parsed.schema_version !== 'physicsos.boundary_labeling.v1') {
      return { artifact: null, error: 'Expected schema_version=physicsos.boundary_labeling.v1' }
    }
    return { artifact: parsed, error: null }
  } catch (error) {
    return { artifact: null, error: (error as Error).message }
  }
}

function boundaryKindFromName(name: string): BoundaryKind {
  const lowered = name.toLowerCase()
  if (lowered.includes('inlet')) return 'inlet'
  if (lowered.includes('outlet')) return 'outlet'
  if (lowered.includes('wall')) return 'wall'
  if (lowered.includes('symmetry')) return 'symmetry'
  if (lowered.includes('periodic')) return 'periodic'
  if (lowered.includes('farfield')) return 'farfield'
  return 'surface'
}

function projectPoints(points: number[][], yaw: number, pitch: number, size: number): ScreenPoint[] {
  if (!points.length) return []
  const min = [0, 1, 2].map((axis) => Math.min(...points.map((point) => point[axis] ?? 0)))
  const max = [0, 1, 2].map((axis) => Math.max(...points.map((point) => point[axis] ?? 0)))
  const center = min.map((value, axis) => (value + max[axis]) / 2)
  const span = Math.max(...max.map((value, axis) => value - min[axis]), 1e-6)
  const cy = Math.cos(yaw)
  const sy = Math.sin(yaw)
  const cp = Math.cos(pitch)
  const sp = Math.sin(pitch)

  return points.map((point) => {
    const x0 = ((point[0] ?? 0) - center[0]) / span
    const y0 = ((point[1] ?? 0) - center[1]) / span
    const z0 = ((point[2] ?? 0) - center[2]) / span
    const x1 = cy * x0 + sy * z0
    const z1 = -sy * x0 + cy * z0
    const y1 = cp * y0 - sp * z1
    const z2 = sp * y0 + cp * z1
    const scale = size * 0.66
    return {
      x: size / 2 + x1 * scale,
      y: size / 2 - y1 * scale,
      z: z2,
    }
  })
}

function GeometryPreview({
  artifact,
  selectedGroupId,
  onSelectGroup,
}: {
  artifact: BoundaryLabelingArtifact
  selectedGroupId: string | null
  onSelectGroup: (id: string) => void
}) {
  const [yaw, setYaw] = useState(-0.65)
  const [pitch, setPitch] = useState(0.45)
  const size = 560
  const points = artifact.viewer_geometry?.points ?? []
  const faces = artifact.viewer_geometry?.faces ?? []
  const edges = artifact.viewer_geometry?.edges ?? []
  const groups = artifact.selectable_groups ?? []
  const projected = useMemo(() => projectPoints(points, yaw, pitch, size), [pitch, points, yaw])
  const groupByFace = new Map<number, SelectableGroup>()

  for (const group of groups) {
    for (const faceId of group.face_ids ?? []) {
      groupByFace.set(faceId, group)
    }
  }

  const orderedFaces = faces
    .map((face, index) => ({
      face,
      index,
      depth: face.reduce((sum, pointIndex) => sum + (projected[pointIndex]?.z ?? 0), 0) / Math.max(face.length, 1),
      group: groupByFace.get(index),
    }))
    .sort((a, b) => a.depth - b.depth)

  return (
    <div className="rounded-[32px] border border-white/10 bg-slate-950/60 p-4 shadow-2xl shadow-cyan-950/20">
      <svg className="h-auto w-full rounded-[24px] bg-[radial-gradient(circle_at_top,_rgba(34,211,238,0.16),_transparent_45%),#020617]" viewBox={`0 0 ${size} ${size}`}>
        {orderedFaces.map(({ face, index, group }) => {
          const polygon = face.map((pointIndex) => `${projected[pointIndex]?.x ?? 0},${projected[pointIndex]?.y ?? 0}`).join(' ')
          const selected = group?.id === selectedGroupId
          return (
            <polygon
              className="cursor-pointer transition"
              fill={selected ? 'rgba(34,211,238,0.42)' : group ? 'rgba(148,163,184,0.18)' : 'rgba(71,85,105,0.12)'}
              key={`face-${index}`}
              onClick={() => group ? onSelectGroup(group.id) : undefined}
              points={polygon}
              stroke={selected ? 'rgb(103,232,249)' : 'rgba(255,255,255,0.16)'}
              strokeWidth={selected ? 3 : 1}
            />
          )
        })}
        {edges.map((edge, index) => {
          const a = projected[edge[0]]
          const b = projected[edge[1]]
          if (!a || !b) return null
          return <line key={`edge-${index}`} stroke="rgba(226,232,240,0.34)" strokeWidth="1.2" x1={a.x} x2={b.x} y1={a.y} y2={b.y} />
        })}
      </svg>
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <label className="text-xs uppercase tracking-[0.24em] text-slate-400">
          Yaw
          <input className="mt-2 w-full accent-cyan-300" max="3.14" min="-3.14" onChange={(event) => setYaw(Number(event.target.value))} step="0.01" type="range" value={yaw} />
        </label>
        <label className="text-xs uppercase tracking-[0.24em] text-slate-400">
          Pitch
          <input className="mt-2 w-full accent-cyan-300" max="1.4" min="-1.4" onChange={(event) => setPitch(Number(event.target.value))} step="0.01" type="range" value={pitch} />
        </label>
      </div>
    </div>
  )
}

export function GeometryLabeler() {
  const [text, setText] = useState(JSON.stringify(SAMPLE_ARTIFACT, null, 2))
  const [{ artifact, error }, setParsed] = useState(parseArtifact(JSON.stringify(SAMPLE_ARTIFACT)))
  const [selectedGroupId, setSelectedGroupId] = useState<string | null>(SAMPLE_ARTIFACT.selectable_groups[0].id)
  const [label, setLabel] = useState('inlet')
  const [kind, setKind] = useState<BoundaryKind>('inlet')
  const [confirmed, setConfirmed] = useState<NonNullable<BoundaryLabelingArtifact['confirmed_boundary_labels']>>([])

  const groups = artifact?.selectable_groups ?? []
  const selectedGroup = groups.find((group) => group.id === selectedGroupId) ?? null
  const outputArtifact = artifact ? { ...artifact, confirmed_boundary_labels: confirmed } : null

  const handleParse = () => {
    const result = parseArtifact(text)
    setParsed(result)
    const firstGroup = result.artifact?.selectable_groups?.[0]
    setSelectedGroupId(firstGroup?.id ?? null)
    setLabel(firstGroup?.name ?? 'boundary')
    setKind(boundaryKindFromName(firstGroup?.name ?? 'boundary'))
    setConfirmed(result.artifact?.confirmed_boundary_labels ?? [])
  }

  const selectGroup = (id: string) => {
    const group = groups.find((item) => item.id === id)
    setSelectedGroupId(id)
    setLabel(group?.name ?? 'boundary')
    setKind(boundaryKindFromName(group?.name ?? 'boundary'))
  }

  const confirmSelected = () => {
    if (!selectedGroup) return
    const next = {
      target_ids: [selectedGroup.id],
      boundary_id: `boundary:${label || selectedGroup.name}`,
      label: label || selectedGroup.name,
      kind,
      confidence: 1,
      confirmed_by: 'user',
    }
    setConfirmed((previous) => [...previous.filter((item) => !item.target_ids.includes(selectedGroup.id)), next])
  }

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(34,211,238,0.18),_transparent_30%),linear-gradient(180deg,_#020617,_#0f172a_52%,_#020617)] px-4 py-8 text-slate-100 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl">
        <header className="mb-6 rounded-[30px] border border-white/10 bg-white/[0.04] p-6">
          <p className="text-xs uppercase tracking-[0.32em] text-cyan-200/70">PhysicsOS Cloud</p>
          <h1 className="mt-3 text-4xl font-semibold tracking-tight text-white">Geometry Boundary Labeler</h1>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-300">
            Paste a <code className="rounded bg-white/10 px-1">physicsos.boundary_labeling.v1</code> artifact, rotate the mesh preview, select face/edge groups, and export confirmed labels.
            Weak suggestions are never applied until they are explicitly confirmed here.
          </p>
        </header>

        <div className="grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
          <section className="rounded-[30px] border border-white/10 bg-white/[0.04] p-5">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-medium text-white">Artifact JSON</h2>
              <button className="rounded-full bg-cyan-300 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-cyan-200" onClick={handleParse} type="button">
                Load artifact
              </button>
            </div>
            <textarea
              className="h-[560px] w-full resize-none rounded-[24px] border border-white/10 bg-slate-950/70 p-4 font-mono text-xs leading-5 text-slate-100 outline-none focus:border-cyan-300/60"
              onChange={(event) => setText(event.target.value)}
              value={text}
            />
            {error ? <p className="mt-3 rounded-2xl border border-rose-300/20 bg-rose-300/10 px-3 py-2 text-sm text-rose-100">{error}</p> : null}
          </section>

          <section className="space-y-5">
            {artifact ? <GeometryPreview artifact={artifact} onSelectGroup={selectGroup} selectedGroupId={selectedGroupId} /> : null}

            <div className="grid gap-5 lg:grid-cols-2">
              <div className="rounded-[30px] border border-white/10 bg-white/[0.04] p-5">
                <h2 className="text-lg font-medium text-white">Selectable Groups</h2>
                <div className="mt-4 max-h-72 space-y-2 overflow-auto pr-1">
                  {groups.map((group) => (
                    <button
                      className={`w-full rounded-2xl border px-3 py-2 text-left text-sm transition ${group.id === selectedGroupId ? 'border-cyan-300/60 bg-cyan-300/10 text-cyan-50' : 'border-white/10 bg-black/20 text-slate-300 hover:border-white/25'}`}
                      key={group.id}
                      onClick={() => selectGroup(group.id)}
                      type="button"
                    >
                      <div className="font-medium">{group.name}</div>
                      <div className="mt-1 text-xs text-slate-500">{group.id}</div>
                      <div className="mt-1 text-xs text-slate-400">faces {group.face_ids?.length ?? 0} · edges {group.edge_ids?.length ?? 0}</div>
                    </button>
                  ))}
                </div>
              </div>

              <div className="rounded-[30px] border border-white/10 bg-white/[0.04] p-5">
                <h2 className="text-lg font-medium text-white">Confirm Label</h2>
                <label className="mt-4 block text-xs uppercase tracking-[0.24em] text-slate-400">
                  Label
                  <input className="mt-2 w-full rounded-2xl border border-white/10 bg-slate-950/70 px-3 py-2 text-sm text-white outline-none focus:border-cyan-300/60" onChange={(event) => setLabel(event.target.value)} value={label} />
                </label>
                <label className="mt-4 block text-xs uppercase tracking-[0.24em] text-slate-400">
                  Boundary kind
                  <select className="mt-2 w-full rounded-2xl border border-white/10 bg-slate-950/70 px-3 py-2 text-sm text-white outline-none focus:border-cyan-300/60" onChange={(event) => setKind(event.target.value as BoundaryKind)} value={kind}>
                    {['inlet', 'outlet', 'wall', 'symmetry', 'periodic', 'interface', 'farfield', 'surface', 'custom'].map((item) => (
                      <option key={item} value={item}>{item}</option>
                    ))}
                  </select>
                </label>
                <button className="mt-4 w-full rounded-full bg-emerald-300 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-emerald-200" disabled={!selectedGroup} onClick={confirmSelected} type="button">
                  Confirm selected group
                </button>
                <div className="mt-4 rounded-2xl border border-white/10 bg-black/20 p-3 text-xs text-slate-300">
                  Confirmed labels: <span className="font-semibold text-white">{confirmed.length}</span>
                </div>
              </div>
            </div>

            <div className="rounded-[30px] border border-white/10 bg-white/[0.04] p-5">
              <h2 className="text-lg font-medium text-white">Confirmed Artifact Output</h2>
              <pre className="mt-4 max-h-96 overflow-auto rounded-[24px] border border-white/10 bg-slate-950/80 p-4 text-xs leading-5 text-slate-200">
                {JSON.stringify(outputArtifact, null, 2)}
              </pre>
            </div>
          </section>
        </div>
      </div>
    </main>
  )
}

import { CommandExitError, FileType, Sandbox } from 'e2b'
import { z } from 'zod'

const ManifestSchema = z.object({
  schema_version: z.string(),
  problem_id: z.string(),
  backend: z.string(),
  backend_command: z.string().nullable().optional(),
  service: z
    .object({
      base_url: z.string().nullable().optional(),
      mode: z.string().optional(),
      requires_approval_token: z.boolean().optional(),
    })
    .optional(),
  budget: z.record(z.string(), z.unknown()).optional(),
  inputs: z.record(z.string(), z.unknown()).optional(),
  execution_policy: z
    .object({
      sandboxed_workspace: z.string().optional(),
      artifact_collection: z.array(z.string()).optional(),
    })
    .optional(),
  openfoam: z
    .object({
      solver: z.string().optional(),
      case_files: z
        .array(
          z.object({
            path: z.string(),
            content: z.string().optional(),
            content_base64: z.string().optional(),
          }),
        )
        .optional(),
      run_commands: z.array(z.string()).optional(),
    })
    .optional(),
})

export type PhysicsOSRunnerManifest = z.infer<typeof ManifestSchema>

export interface PhysicsOSOpenFOAMRunResult {
  job_id: string
  status: 'completed' | 'failed'
  sandbox_id: string
  backend: string
  problem_id: string
  command: string
  stdout_tail: string
  stderr_tail: string
  artifacts: Array<{
    name: string
    path: string
    size: number
    content_base64?: string
  }>
}

function sandboxApiOpts() {
  return {
    apiKey: process.env.E2B_API_KEY,
  }
}

async function runCommand(sandbox: Sandbox, cmd: string, timeoutMs: number) {
  try {
    return await sandbox.commands.run(cmd, { timeoutMs })
  } catch (error) {
    if (error instanceof CommandExitError) {
      return {
        stdout: error.stdout || '',
        stderr: error.stderr || '',
        exitCode: error.exitCode,
      }
    }
    throw error
  }
}

async function writeCaseFiles(sandbox: Sandbox, files: NonNullable<PhysicsOSRunnerManifest['openfoam']>['case_files']) {
  if (!files?.length) return
  await sandbox.commands.run('rm -rf /workspace/case && mkdir -p /workspace/case', { timeoutMs: 10000 })
  for (const file of files) {
    if (!file.path || file.path.includes('..') || file.path.startsWith('/')) {
      throw new Error(`Unsafe OpenFOAM case file path: ${file.path}`)
    }
    const target = `/workspace/case/${file.path}`
    await sandbox.commands.run(`mkdir -p '${target.split('/').slice(0, -1).join('/')}'`, { timeoutMs: 10000 })
    const content = file.content_base64 ? Buffer.from(file.content_base64, 'base64').toString('utf-8') : file.content || ''
    await sandbox.files.write(target, content)
  }
}

function defaultOpenFOAMCommand(manifest: PhysicsOSRunnerManifest) {
  const hasCaseFiles = Boolean(manifest.openfoam?.case_files?.length)
  const solver = manifest.openfoam?.solver || (hasCaseFiles ? manifest.backend_command : 'icoFoam') || 'icoFoam'
  const userCommands = manifest.openfoam?.run_commands
  if (userCommands?.length) {
    return userCommands.join(' && ')
  }
  return [
    'set -e',
    'source /opt/openfoam*/etc/bashrc || source /usr/lib/openfoam/openfoam*/etc/bashrc || true',
    'mkdir -p /workspace/output',
    'if [ ! -d /workspace/case/system ]; then cp -r "$FOAM_TUTORIALS/incompressible/icoFoam/cavity/cavity" /workspace/case; fi',
    'cd /workspace/case',
    'blockMesh 2>&1 | tee /workspace/output/log.blockMesh',
    `${solver} 2>&1 | tee /workspace/output/log.${solver}`,
    'foamToVTK 2>&1 | tee /workspace/output/log.foamToVTK || true',
    'find /workspace/case -maxdepth 2 -type f \\( -name "log.*" -o -name "*.OpenFOAM" \\) -exec cp {} /workspace/output/ \\; || true',
  ].join(' && ')
}

async function collectArtifacts(sandbox: Sandbox) {
  const artifacts: PhysicsOSOpenFOAMRunResult['artifacts'] = []
  const entries = await sandbox.files.list('/workspace/output')
  for (const entry of entries) {
    if (entry.type !== FileType.FILE) continue
    const path = `/workspace/output/${entry.name}`
    const bytes = await sandbox.files.read(path, { format: 'bytes' })
    artifacts.push({
      name: entry.name,
      path,
      size: bytes.length,
      content_base64: Buffer.from(bytes).toString('base64'),
    })
  }
  return artifacts
}

export async function runOpenFOAMManifest(rawManifest: unknown): Promise<PhysicsOSOpenFOAMRunResult> {
  const manifest = ManifestSchema.parse(rawManifest)
  if (manifest.backend !== 'openfoam') {
    throw new Error(`foamvm only accepts backend=openfoam, got ${manifest.backend}`)
  }
  const templateId = process.env.E2B_TEMPLATE_ID || 'base'
  const sandbox = await Sandbox.create(templateId, {
    ...sandboxApiOpts(),
    timeoutMs: 30 * 60 * 1000,
  })
  try {
    await sandbox.commands.run('mkdir -p /workspace/output /workspace/physicsos', { timeoutMs: 10000 })
    await sandbox.files.write('/workspace/physicsos/manifest.json', JSON.stringify(manifest, null, 2))
    await writeCaseFiles(sandbox, manifest.openfoam?.case_files)
    const command = defaultOpenFOAMCommand(manifest)
    const result = await runCommand(sandbox, `bash -lc '${command.replace(/'/g, `'"'"'`)}'`, 20 * 60 * 1000)
    const artifacts = await collectArtifacts(sandbox)
    const exitCode = 'exitCode' in result ? result.exitCode : 0
    return {
      job_id: `e2b:${sandbox.sandboxId}:${manifest.problem_id}`,
      status: exitCode === 0 ? 'completed' : 'failed',
      sandbox_id: sandbox.sandboxId,
      backend: manifest.backend,
      problem_id: manifest.problem_id,
      command,
      stdout_tail: (result.stdout || '').slice(-4000),
      stderr_tail: (result.stderr || '').slice(-4000),
      artifacts,
    }
  } finally {
    await sandbox.kill().catch(() => {})
  }
}

import { CommandExitError, FileType, Sandbox } from 'e2b'

import type { SandboxOutputFile } from '@/lib/run-tokens'
import { CLAUDE_MD } from './claude-md'

const HPC_SKILLS_REPO = 'https://github.com/SciMate-AI/HPC-Skills'

async function run(sandbox: Sandbox, cmd: string, timeoutMs = 60000): Promise<string> {
  try {
    const result = await sandbox.commands.run(cmd, { timeoutMs })
    return result.stdout
  } catch (error) {
    if (error instanceof CommandExitError) {
      throw new Error(
        `Command failed (exit ${error.exitCode}): ${cmd.slice(0, 80)}\n` +
        `stdout: ${error.stdout?.slice(-500) || '(empty)'}\n` +
        `stderr: ${error.stderr?.slice(-500) || '(empty)'}`,
      )
    }

    throw error
  }
}

function inferContentType(filename: string): string {
  const ext = filename.split('.').pop()?.toLowerCase() || ''
  const contentTypeMap: Record<string, string> = {
    csv: 'text/csv',
    gif: 'image/gif',
    jpeg: 'image/jpeg',
    jpg: 'image/jpeg',
    json: 'application/json',
    log: 'text/plain',
    pdf: 'application/pdf',
    png: 'image/png',
    svg: 'image/svg+xml',
    txt: 'text/plain',
    vtk: 'application/octet-stream',
    vtu: 'application/octet-stream',
    webp: 'image/webp',
  }

  return contentTypeMap[ext] || 'application/octet-stream'
}

const activeSandboxes = new Map<string, Sandbox>()

function sandboxApiOpts() {
  return {
    apiKey: process.env.E2B_API_KEY,
  }
}

export async function createCFDSandbox(): Promise<{ sandbox: Sandbox; sessionId: string }> {
  const templateId = process.env.E2B_TEMPLATE_ID || 'base'

  const sandbox = await Sandbox.create(templateId, {
    ...sandboxApiOpts(),
    timeoutMs: 30 * 60 * 1000,
    envs: {
      ANTHROPIC_API_KEY: process.env.ANTHROPIC_API_KEY || '',
      ANTHROPIC_BASE_URL: process.env.ANTHROPIC_BASE_URL || '',
    },
  })

  const sessionId = sandbox.sandboxId
  activeSandboxes.set(sessionId, sandbox)

  await run(sandbox, 'mkdir -p /workspace/output /workspace/.claude/skills', 10000)
  await sandbox.files.write('/workspace/CLAUDE.md', CLAUDE_MD)
  await run(sandbox, `git clone --depth 1 ${HPC_SKILLS_REPO} /tmp/hpc-skills`, 60000)
  await run(sandbox, 'cp -r /tmp/hpc-skills/skills /workspace/.claude/', 10000)

  return { sandbox, sessionId }
}

export async function connectCFDSandbox(sessionId: string): Promise<Sandbox> {
  const existing = activeSandboxes.get(sessionId)
  if (existing) {
    return existing
  }

  const sandbox = await Sandbox.connect(sessionId, sandboxApiOpts())
  activeSandboxes.set(sessionId, sandbox)
  return sandbox
}

export async function killSandbox(sessionId: string) {
  const sandbox = activeSandboxes.get(sessionId)
  if (sandbox) {
    try {
      await sandbox.kill()
    } catch {
      // Ignore cleanup errors.
    }
    activeSandboxes.delete(sessionId)
    return
  }

  try {
    await Sandbox.kill(sessionId, sandboxApiOpts())
  } catch {
    // Ignore cleanup errors.
  }
}

export async function collectOutputFiles(sandbox: Sandbox): Promise<SandboxOutputFile[]> {
  const files: SandboxOutputFile[] = []

  try {
    const entries = await sandbox.files.list('/workspace/output')
    for (const entry of entries) {
      if (entry.type !== FileType.FILE) continue

      try {
        const bytes = await sandbox.files.read(`/workspace/output/${entry.name}`, { format: 'bytes' })
        files.push({
          name: entry.name,
          isImage: /\.(png|jpg|jpeg|gif|svg|webp)$/i.test(entry.name),
          size: bytes.length,
          bytes,
          contentType: inferContentType(entry.name),
        })
      } catch {
        // Skip unreadable files.
      }
    }
  } catch {
    // Output directory may not exist.
  }

  return files
}

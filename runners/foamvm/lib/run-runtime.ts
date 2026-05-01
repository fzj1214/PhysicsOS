import { CommandExitError } from 'e2b'
import type { Sandbox } from 'e2b'

import { appendRunEvent, getRunConsumption, type RunEventPayload, type RunEventRecord } from '@/lib/runs'
import { collectOutputFiles, connectCFDSandbox, createCFDSandbox, killSandbox } from '@/lib/sandbox'
import { toDataUrl, updateRunConsumption, uploadRunOutputs } from '@/lib/run-tokens'

interface ActiveRunState {
  listeners: Set<(event: RunEventRecord) => void>
  task: Promise<void> | null
  serial: Promise<void>
  eventError: Error | null
  cleanupTimer: ReturnType<typeof setTimeout> | null
}

const activeRuns = new Map<string, ActiveRunState>()

interface CommandHandleLike {
  wait(): Promise<unknown>
}

function ensureState(runId: string): ActiveRunState {
  const existing = activeRuns.get(runId)
  if (existing) {
    if (existing.cleanupTimer) {
      clearTimeout(existing.cleanupTimer)
      existing.cleanupTimer = null
    }
    return existing
  }

  const created: ActiveRunState = {
    listeners: new Set(),
    task: null,
    serial: Promise.resolve(),
    eventError: null,
    cleanupTimer: null,
  }
  activeRuns.set(runId, created)
  return created
}

function scheduleCleanup(runId: string) {
  const state = activeRuns.get(runId)
  if (!state || state.listeners.size > 0 || state.task) {
    return
  }

  if (state.cleanupTimer) {
    clearTimeout(state.cleanupTimer)
  }

  state.cleanupTimer = setTimeout(() => {
    const latest = activeRuns.get(runId)
    if (latest && latest.listeners.size === 0 && !latest.task) {
      activeRuns.delete(runId)
    }
  }, 15 * 60 * 1000)
}

function normalizeError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error))
}

async function enqueueRunEvent(runId: string, payload: RunEventPayload): Promise<RunEventRecord> {
  const state = ensureState(runId)
  const operation = state.serial.then(async () => {
    if (state.eventError) {
      throw state.eventError
    }

    const stored = await appendRunEvent({
      consumptionId: runId,
      payload,
    })

    for (const listener of state.listeners) {
      try {
        listener(stored)
      } catch {
        // Ignore subscriber errors and keep the run alive.
      }
    }

    return stored
  })

  state.serial = operation.then(
    () => undefined,
    (error) => {
      state.eventError = normalizeError(error)
    },
  )

  return operation
}

function enqueueRunEventBackground(runId: string, payload: RunEventPayload) {
  void enqueueRunEvent(runId, payload).catch(() => {})
}

async function drainRunEvents(runId: string) {
  const state = ensureState(runId)
  await state.serial
  if (state.eventError) {
    throw state.eventError
  }
}

function createLineProcessor(onLine: (line: string) => void) {
  let buffer = ''

  return {
    push(chunk: string) {
      buffer += chunk
      const parts = buffer.split('\n')
      buffer = parts.pop() ?? ''
      for (const part of parts) {
        onLine(part)
      }
    },
    flush() {
      if (buffer) {
        onLine(buffer)
        buffer = ''
      }
    },
  }
}

function buildPromptExcerpt(prompt: string): string {
  const cleaned = prompt.replace(/\s+/g, ' ').trim()
  return cleaned.slice(0, 280)
}

async function waitForCommand(
  sandbox: Sandbox,
  pid: number,
  initialHandle: CommandHandleLike,
  onStdout: (data: string) => void,
  onStderr: (data: string) => void,
): Promise<void> {
  const keepaliveMs = 4 * 60 * 1000
  const reconnectDelayMs = 2000
  const maxReconnect = 8

  const keepalive = setInterval(async () => {
    try {
      await sandbox.setTimeout(30 * 60 * 1000)
    } catch {
      // Ignore sandbox keepalive failures.
    }
  }, keepaliveMs)

  try {
    let currentHandle = initialHandle
    let attempts = 0

    while (true) {
      try {
        await currentHandle.wait()
        return
      } catch (error) {
        if (!(error instanceof CommandExitError)) {
          throw error
        }

        const isConnectionDrop =
          error.message?.includes('terminated') ||
          error.message?.includes('disconnect') ||
          error.message?.includes('unknown')

        if (!isConnectionDrop) {
          if (error.stderr?.trim()) onStderr(error.stderr.slice(-500))
          throw error
        }

        if (attempts >= maxReconnect) {
          throw new Error(`Lost command stream after ${maxReconnect} reconnect attempts.`)
        }

        attempts += 1
        onStderr(`[reconnect attempt ${attempts}/${maxReconnect}]`)
        await new Promise((resolve) => setTimeout(resolve, reconnectDelayMs))
        currentHandle = await sandbox.commands.connect(pid, { onStdout, onStderr, timeoutMs: 0 })
      }
    }
  } finally {
    clearInterval(keepalive)
  }
}

async function finalizeSuccessfulRun(params: {
  runId: string
  userId: string
  sandbox: Sandbox
  sessionId: string
  commandPid: number | null
}) {
  await enqueueRunEvent(params.runId, { type: 'status', message: 'Collecting output files...' })

  const outputFiles = await collectOutputFiles(params.sandbox)
  const uploadedFiles = await uploadRunOutputs({
    userId: params.userId,
    consumptionId: params.runId,
    files: outputFiles,
  })

  for (const file of outputFiles) {
    const uploaded = uploadedFiles.get(file.name)
    if (!uploaded) {
      continue
    }

    if (file.isImage) {
      const dataUrl = toDataUrl(file)
      if (dataUrl) {
        await enqueueRunEvent(params.runId, {
          type: 'image',
          name: file.name,
          dataUrl,
        })
      }
    }

    await enqueueRunEvent(params.runId, {
      type: 'file',
      name: file.name,
      url: uploaded.url,
      size: file.size,
    })
  }

  await updateRunConsumption({
    consumptionId: params.runId,
    status: 'completed',
    sandboxSessionId: params.sessionId,
    commandPid: params.commandPid,
    errorMessage: null,
  })

  await enqueueRunEvent(params.runId, {
    type: 'done',
    sessionId: params.sessionId,
  })
}

async function failRun(params: {
  runId: string
  sessionId: string | null
  commandPid: number | null
  error: unknown
}) {
  const message = normalizeError(params.error).message

  await updateRunConsumption({
    consumptionId: params.runId,
    status: 'failed',
    sandboxSessionId: params.sessionId,
    commandPid: params.commandPid,
    errorMessage: message,
  }).catch(() => {})

  await enqueueRunEvent(params.runId, {
    type: 'error',
    message,
  }).catch(() => {})
}

async function runFreshExecution(params: {
  runId: string
  userId: string
  prompt: string
  remainingRuns: number
}) {
  let sessionId: string | null = null
  let commandPid: number | null = null

  try {
    await enqueueRunEvent(params.runId, {
      type: 'credit',
      remainingRuns: params.remainingRuns,
    })
    await enqueueRunEvent(params.runId, {
      type: 'status',
      message: 'Creating sandbox...',
    })

    const { sandbox, sessionId: createdSessionId } = await createCFDSandbox()
    sessionId = createdSessionId

    await updateRunConsumption({
      consumptionId: params.runId,
      status: 'running',
      sandboxSessionId: sessionId,
      errorMessage: null,
    })

    await enqueueRunEvent(params.runId, {
      type: 'session',
      sessionId,
    })
    await enqueueRunEvent(params.runId, {
      type: 'status',
      message: 'Verifying sandbox environment...',
    })

    const check = await sandbox.commands.run('which claude && claude --version && node --version', {
      timeoutMs: 30000,
    })
    await enqueueRunEvent(params.runId, {
      type: 'log',
      text: `[env] ${check.stdout.trim()}`,
    })

    const skills = await sandbox.commands.run(
      'ls /workspace/.claude/skills/hpc-openfoam/SKILL.md 2>/dev/null && echo ok || echo missing',
      { timeoutMs: 10000 },
    )
    await enqueueRunEvent(params.runId, {
      type: 'log',
      text: `[skills] ${skills.stdout.trim()}`,
    })
    await enqueueRunEvent(params.runId, {
      type: 'status',
      message: 'Launching Claude Code...',
    })

    const stdoutLines = createLineProcessor((line) => {
      const trimmed = line.trim()
      if (!trimmed) return

      try {
        enqueueRunEventBackground(params.runId, {
          type: 'claude',
          payload: JSON.parse(trimmed) as Record<string, unknown>,
        })
      } catch {
        enqueueRunEventBackground(params.runId, {
          type: 'log',
          text: trimmed,
        })
      }
    })

    const stderrLines = createLineProcessor((line) => {
      const trimmed = line.trim()
      if (!trimmed) return
      enqueueRunEventBackground(params.runId, {
        type: 'stderr',
        text: trimmed,
      })
    })

    const escapedPrompt = params.prompt.replace(/'/g, `'"'"'`)
    const claudeCmd = `claude -p '${escapedPrompt}' --output-format stream-json --verbose --dangerously-skip-permissions`
    const handle = await sandbox.commands.run(claudeCmd, {
      cwd: '/workspace',
      envs: {
        ANTHROPIC_API_KEY: process.env.ANTHROPIC_API_KEY || '',
        ANTHROPIC_BASE_URL: process.env.ANTHROPIC_BASE_URL || '',
      },
      background: true,
      timeoutMs: 0,
      onStdout: (data) => stdoutLines.push(data),
      onStderr: (data) => stderrLines.push(data),
    } as Parameters<typeof sandbox.commands.run>[1] & { background: true })

    commandPid = handle.pid

    await updateRunConsumption({
      consumptionId: params.runId,
      status: 'running',
      sandboxSessionId: sessionId,
      commandPid,
      errorMessage: null,
    })

    await waitForCommand(
      sandbox,
      handle.pid,
      handle,
      (data) => stdoutLines.push(data),
      (data) => stderrLines.push(data),
    )

    stdoutLines.flush()
    stderrLines.flush()
    await drainRunEvents(params.runId)

    await finalizeSuccessfulRun({
      runId: params.runId,
      userId: params.userId,
      sandbox,
      sessionId,
      commandPid,
    })
  } catch (error) {
    await failRun({
      runId: params.runId,
      sessionId,
      commandPid,
      error,
    })
  } finally {
    if (sessionId) {
      const sandboxSessionId = sessionId
      setTimeout(() => {
        void killSandbox(sandboxSessionId)
      }, 10 * 60 * 1000)
    }
  }
}

async function runResumedExecution(runId: string) {
  const run = await getRunConsumption(runId)
  if (!run) {
    throw new Error('Run not found while resuming.')
  }

  if (!run.sandboxSessionId || !run.commandPid) {
    throw new Error('Run cannot be resumed because sandbox session or command pid is missing.')
  }

  let sessionId: string | null = run.sandboxSessionId
  let commandPid: number | null = run.commandPid

  try {
    const sandbox = await connectCFDSandbox(run.sandboxSessionId)
    await enqueueRunEvent(runId, {
      type: 'status',
      message: 'Reconnecting to running sandbox...',
    })

    const stdoutLines = createLineProcessor((line) => {
      const trimmed = line.trim()
      if (!trimmed) return

      try {
        enqueueRunEventBackground(runId, {
          type: 'claude',
          payload: JSON.parse(trimmed) as Record<string, unknown>,
        })
      } catch {
        enqueueRunEventBackground(runId, {
          type: 'log',
          text: trimmed,
        })
      }
    })

    const stderrLines = createLineProcessor((line) => {
      const trimmed = line.trim()
      if (!trimmed) return
      enqueueRunEventBackground(runId, {
        type: 'stderr',
        text: trimmed,
      })
    })

    const handle = await sandbox.commands.connect(run.commandPid, {
      onStdout: (data) => stdoutLines.push(data),
      onStderr: (data) => stderrLines.push(data),
      timeoutMs: 0,
    })

    await waitForCommand(
      sandbox,
      run.commandPid,
      handle,
      (data) => stdoutLines.push(data),
      (data) => stderrLines.push(data),
    )

    stdoutLines.flush()
    stderrLines.flush()
    await drainRunEvents(runId)

    await finalizeSuccessfulRun({
      runId,
      userId: run.userId,
      sandbox,
      sessionId: run.sandboxSessionId,
      commandPid: run.commandPid,
    })
  } catch (error) {
    await failRun({
      runId,
      sessionId,
      commandPid,
      error,
    })
  } finally {
    if (run.sandboxSessionId) {
      setTimeout(() => {
        void killSandbox(run.sandboxSessionId!)
      }, 10 * 60 * 1000)
    }
  }
}

export function subscribeToRun(runId: string, listener: (event: RunEventRecord) => void): () => void {
  const state = ensureState(runId)
  state.listeners.add(listener)

  return () => {
    const latest = activeRuns.get(runId)
    if (!latest) {
      return
    }

    latest.listeners.delete(listener)
    scheduleCleanup(runId)
  }
}

export function hasActiveRunTask(runId: string): boolean {
  return Boolean(activeRuns.get(runId)?.task)
}

export async function startRunExecution(params: {
  runId: string
  userId: string
  prompt: string
  remainingRuns: number
}) {
  const state = ensureState(params.runId)
  if (state.task) {
    return
  }

  const task = runFreshExecution(params).finally(() => {
    const latest = activeRuns.get(params.runId)
    if (latest) {
      latest.task = null
      scheduleCleanup(params.runId)
    }
  })

  state.task = task
}

export async function ensureRunExecution(runId: string) {
  const state = ensureState(runId)
  if (state.task) {
    return
  }

  const run = await getRunConsumption(runId)
  if (!run || (run.status !== 'starting' && run.status !== 'running')) {
    scheduleCleanup(runId)
    return
  }

  if (!run.commandPid) {
    scheduleCleanup(runId)
    return
  }

  const task = runResumedExecution(runId)

  state.task = task.finally(() => {
    const latest = activeRuns.get(runId)
    if (latest) {
      latest.task = null
      scheduleCleanup(runId)
    }
  })
}

export function buildRunPromptExcerpt(prompt: string): string {
  return buildPromptExcerpt(prompt)
}

import { NextRequest, NextResponse } from 'next/server'

import { getAuthenticatedUser } from '@/lib/auth'
import { appendRunEvent, getUserRunConsumption, listUserRunEvents } from '@/lib/runs'
import { hasActiveRunTask } from '@/lib/run-runtime'
import { killSandbox } from '@/lib/sandbox'
import { updateRunConsumption } from '@/lib/run-tokens'

export const runtime = 'nodejs'

export async function GET(_request: NextRequest, context: { params: Promise<{ id: string }> }) {
  const user = await getAuthenticatedUser()
  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const { id } = await context.params
  const run = await getUserRunConsumption({
    consumptionId: id,
    userId: user.id,
  })

  if (!run) {
    return NextResponse.json({ error: 'Run not found' }, { status: 404 })
  }

  const events = await listUserRunEvents({
    consumptionId: id,
    userId: user.id,
  })

  return NextResponse.json({
    run,
    events,
  })
}

export async function DELETE(_request: NextRequest, context: { params: Promise<{ id: string }> }) {
  const user = await getAuthenticatedUser()
  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const { id } = await context.params
  const run = await getUserRunConsumption({
    consumptionId: id,
    userId: user.id,
  })

  if (!run) {
    return NextResponse.json({ error: 'Run not found' }, { status: 404 })
  }

  if (run.sandboxSessionId) {
    await killSandbox(run.sandboxSessionId)
  }

  if (!hasActiveRunTask(run.id)) {
    await updateRunConsumption({
      consumptionId: run.id,
      status: 'failed',
      sandboxSessionId: run.sandboxSessionId,
      commandPid: run.commandPid,
      errorMessage: 'Run stopped by user.',
    }).catch(() => {})

    await appendRunEvent({
      consumptionId: run.id,
      payload: {
        type: 'error',
        message: 'Run stopped by user.',
      },
    }).catch(() => {})
  }

  return NextResponse.json({ ok: true })
}

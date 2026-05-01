import { NextRequest, NextResponse } from 'next/server'

import { getAuthenticatedUser } from '@/lib/auth'
import { ensureRunExecution, hasActiveRunTask, subscribeToRun } from '@/lib/run-runtime'
import { getUserRunConsumption, listUserRunEvents, type RunEventRecord } from '@/lib/runs'

export const runtime = 'nodejs'
export const maxDuration = 800

function encodeEvent(event: RunEventRecord): string {
  return `id: ${event.id}\ndata: ${JSON.stringify({ ...event.payload, eventId: event.id })}\n\n`
}

export async function GET(request: NextRequest, context: { params: Promise<{ id: string }> }) {
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

  const afterRaw = new URL(request.url).searchParams.get('after')
  const afterId = afterRaw ? Number(afterRaw) : 0
  if (!Number.isFinite(afterId) || afterId < 0) {
    return NextResponse.json({ error: 'Invalid after cursor' }, { status: 400 })
  }

  const hasLiveTask = hasActiveRunTask(run.id)

  if ((run.status === 'starting' || run.status === 'running') && !hasLiveTask && run.commandPid) {
    await ensureRunExecution(run.id)
  }

  const encoder = new TextEncoder()

  const stream = new ReadableStream({
    async start(controller) {
      let closed = false
      let lastSentId = afterId
      const queuedLiveEvents: RunEventRecord[] = []
      let replayFinished = false
      const canTail = (run.status === 'starting' || run.status === 'running') && (hasLiveTask || Boolean(run.commandPid))

      const close = () => {
        if (closed) {
          return
        }
        closed = true
        unsubscribe?.()
        if (heartbeat) {
          clearInterval(heartbeat)
        }
        controller.close()
      }

      const send = (event: RunEventRecord) => {
        if (event.id <= lastSentId || closed) {
          return
        }

        lastSentId = event.id
        try {
          controller.enqueue(encoder.encode(encodeEvent(event)))
        } catch {
          close()
        }
      }

      const onLiveEvent = (event: RunEventRecord) => {
        if (!replayFinished) {
          queuedLiveEvents.push(event)
          return
        }
        send(event)
      }

      const unsubscribe = canTail ? subscribeToRun(run.id, onLiveEvent) : null
      const heartbeat = canTail
        ? setInterval(() => {
            if (closed) {
              return
            }
            try {
              controller.enqueue(encoder.encode(': keep-alive\n\n'))
            } catch {
              close()
            }
          }, 15000)
        : null

      request.signal.addEventListener('abort', close)

      try {
        const backlog = await listUserRunEvents({
          consumptionId: run.id,
          userId: user.id,
          afterId,
        })

        for (const event of backlog) {
          send(event)
        }

        replayFinished = true
        queuedLiveEvents.sort((a, b) => a.id - b.id)
        for (const event of queuedLiveEvents) {
          send(event)
        }
        queuedLiveEvents.length = 0

        if (!canTail) {
          close()
        }
      } catch {
        close()
      }
    },
  })

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
    },
  })
}

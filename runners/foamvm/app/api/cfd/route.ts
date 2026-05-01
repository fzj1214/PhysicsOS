import { NextRequest, NextResponse } from 'next/server'

import { getAuthenticatedUser } from '@/lib/auth'
import { buildRunPromptExcerpt, startRunExecution } from '@/lib/run-runtime'
import { getLatestActiveRunForUser } from '@/lib/runs'
import { consumeRunTokenForUser } from '@/lib/run-tokens'

export const runtime = 'nodejs'
export const maxDuration = 800

export async function GET() {
  const user = await getAuthenticatedUser()
  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const run = await getLatestActiveRunForUser(user.id)
  return NextResponse.json({ run })
}

export async function POST(request: NextRequest) {
  const user = await getAuthenticatedUser()
  if (!user) {
    return NextResponse.json({ error: 'You must sign in before running CFD jobs.' }, { status: 401 })
  }

  const { prompt } = await request.json()
  if (!prompt?.trim()) {
    return NextResponse.json({ error: 'Missing prompt' }, { status: 400 })
  }

  const consumeResult = await consumeRunTokenForUser({
    userId: user.id,
    promptExcerpt: buildRunPromptExcerpt(prompt),
  })

  if (!consumeResult.success || !consumeResult.consumptionId) {
    return NextResponse.json({ error: consumeResult.message }, { status: 403 })
  }

  await startRunExecution({
    runId: consumeResult.consumptionId,
    userId: user.id,
    prompt,
    remainingRuns: consumeResult.remainingRuns,
  })

  return NextResponse.json({
    runId: consumeResult.consumptionId,
    remainingRuns: consumeResult.remainingRuns,
    promptExcerpt: buildRunPromptExcerpt(prompt),
    status: 'starting',
  })
}

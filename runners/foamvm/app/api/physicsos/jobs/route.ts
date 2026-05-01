import { NextRequest, NextResponse } from 'next/server'

import { runOpenFOAMManifest } from '@/lib/physicsos-openfoam-runner'

export const runtime = 'nodejs'
export const maxDuration = 1200

function authorized(request: NextRequest) {
  const expected = process.env.PHYSICSOS_RUNNER_TOKEN
  if (!expected) return true
  const header = request.headers.get('authorization') || ''
  return header === `Bearer ${expected}`
}

export async function POST(request: NextRequest) {
  if (!authorized(request)) {
    return NextResponse.json({ error: 'Unauthorized PhysicsOS runner request.' }, { status: 401 })
  }

  try {
    const manifest = await request.json()
    const result = await runOpenFOAMManifest(manifest)
    return NextResponse.json(result, { status: result.status === 'completed' ? 200 : 500 })
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    return NextResponse.json({ error: message }, { status: 400 })
  }
}

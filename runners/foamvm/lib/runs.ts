import { createAdminSupabaseClient } from '@/lib/supabase/admin'

export type RunStatus = 'starting' | 'running' | 'completed' | 'failed'

export interface RunConsumptionRecord {
  id: string
  userId: string
  status: RunStatus
  sandboxSessionId: string | null
  commandPid: number | null
  promptExcerpt: string | null
  createdAt: string
  errorMessage: string | null
}

export interface RunEventPayload {
  type: string
  [key: string]: unknown
}

export interface RunEventRecord {
  id: number
  consumptionId: string
  payload: RunEventPayload
  createdAt: string
}

function mapRunConsumption(row: Record<string, unknown>): RunConsumptionRecord {
  return {
    id: row.id as string,
    userId: row.user_id as string,
    status: row.status as RunStatus,
    sandboxSessionId: (row.sandbox_session_id as string | null) ?? null,
    commandPid: row.command_pid == null ? null : Number(row.command_pid),
    promptExcerpt: (row.prompt_excerpt as string | null) ?? null,
    createdAt: row.created_at as string,
    errorMessage: (row.error_message as string | null) ?? null,
  }
}

function mapRunEvent(row: Record<string, unknown>): RunEventRecord {
  return {
    id: Number(row.id),
    consumptionId: row.consumption_id as string,
    payload: row.payload as RunEventPayload,
    createdAt: row.created_at as string,
  }
}

export async function getRunConsumption(consumptionId: string): Promise<RunConsumptionRecord | null> {
  const admin = createAdminSupabaseClient()
  const { data, error } = await admin
    .from('run_consumptions')
    .select('id, user_id, status, sandbox_session_id, command_pid, prompt_excerpt, created_at, error_message')
    .eq('id', consumptionId)
    .single()

  if (error || !data) {
    return null
  }

  return mapRunConsumption(data as Record<string, unknown>)
}

export async function getUserRunConsumption(params: {
  consumptionId: string
  userId: string
}): Promise<RunConsumptionRecord | null> {
  const admin = createAdminSupabaseClient()
  const { data, error } = await admin
    .from('run_consumptions')
    .select('id, user_id, status, sandbox_session_id, command_pid, prompt_excerpt, created_at, error_message')
    .eq('id', params.consumptionId)
    .eq('user_id', params.userId)
    .single()

  if (error || !data) {
    return null
  }

  return mapRunConsumption(data as Record<string, unknown>)
}

export async function getLatestActiveRunForUser(userId: string): Promise<RunConsumptionRecord | null> {
  const admin = createAdminSupabaseClient()
  const { data, error } = await admin
    .from('run_consumptions')
    .select('id, user_id, status, sandbox_session_id, command_pid, prompt_excerpt, created_at, error_message')
    .eq('user_id', userId)
    .in('status', ['starting', 'running'])
    .order('created_at', { ascending: false })
    .limit(1)

  if (error) {
    throw new Error(`Failed to load active run: ${error.message}`)
  }

  const row = Array.isArray(data) ? data[0] : null
  return row ? mapRunConsumption(row as Record<string, unknown>) : null
}

export async function appendRunEvent(params: {
  consumptionId: string
  payload: RunEventPayload
}): Promise<RunEventRecord> {
  const admin = createAdminSupabaseClient()
  const { data, error } = await admin
    .from('run_events')
    .insert({
      consumption_id: params.consumptionId,
      payload: params.payload,
    })
    .select('id, consumption_id, payload, created_at')
    .single()

  if (error || !data) {
    throw new Error(`Failed to append run event: ${error?.message || 'unknown error'}`)
  }

  return mapRunEvent(data as Record<string, unknown>)
}

export async function listUserRunEvents(params: {
  consumptionId: string
  userId: string
  afterId?: number
  limit?: number
}): Promise<RunEventRecord[]> {
  const admin = createAdminSupabaseClient()
  const run = await getUserRunConsumption({
    consumptionId: params.consumptionId,
    userId: params.userId,
  })

  if (!run) {
    return []
  }

  let query = admin
    .from('run_events')
    .select('id, consumption_id, payload, created_at')
    .eq('consumption_id', params.consumptionId)
    .order('id', { ascending: true })

  if (params.afterId != null) {
    query = query.gt('id', params.afterId)
  }

  if (params.limit != null) {
    query = query.limit(params.limit)
  }

  const { data, error } = await query

  if (error) {
    throw new Error(`Failed to load run events: ${error.message}`)
  }

  return (data ?? []).map((row) => mapRunEvent(row as Record<string, unknown>))
}

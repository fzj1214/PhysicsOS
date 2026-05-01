import { createHash, randomBytes } from 'node:crypto'

import { getOutputBucketName } from '@/lib/env'
import { createAdminSupabaseClient } from '@/lib/supabase/admin'

export interface RedeemRunTokenResult {
  success: boolean
  message: string
  remainingRuns: number
}

export interface ConsumeRunTokenResult {
  success: boolean
  message: string
  remainingRuns: number
  tokenId: string | null
  consumptionId: string | null
}

export interface RunHistoryItem {
  id: string
  status: string
  sandboxSessionId: string | null
  promptExcerpt: string | null
  createdAt: string
  errorMessage: string | null
}

export interface UserSandboxOutputFile {
  id: string
  filename: string
  sizeBytes: number
  contentType: string | null
  createdAt: string
  isImage: boolean
  url: string
}

export interface UserSandboxOutputGroup {
  runId: string
  sandboxSessionId: string | null
  promptExcerpt: string | null
  status: string
  createdAt: string
  files: UserSandboxOutputFile[]
}

export interface AdminTokenRow {
  id: string
  status: string
  assignedEmail: string | null
  redeemedBy: string | null
  note: string | null
  createdAt: string
  redeemedAt: string | null
  consumedAt: string | null
  expiresAt: string | null
}

export interface UploadedOutput {
  id: string
  url: string
}

export interface SandboxOutputFile {
  name: string
  isImage: boolean
  size: number
  bytes: Uint8Array
  contentType: string
}

function takeFirstRow<T>(data: T | T[] | null): T | null {
  if (Array.isArray(data)) {
    return data[0] ?? null
  }

  return data
}

export function hashRunToken(code: string): string {
  return createHash('sha256').update(code.trim().toUpperCase()).digest('hex')
}

function createTokenChunk(length: number): string {
  const alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
  const bytes = randomBytes(length)
  let output = ''

  for (let i = 0; i < length; i += 1) {
    output += alphabet[bytes[i] % alphabet.length]
  }

  return output
}

export function generateRunTokenCode(): string {
  return `SCM-${createTokenChunk(4)}-${createTokenChunk(4)}-${createTokenChunk(4)}`
}

export async function redeemRunTokenForUser(params: {
  userId: string
  email: string | null
  code: string
}): Promise<RedeemRunTokenResult> {
  const admin = createAdminSupabaseClient()
  const { data, error } = await admin.rpc('redeem_run_token', {
    p_user_id: params.userId,
    p_user_email: params.email,
    p_code_hash: hashRunToken(params.code),
  })

  if (error) {
    throw new Error(`Failed to redeem run token: ${error.message}`)
  }

  const result = takeFirstRow<{ success: boolean; message: string; remaining_runs: number }>(data)
  if (!result) {
    throw new Error('Run token redeem returned no result')
  }

  return {
    success: result.success,
    message: result.message,
    remainingRuns: result.remaining_runs ?? 0,
  }
}

export async function consumeRunTokenForUser(params: {
  userId: string
  promptExcerpt: string | null
}): Promise<ConsumeRunTokenResult> {
  const admin = createAdminSupabaseClient()
  const { data, error } = await admin.rpc('consume_redeemed_run_token', {
    p_user_id: params.userId,
    p_prompt_excerpt: params.promptExcerpt,
  })

  if (error) {
    throw new Error(`Failed to consume run token: ${error.message}`)
  }

  const result = takeFirstRow<{
    success: boolean
    message: string
    remaining_runs: number
    token_id: string | null
    consumption_id: string | null
  }>(data)

  if (!result) {
    throw new Error('Run token consume returned no result')
  }

  return {
    success: result.success,
    message: result.message,
    remainingRuns: result.remaining_runs ?? 0,
    tokenId: result.token_id,
    consumptionId: result.consumption_id,
  }
}

export async function updateRunConsumption(params: {
  consumptionId: string
  status: 'starting' | 'running' | 'completed' | 'failed'
  sandboxSessionId?: string | null
  commandPid?: number | null
  errorMessage?: string | null
}): Promise<void> {
  const admin = createAdminSupabaseClient()
  const { error } = await admin
    .from('run_consumptions')
    .update({
      status: params.status,
      sandbox_session_id: params.sandboxSessionId ?? null,
      command_pid: params.commandPid ?? null,
      error_message: params.errorMessage ?? null,
    })
    .eq('id', params.consumptionId)

  if (error) {
    throw new Error(`Failed to update run consumption: ${error.message}`)
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

export function toDataUrl(file: SandboxOutputFile): string | null {
  if (!file.isImage) {
    return null
  }

  return `data:${file.contentType};base64,${Buffer.from(file.bytes).toString('base64')}`
}

export async function uploadRunOutputs(params: {
  userId: string
  consumptionId: string
  files: SandboxOutputFile[]
}): Promise<Map<string, UploadedOutput>> {
  const admin = createAdminSupabaseClient()
  const bucket = getOutputBucketName()
  const uploaded = new Map<string, UploadedOutput>()

  for (const file of params.files) {
    const storagePath = `${params.userId}/${params.consumptionId}/${file.name}`
    const { error: uploadError } = await admin.storage
      .from(bucket)
      .upload(storagePath, Buffer.from(file.bytes), {
        contentType: file.contentType || inferContentType(file.name),
        upsert: false,
      })

    if (uploadError) {
      throw new Error(`Failed to upload ${file.name}: ${uploadError.message}`)
    }

    const { data, error } = await admin
      .from('run_output_files')
      .insert({
        consumption_id: params.consumptionId,
        filename: file.name,
        storage_path: storagePath,
        content_type: file.contentType || inferContentType(file.name),
        size_bytes: file.size,
        is_image: file.isImage,
      })
      .select('id')
      .single()

    if (error || !data) {
      throw new Error(`Failed to register output file ${file.name}`)
    }

    uploaded.set(file.name, {
      id: data.id as string,
      url: `/api/files?id=${encodeURIComponent(data.id as string)}`,
    })
  }

  return uploaded
}

export async function listUserRunHistory(userId: string): Promise<RunHistoryItem[]> {
  const admin = createAdminSupabaseClient()
  const { data, error } = await admin
    .from('run_consumptions')
    .select('id, status, sandbox_session_id, prompt_excerpt, created_at, error_message')
    .eq('user_id', userId)
    .order('created_at', { ascending: false })
    .limit(20)

  if (error) {
    throw new Error(`Failed to load run history: ${error.message}`)
  }

  return (data ?? []).map((row) => ({
    id: row.id as string,
    status: row.status as string,
    sandboxSessionId: (row.sandbox_session_id as string | null) ?? null,
    promptExcerpt: (row.prompt_excerpt as string | null) ?? null,
    createdAt: row.created_at as string,
    errorMessage: (row.error_message as string | null) ?? null,
  }))
}

export async function listUserSandboxOutputGroups(userId: string): Promise<UserSandboxOutputGroup[]> {
  const admin = createAdminSupabaseClient()
  const { data, error } = await admin
    .from('run_consumptions')
    .select(`
      id,
      sandbox_session_id,
      prompt_excerpt,
      status,
      created_at,
      run_output_files (
        id,
        filename,
        size_bytes,
        content_type,
        is_image,
        created_at
      )
    `)
    .eq('user_id', userId)
    .order('created_at', { ascending: false })
    .limit(50)

  if (error) {
    throw new Error(`Failed to load sandbox outputs: ${error.message}`)
  }

  return (data ?? [])
    .map((row) => {
      const files = Array.isArray((row as { run_output_files?: unknown[] }).run_output_files)
        ? (row as { run_output_files: Record<string, unknown>[] }).run_output_files
        : []

      return {
        runId: row.id as string,
        sandboxSessionId: (row.sandbox_session_id as string | null) ?? null,
        promptExcerpt: (row.prompt_excerpt as string | null) ?? null,
        status: row.status as string,
        createdAt: row.created_at as string,
        files: files.map((file) => ({
          id: file.id as string,
          filename: file.filename as string,
          sizeBytes: Number(file.size_bytes ?? 0),
          contentType: (file.content_type as string | null) ?? null,
          createdAt: file.created_at as string,
          isImage: Boolean(file.is_image),
          url: `/api/files?id=${encodeURIComponent(file.id as string)}`,
        })),
      }
    })
    .filter((group) => group.files.length > 0)
}

export async function listAdminRunTokens(): Promise<AdminTokenRow[]> {
  const admin = createAdminSupabaseClient()
  const { data, error } = await admin
    .from('run_tokens')
    .select('id, status, assigned_email, redeemed_by, note, created_at, redeemed_at, consumed_at, expires_at')
    .order('created_at', { ascending: false })
    .limit(100)

  if (error) {
    throw new Error(`Failed to load run tokens: ${error.message}`)
  }

  return (data ?? []).map((row) => ({
    id: row.id as string,
    status: row.status as string,
    assignedEmail: (row.assigned_email as string | null) ?? null,
    redeemedBy: (row.redeemed_by as string | null) ?? null,
    note: (row.note as string | null) ?? null,
    createdAt: row.created_at as string,
    redeemedAt: (row.redeemed_at as string | null) ?? null,
    consumedAt: (row.consumed_at as string | null) ?? null,
    expiresAt: (row.expires_at as string | null) ?? null,
  }))
}

export async function createRunTokenBatch(params: {
  createdBy: string
  quantity: number
  assignedEmail?: string | null
  note?: string | null
  expiresAt?: string | null
}): Promise<{ batchId: string; codes: string[] }> {
  const admin = createAdminSupabaseClient()
  const { data: batch, error: batchError } = await admin
    .from('invite_batches')
    .insert({
      created_by: params.createdBy,
      quantity: params.quantity,
      note: params.note ?? null,
    })
    .select('id')
    .single()

  if (batchError || !batch) {
    throw new Error(`Failed to create token batch: ${batchError?.message || 'unknown error'}`)
  }

  const codes = Array.from({ length: params.quantity }, () => generateRunTokenCode())
  const rows = codes.map((code) => ({
    batch_id: batch.id,
    code_hash: hashRunToken(code),
    assigned_email: params.assignedEmail?.toLowerCase() || null,
    created_by: params.createdBy,
    status: 'unused',
    note: params.note ?? null,
    expires_at: params.expiresAt ?? null,
  }))

  const { error: insertError } = await admin.from('run_tokens').insert(rows)
  if (insertError) {
    throw new Error(`Failed to store generated tokens: ${insertError.message}`)
  }

  const { error: auditError } = await admin.from('admin_audit_logs').insert({
    actor_user_id: params.createdBy,
    action: 'generate_run_tokens',
    details: {
      batch_id: batch.id,
      quantity: params.quantity,
      assigned_email: params.assignedEmail ?? null,
      note: params.note ?? null,
    },
  })

  if (auditError) {
    throw new Error(`Failed to write audit log: ${auditError.message}`)
  }

  return {
    batchId: batch.id as string,
    codes,
  }
}

export async function revokeRunToken(params: {
  tokenId: string
  actorUserId: string
}): Promise<void> {
  const admin = createAdminSupabaseClient()
  const { data, error } = await admin
    .from('run_tokens')
    .update({ status: 'revoked' })
    .eq('id', params.tokenId)
    .eq('status', 'unused')
    .select('id')
    .single()

  if (error || !data) {
    throw new Error('Only unused tokens can be revoked')
  }

  const { error: auditError } = await admin.from('admin_audit_logs').insert({
    actor_user_id: params.actorUserId,
    action: 'revoke_run_token',
    details: {
      token_id: params.tokenId,
    },
  })

  if (auditError) {
    throw new Error(`Failed to write audit log: ${auditError.message}`)
  }
}

export async function getAuthorizedOutputFile(params: {
  fileId: string
  userId: string
}): Promise<{
  filename: string
  contentType: string
  sizeBytes: number
  storagePath: string
} | null> {
  const admin = createAdminSupabaseClient()
  const { data: fileRecord, error: fileError } = await admin
    .from('run_output_files')
    .select('id, consumption_id, filename, content_type, size_bytes, storage_path')
    .eq('id', params.fileId)
    .single()

  if (fileError || !fileRecord) {
    return null
  }

  const { data: consumption, error: consumptionError } = await admin
    .from('run_consumptions')
    .select('user_id')
    .eq('id', fileRecord.consumption_id as string)
    .single()

  if (consumptionError || !consumption || consumption.user_id !== params.userId) {
    return null
  }

  return {
    filename: fileRecord.filename as string,
    contentType: (fileRecord.content_type as string | null) || inferContentType(fileRecord.filename as string),
    sizeBytes: Number(fileRecord.size_bytes),
    storagePath: fileRecord.storage_path as string,
  }
}

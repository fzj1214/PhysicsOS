import { redirect } from 'next/navigation'
import type { User } from '@supabase/supabase-js'

import { createAdminSupabaseClient } from '@/lib/supabase/admin'
import { createServerSupabaseClient } from '@/lib/supabase/server'

export interface ViewerContext {
  user: User | null
  isAdmin: boolean
  availableRuns: number
  consumedRuns: number
}

interface ProfileRow {
  id: string
  role: 'admin' | 'user'
}

export async function getAuthenticatedUser(): Promise<User | null> {
  const supabase = await createServerSupabaseClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  return user
}

export async function requireAuthenticatedUser(nextPath?: string): Promise<User> {
  const user = await getAuthenticatedUser()
  if (!user) {
    const suffix = nextPath ? `?next=${encodeURIComponent(nextPath)}` : ''
    redirect(`/auth/login${suffix}`)
  }

  return user
}

export async function ensureProfileRecord(user: User): Promise<ProfileRow> {
  const admin = createAdminSupabaseClient()

  await admin.from('profiles').upsert(
    {
      id: user.id,
      email: user.email ?? null,
    },
    {
      onConflict: 'id',
      ignoreDuplicates: false,
    },
  )

  const { data, error } = await admin
    .from('profiles')
    .select('id, role')
    .eq('id', user.id)
    .single()

  if (error || !data) {
    throw new Error('Unable to load profile')
  }

  return data as ProfileRow
}

export async function isUserAdmin(userId: string): Promise<boolean> {
  const admin = createAdminSupabaseClient()
  const { data, error } = await admin
    .from('profiles')
    .select('role')
    .eq('id', userId)
    .single()

  if (error || !data) {
    return false
  }

  return data.role === 'admin'
}

export async function requireAdminUser(nextPath = '/admin/tokens'): Promise<User> {
  const user = await requireAuthenticatedUser(nextPath)
  const profile = await ensureProfileRecord(user)

  if (profile.role !== 'admin') {
    redirect('/')
  }

  return user
}

export async function getViewerContext(): Promise<ViewerContext> {
  const user = await getAuthenticatedUser()
  if (!user) {
    return {
      user: null,
      isAdmin: false,
      availableRuns: 0,
      consumedRuns: 0,
    }
  }

  const profile = await ensureProfileRecord(user)
  const admin = createAdminSupabaseClient()

  const [{ count: availableRuns }, { count: consumedRuns }] = await Promise.all([
    admin
      .from('run_tokens')
      .select('*', { head: true, count: 'exact' })
      .eq('redeemed_by', user.id)
      .eq('status', 'redeemed'),
    admin
      .from('run_tokens')
      .select('*', { head: true, count: 'exact' })
      .eq('redeemed_by', user.id)
      .eq('status', 'consumed'),
  ])

  return {
    user,
    isAdmin: profile.role === 'admin',
    availableRuns: availableRuns ?? 0,
    consumedRuns: consumedRuns ?? 0,
  }
}

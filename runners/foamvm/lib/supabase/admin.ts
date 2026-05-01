import { createClient } from '@supabase/supabase-js'

import { getSupabaseServiceRoleKey, getSupabaseUrl } from '@/lib/env'

export function createAdminSupabaseClient() {
  return createClient(getSupabaseUrl(), getSupabaseServiceRoleKey(), {
    auth: {
      autoRefreshToken: false,
      persistSession: false,
    },
  })
}

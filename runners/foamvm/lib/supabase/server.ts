import { createServerClient } from '@supabase/ssr'
import { cookies } from 'next/headers'

import { getSupabaseAnonKey, getSupabaseUrl } from '@/lib/env'

export async function createServerSupabaseClient() {
  const cookieStore = await cookies()

  return createServerClient(getSupabaseUrl(), getSupabaseAnonKey(), {
    cookies: {
      getAll() {
        return cookieStore.getAll()
      },
      setAll(cookiesToSet) {
        try {
          cookiesToSet.forEach(({ name, value, options }) => {
            cookieStore.set(name, value, options)
          })
        } catch {
          // Server Components cannot always mutate cookies. Proxy keeps them fresh.
        }
      },
    },
  })
}

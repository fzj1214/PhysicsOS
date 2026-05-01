import { createServerClient } from '@supabase/ssr'
import { NextRequest, NextResponse } from 'next/server'

import { getSupabaseAnonKey, getSupabaseUrl } from '@/lib/env'

export const runtime = 'nodejs'

const CALLBACK_TYPES = ['email', 'recovery', 'invite', 'email_change', 'magiclink'] as const
type CallbackType = (typeof CALLBACK_TYPES)[number]

function normalizeNextPath(value: string | null): string {
  if (!value || !value.startsWith('/') || value.startsWith('//')) {
    return '/'
  }

  return value
}

export async function GET(request: NextRequest) {
  const url = new URL(request.url)
  const nextPath = normalizeNextPath(url.searchParams.get('next'))
  const code = url.searchParams.get('code')
  const tokenHash = url.searchParams.get('token_hash')
  const type = url.searchParams.get('type')

  let response = NextResponse.redirect(new URL(nextPath, request.url))

  const supabase = createServerClient(getSupabaseUrl(), getSupabaseAnonKey(), {
    cookies: {
      getAll() {
        return request.cookies.getAll()
      },
      setAll(cookiesToSet) {
        cookiesToSet.forEach(({ name, value }) => request.cookies.set(name, value))
        response = NextResponse.redirect(new URL(nextPath, request.url))
        cookiesToSet.forEach(({ name, value, options }) => {
          response.cookies.set(name, value, options)
        })
      },
    },
  })

  if (code) {
    const { error } = await supabase.auth.exchangeCodeForSession(code)
    if (!error) {
      return response
    }
  }

  if (tokenHash && type && CALLBACK_TYPES.includes(type as CallbackType)) {
    const { error } = await supabase.auth.verifyOtp({
      token_hash: tokenHash,
      type: type as CallbackType,
    })

    if (!error) {
      return response
    }
  }

  return NextResponse.redirect(new URL('/auth/login?error=invalid_callback', request.url))
}

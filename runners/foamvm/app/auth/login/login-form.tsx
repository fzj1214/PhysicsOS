'use client'

import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { useState, useTransition } from 'react'
import { useSearchParams } from 'next/navigation'

import { getBrowserSupabaseClient } from '@/lib/supabase/client'

function normalizeNextPath(value: string | null): string {
  if (!value || !value.startsWith('/') || value.startsWith('//')) {
    return '/'
  }

  return value
}

export function LoginForm() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [email, setEmail] = useState('')
  const [code, setCode] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const [codeSent, setCodeSent] = useState(false)
  const [pending, startTransition] = useTransition()
  const next = normalizeNextPath(searchParams.get('next'))
  const callbackError = searchParams.get('error')

  const sendLoginEmail = () => {
    setError(null)
    setMessage(null)

    startTransition(async () => {
      const supabase = getBrowserSupabaseClient()
      const redirectTarget = new URL('/auth/confirm', window.location.origin)
      redirectTarget.searchParams.set('next', next)

      const { error: signInError } = await supabase.auth.signInWithOtp({
        email: email.trim(),
        options: {
          emailRedirectTo: redirectTarget.toString(),
        },
      })

      if (signInError) {
        setError(signInError.message)
        return
      }

      setCodeSent(true)
      setCode('')
      setMessage('We sent a sign-in code to your email. Enter it below to continue.')
    })
  }

  const handleSendSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    sendLoginEmail()
  }

  const handleVerifySubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setError(null)
    setMessage(null)

    startTransition(async () => {
      const supabase = getBrowserSupabaseClient()
      const { error: verifyError } = await supabase.auth.verifyOtp({
        email: email.trim(),
        token: code.trim(),
        type: 'email',
      })

      if (verifyError) {
        setError(verifyError.message)
        return
      }

      setMessage('Signed in. Redirecting...')
      router.refresh()
      window.location.assign(next)
    })
  }

  return (
    <div className="rounded-[28px] border border-white/10 bg-white/[0.03] p-6">
      <div className="mb-6">
        <div className="text-xs uppercase tracking-[0.28em] text-cyan-200/70">Passwordless email sign-in</div>
        <h2 className="mt-2 text-3xl font-semibold text-white">Sign in to unlock invited runs</h2>
        <p className="mt-3 text-sm leading-6 text-slate-300">
          Enter your email to receive a sign-in code. First-time users are created automatically. The actual gate is your
          run-token balance, not public signup.
        </p>
      </div>

      <form className="space-y-4" onSubmit={handleSendSubmit}>
        <label className="block">
          <span className="mb-2 block text-sm text-slate-300">Email</span>
          <input
            autoComplete="email"
            className="w-full rounded-2xl border border-white/10 bg-slate-950/70 px-4 py-3 text-white outline-none transition placeholder:text-slate-500 focus:border-cyan-300/60"
            onChange={(event) => setEmail(event.target.value)}
            placeholder="you@example.com"
            required
            type="email"
            value={email}
          />
        </label>

        <button
          className="w-full rounded-full bg-cyan-300 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-200 disabled:cursor-wait disabled:opacity-60"
          disabled={pending}
          type="submit"
        >
          {pending ? 'Sending sign-in email...' : codeSent ? 'Resend sign-in email' : 'Send sign-in email'}
        </button>
      </form>

      {codeSent ? (
        <form className="mt-5 space-y-4 rounded-[24px] border border-white/10 bg-black/20 p-4" onSubmit={handleVerifySubmit}>
          <div>
            <div className="text-xs uppercase tracking-[0.24em] text-cyan-200/70">Enter code from email</div>
            <p className="mt-2 text-sm leading-6 text-slate-300">
              This always works, even if the email was opened on a different browser or device.
            </p>
          </div>

          <label className="block">
            <span className="mb-2 block text-sm text-slate-300">Code from email</span>
            <input
              autoComplete="one-time-code"
              className="w-full rounded-2xl border border-white/10 bg-slate-950/70 px-4 py-3 text-white outline-none transition placeholder:text-slate-500 focus:border-cyan-300/60"
              inputMode="text"
              onChange={(event) => setCode(event.target.value)}
              placeholder="Enter the code exactly as shown"
              required
              type="text"
              value={code}
            />
          </label>

          <button
            className="w-full rounded-full border border-cyan-300/40 px-5 py-3 text-sm font-semibold text-cyan-100 transition hover:border-cyan-200 hover:text-white disabled:cursor-wait disabled:opacity-60"
            disabled={pending}
            type="submit"
          >
            {pending ? 'Verifying code...' : 'Sign in with code'}
          </button>
        </form>
      ) : null}

      {callbackError === 'invalid_callback' ? (
        <p className="mt-4 rounded-2xl border border-amber-300/20 bg-amber-300/10 px-4 py-3 text-sm text-amber-100">
          That sign-in link is invalid or expired. Request a fresh email and use the code from the email instead.
        </p>
      ) : null}
      {error ? <p className="mt-4 rounded-2xl border border-rose-300/20 bg-rose-300/10 px-4 py-3 text-sm text-rose-100">{error}</p> : null}
      {message ? <p className="mt-4 rounded-2xl border border-emerald-300/20 bg-emerald-300/10 px-4 py-3 text-sm text-emerald-100">{message}</p> : null}

      <div className="mt-6 text-sm leading-6 text-slate-400">
        Use the code from the email to finish sign-in. This avoids the cross-browser confusion of magic links.
      </div>

      <div className="mt-4 text-sm text-slate-400">
        Need a token after login? <Link className="text-cyan-200 underline" href="/redeem">Redeem one here</Link>.
      </div>
    </div>
  )
}

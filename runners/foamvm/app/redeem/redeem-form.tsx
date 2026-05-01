'use client'

import { useActionState } from 'react'

import type { FormState } from '@/app/actions/tokens'
import { redeemRunTokenAction } from '@/app/actions/tokens'

const initialState: FormState = {
  success: false,
  message: '',
}

export function RedeemForm() {
  const [state, action, pending] = useActionState(redeemRunTokenAction, initialState)

  return (
    <div className="rounded-[28px] border border-white/10 bg-white/[0.03] p-6">
      <div className="mb-5">
        <div className="text-xs uppercase tracking-[0.28em] text-cyan-200/70">Redeem a run token</div>
        <h2 className="mt-2 text-3xl font-semibold text-white">Convert one code into one ready-to-run credit</h2>
      </div>
      <form action={action} className="space-y-4">
        <label className="block">
          <span className="mb-2 block text-sm text-slate-300">Run token</span>
          <input
            className="w-full rounded-2xl border border-white/10 bg-slate-950/70 px-4 py-3 font-mono text-white uppercase outline-none transition placeholder:text-slate-500 focus:border-cyan-300/60"
            name="code"
            placeholder="SCM-XXXX-XXXX-XXXX"
            required
            type="text"
          />
        </label>
        <button
          className="rounded-full bg-cyan-300 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-200 disabled:cursor-wait disabled:opacity-60"
          disabled={pending}
          type="submit"
        >
          {pending ? 'Redeeming...' : 'Redeem token'}
        </button>
      </form>
      {state.message ? (
        <p className={`mt-4 rounded-2xl px-4 py-3 text-sm ${state.success ? 'border border-emerald-300/20 bg-emerald-300/10 text-emerald-50' : 'border border-rose-300/20 bg-rose-300/10 text-rose-100'}`}>
          {state.message}
        </p>
      ) : null}
    </div>
  )
}

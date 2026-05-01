import Link from 'next/link'

import { RedeemForm } from '@/app/redeem/redeem-form'
import { getViewerContext, requireAuthenticatedUser } from '@/lib/auth'

export default async function RedeemPage() {
  await requireAuthenticatedUser('/redeem')
  const viewer = await getViewerContext()

  return (
    <main className="min-h-screen bg-[linear-gradient(180deg,_#020617_0%,_#0f172a_100%)] px-4 py-12 text-slate-100">
      <div className="mx-auto max-w-3xl">
        <div className="mb-8 flex flex-wrap items-center gap-3">
          <Link className="rounded-full border border-white/10 px-3 py-1 text-sm text-slate-300 transition hover:border-white/30 hover:text-white" href="/">
            Back to home
          </Link>
          <Link className="rounded-full border border-white/10 px-3 py-1 text-sm text-slate-300 transition hover:border-white/30 hover:text-white" href="/account">
            Account
          </Link>
          <div className="rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-1 text-sm text-cyan-100">
            {viewer.availableRuns} ready run{viewer.availableRuns === 1 ? '' : 's'}
          </div>
        </div>

        <RedeemForm />
      </div>
    </main>
  )
}

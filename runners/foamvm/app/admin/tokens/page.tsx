import Link from 'next/link'

import { revokeRunTokenAction } from '@/app/actions/tokens'
import { GenerateTokensForm } from '@/app/admin/tokens/generate-form'
import { requireAdminUser } from '@/lib/auth'
import { listAdminRunTokens } from '@/lib/run-tokens'

export default async function AdminTokensPage() {
  await requireAdminUser('/admin/tokens')
  const tokens = await listAdminRunTokens()

  return (
    <main className="min-h-screen bg-[linear-gradient(180deg,_#020617_0%,_#111827_100%)] px-4 py-12 text-slate-100">
      <div className="mx-auto max-w-6xl">
        <div className="mb-8 flex flex-wrap items-center gap-3">
          <Link className="rounded-full border border-white/10 px-3 py-1 text-sm text-slate-300 transition hover:border-white/30 hover:text-white" href="/">
            Back to home
          </Link>
          <Link className="rounded-full border border-white/10 px-3 py-1 text-sm text-slate-300 transition hover:border-white/30 hover:text-white" href="/account">
            Account
          </Link>
        </div>

        <GenerateTokensForm />

        <section className="mt-6 rounded-[28px] border border-white/10 bg-white/[0.03] p-6">
          <div className="text-xs uppercase tracking-[0.28em] text-amber-200/70">Latest tokens</div>
          <div className="mt-5 overflow-hidden rounded-3xl border border-white/10">
            <table className="min-w-full divide-y divide-white/10 text-left text-sm">
              <thead className="bg-black/20 text-slate-400">
                <tr>
                  <th className="px-4 py-3 font-medium">Status</th>
                  <th className="px-4 py-3 font-medium">Assigned email</th>
                  <th className="px-4 py-3 font-medium">Redeemed by</th>
                  <th className="px-4 py-3 font-medium">Note</th>
                  <th className="px-4 py-3 font-medium">Created</th>
                  <th className="px-4 py-3 font-medium">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 bg-slate-950/30">
                {tokens.map((token) => (
                  <tr key={token.id}>
                    <td className="px-4 py-3 uppercase tracking-[0.22em] text-slate-200">{token.status}</td>
                    <td className="px-4 py-3 text-slate-300">{token.assignedEmail || '-'}</td>
                    <td className="px-4 py-3 text-slate-400">{token.redeemedBy || '-'}</td>
                    <td className="px-4 py-3 text-slate-300">{token.note || '-'}</td>
                    <td className="px-4 py-3 text-slate-400">{new Date(token.createdAt).toLocaleString()}</td>
                    <td className="px-4 py-3">
                      {token.status === 'unused' ? (
                        <form action={revokeRunTokenAction}>
                          <input name="tokenId" type="hidden" value={token.id} />
                          <button className="rounded-full border border-rose-300/20 bg-rose-300/10 px-3 py-1 text-xs text-rose-100 transition hover:border-rose-200/40" type="submit">
                            Revoke
                          </button>
                        </form>
                      ) : (
                        <span className="text-xs text-slate-500">Locked</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </main>
  )
}

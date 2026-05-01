import Link from 'next/link'
import { redirect } from 'next/navigation'

import { LoginForm } from '@/app/auth/login/login-form'
import { getAuthenticatedUser } from '@/lib/auth'

export default async function LoginPage() {
  const user = await getAuthenticatedUser()
  if (user) {
    redirect('/')
  }

  return (
    <main className="min-h-screen bg-[linear-gradient(180deg,_#020617_0%,_#0f172a_100%)] px-4 py-12 text-slate-100">
      <div className="mx-auto max-w-2xl">
        <Link className="mb-8 inline-flex rounded-full border border-white/10 px-3 py-1 text-sm text-slate-300 transition hover:border-white/30 hover:text-white" href="/">
          Back to home
        </Link>
        <LoginForm />
      </div>
    </main>
  )
}

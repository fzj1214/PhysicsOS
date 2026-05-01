import type { Metadata } from 'next'

import './globals.css'

export const metadata: Metadata = {
  title: 'SciMate | Invite-Gated CFD Runtime',
  description: 'Run OpenFOAM CFD workloads behind invite-controlled one-time run tokens.',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html className="h-full" lang="en">
      <body className="min-h-full bg-slate-950 text-slate-100 antialiased">{children}</body>
    </html>
  )
}

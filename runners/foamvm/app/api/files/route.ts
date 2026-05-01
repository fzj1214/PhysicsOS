import { NextRequest, NextResponse } from 'next/server'

import { getAuthenticatedUser } from '@/lib/auth'
import { getOutputBucketName } from '@/lib/env'
import { getAuthorizedOutputFile } from '@/lib/run-tokens'
import { createAdminSupabaseClient } from '@/lib/supabase/admin'

export const runtime = 'nodejs'

export async function GET(request: NextRequest) {
  const user = await getAuthenticatedUser()
  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const { searchParams } = new URL(request.url)
  const fileId = searchParams.get('id')

  if (!fileId) {
    return NextResponse.json({ error: 'Missing file id' }, { status: 400 })
  }

  const file = await getAuthorizedOutputFile({
    fileId,
    userId: user.id,
  })

  if (!file) {
    return NextResponse.json({ error: 'File not found' }, { status: 404 })
  }

  const admin = createAdminSupabaseClient()
  const { data, error } = await admin.storage
    .from(getOutputBucketName())
    .createSignedUrl(file.storagePath, 60 * 10, {
      download: file.filename,
    })

  if (error || !data?.signedUrl) {
    return NextResponse.json(
      {
        error: `Unable to create download link${error?.message ? `: ${error.message}` : ''}`,
      },
      { status: 500 },
    )
  }

  return NextResponse.redirect(data.signedUrl, {
    status: 302,
    headers: {
      'Cache-Control': 'private, no-store',
    },
  })
}

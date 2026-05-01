'use server'

import { revalidatePath } from 'next/cache'
import { z } from 'zod'

import { ensureProfileRecord, requireAdminUser, requireAuthenticatedUser } from '@/lib/auth'
import { createRunTokenBatch, redeemRunTokenForUser, revokeRunToken } from '@/lib/run-tokens'

export interface FormState {
  success: boolean
  message: string
  codes?: string[]
}

const redeemSchema = z.object({
  code: z.string().trim().min(6).max(64),
})

const generateSchema = z.object({
  quantity: z.coerce.number().int().min(1).max(100),
  assignedEmail: z.string().trim().optional().or(z.literal('')),
  note: z.string().trim().max(200).optional().or(z.literal('')),
  expiresAt: z.string().trim().optional().or(z.literal('')),
})

export async function redeemRunTokenAction(_: FormState, formData: FormData): Promise<FormState> {
  const user = await requireAuthenticatedUser('/redeem')
  await ensureProfileRecord(user)

  const parsed = redeemSchema.safeParse({
    code: formData.get('code'),
  })

  if (!parsed.success) {
    return {
      success: false,
      message: 'Enter a valid run token.',
    }
  }

  const result = await redeemRunTokenForUser({
    userId: user.id,
    email: user.email ?? null,
    code: parsed.data.code,
  })

  revalidatePath('/')
  revalidatePath('/redeem')
  revalidatePath('/account')

  return {
    success: result.success,
    message: result.success
      ? `${result.message} ${result.remainingRuns} run token(s) ready.`
      : result.message,
  }
}

export async function generateRunTokensAction(_: FormState, formData: FormData): Promise<FormState> {
  const adminUser = await requireAdminUser('/admin/tokens')

  const parsed = generateSchema.safeParse({
    quantity: formData.get('quantity'),
    assignedEmail: formData.get('assignedEmail'),
    note: formData.get('note'),
    expiresAt: formData.get('expiresAt'),
  })

  if (!parsed.success) {
    return {
      success: false,
      message: 'Check the quantity, email, and expiry fields.',
    }
  }

  const assignedEmail = parsed.data.assignedEmail || null
  if (assignedEmail && !z.email().safeParse(assignedEmail).success) {
    return {
      success: false,
      message: 'Assigned email must be a valid email address.',
    }
  }

  const expiresAt = parsed.data.expiresAt ? new Date(parsed.data.expiresAt) : null
  if (expiresAt && Number.isNaN(expiresAt.getTime())) {
    return {
      success: false,
      message: 'Expiry must be a valid date and time.',
    }
  }

  const result = await createRunTokenBatch({
    createdBy: adminUser.id,
    quantity: parsed.data.quantity,
    assignedEmail,
    note: parsed.data.note || null,
    expiresAt: expiresAt ? expiresAt.toISOString() : null,
  })

  revalidatePath('/admin/tokens')

  return {
    success: true,
    message: `Generated ${result.codes.length} run token(s). Copy them now; plaintext codes are not stored.`,
    codes: result.codes,
  }
}

export async function revokeRunTokenAction(formData: FormData) {
  const adminUser = await requireAdminUser('/admin/tokens')
  const tokenId = formData.get('tokenId')

  if (typeof tokenId !== 'string' || !tokenId) {
    throw new Error('Missing token id')
  }

  await revokeRunToken({
    actorUserId: adminUser.id,
    tokenId,
  })

  revalidatePath('/admin/tokens')
}

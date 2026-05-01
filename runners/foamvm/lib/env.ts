function requireServerEnv(name: string): string {
  const value = process.env[name]
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`)
  }
  return value
}

export function getSupabaseUrl(): string {
  const value = process.env.NEXT_PUBLIC_SUPABASE_URL
  if (!value) {
    throw new Error('Missing required environment variable: NEXT_PUBLIC_SUPABASE_URL')
  }
  return value
}

export function getSupabaseAnonKey(): string {
  const value = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
  if (!value) {
    throw new Error('Missing required environment variable: NEXT_PUBLIC_SUPABASE_ANON_KEY')
  }
  return value
}

export function getSupabaseServiceRoleKey(): string {
  return requireServerEnv('SUPABASE_SERVICE_ROLE_KEY')
}

export function getOutputBucketName(): string {
  return process.env.SUPABASE_OUTPUT_BUCKET || 'run-outputs'
}

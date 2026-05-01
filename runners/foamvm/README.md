# PhysicsOS foamvm runner

This copied runner is the PhysicsOS E2B/OpenFOAM external service runner.

It keeps the original E2B sandbox model, but PhysicsOS should call the
structured endpoint instead of the original prompt-based CFD UI:

```text
POST /api/physicsos/jobs
Authorization: Bearer $PHYSICSOS_RUNNER_TOKEN
Content-Type: application/json

{
  "schema_version": "physicsos.full_solver_job.v1",
  "problem_id": "problem:openfoam-smoke",
  "backend": "openfoam",
  "backend_command": "icoFoam",
  "openfoam": {
    "solver": "icoFoam"
  }
}
```

The same endpoint also accepts mesh-conversion manifests prepared by
PhysicsOS:

```text
{
  "schema_version": "physicsos.mesh_conversion_job.v1",
  "job_type": "mesh_conversion",
  "backend": "openfoam",
  "inputs": {
    "source_mesh_file": {
      "path": "mesh.msh",
      "content_base64": "..."
    },
    "boundary_exports": []
  }
}
```

For mesh conversion jobs, the runner decodes the inline `.msh`, writes
`boundary_mapping.json`, and executes the backend-specific converter in E2B.
OpenFOAM uses `gmshToFoam`; SU2/FEniCSx use `meshio` when available in the
template.

If `openfoam.case_files` is omitted, the runner executes the built-in
OpenFOAM cavity smoke case from `$FOAM_TUTORIALS`. Output files are returned
as base64 artifacts in the JSON response.

The original prompt-based `/api/cfd` flow remains copied for reference, but
PhysicsOS integration should use `/api/physicsos/jobs`.

## Local smoke test

```bash
cd runners/foamvm
npm install
npm run build
npm run dev
```

Then POST a manifest to `http://localhost:3000/api/physicsos/jobs`.

Required environment variables:

```text
E2B_API_KEY=...
E2B_TEMPLATE_ID=...
PHYSICSOS_RUNNER_TOKEN=...
```

---

# Original SciMate notes

SciMate is a Next.js 16 app for running CFD jobs through E2B sandboxes, gated by Supabase authentication and one-time run tokens.

## Stack

- Next.js 16 App Router
- Supabase Auth + Postgres + Storage
- E2B sandboxes for long-running CFD execution
- Vercel for deployment

## Local setup

1. Copy `.env.example` to `.env.local`.
2. Fill in the Supabase, E2B, and Anthropic environment variables.
3. In Supabase SQL Editor, run [`supabase/schema.sql`](./supabase/schema.sql).
4. Mark at least one user as admin:

```sql
update public.profiles
set role = 'admin'
where email = 'you@example.com';
```

5. Install dependencies and start development:

```bash
npm install
npm run dev
```

## Run-token flow

- Users sign in with Supabase passwordless email codes. This avoids the browser confusion that magic links can cause.
- Admins generate random run tokens from `/admin/tokens`.
- A user redeems a token on `/redeem`.
- Each call to `/api/cfd` consumes exactly one redeemed token before starting the sandbox.
- Output files are persisted to Supabase Storage and downloaded through the protected `/api/files` route.

## Vercel deployment

1. Create a Vercel project from this repository.
2. Connect the same Supabase project you used locally.
3. Add all variables from `.env.example` to Vercel Project Settings.
4. Run the SQL in [`supabase/schema.sql`](./supabase/schema.sql) on production.
5. In Supabase Auth settings, set the site URL and redirect URL to:

```text
https://your-domain.com/auth/confirm
```

6. In the Supabase magic-link email template, include `{{ .Token }}` so users can complete login by typing the code into `/auth/login`.

## Important note

The existing `.env.local` in this workspace currently contains real secrets. Move them into Vercel environment variables and rotate them before publishing the project.

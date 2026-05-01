create extension if not exists pgcrypto;

create table if not exists public.profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  email text,
  role text not null default 'user' check (role in ('admin', 'user')),
  created_at timestamptz not null default now()
);

create table if not exists public.invite_batches (
  id uuid primary key default gen_random_uuid(),
  created_by uuid not null references public.profiles (id) on delete restrict,
  quantity integer not null check (quantity > 0),
  note text,
  created_at timestamptz not null default now()
);

create table if not exists public.run_tokens (
  id uuid primary key default gen_random_uuid(),
  batch_id uuid references public.invite_batches (id) on delete set null,
  code_hash text not null unique,
  assigned_email text,
  created_by uuid not null references public.profiles (id) on delete restrict,
  redeemed_by uuid references public.profiles (id) on delete set null,
  status text not null default 'unused' check (status in ('unused', 'redeemed', 'consumed', 'revoked')),
  note text,
  redeemed_at timestamptz,
  consumed_at timestamptz,
  expires_at timestamptz,
  created_at timestamptz not null default now()
);

create index if not exists run_tokens_redeemed_by_status_idx
  on public.run_tokens (redeemed_by, status, redeemed_at);

create table if not exists public.run_consumptions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.profiles (id) on delete cascade,
  run_token_id uuid not null references public.run_tokens (id) on delete restrict,
  prompt_excerpt text,
  sandbox_session_id text,
  command_pid bigint,
  status text not null default 'starting' check (status in ('starting', 'running', 'completed', 'failed')),
  error_message text,
  created_at timestamptz not null default now()
);

create table if not exists public.run_events (
  id bigint generated always as identity primary key,
  consumption_id uuid not null references public.run_consumptions (id) on delete cascade,
  payload jsonb not null,
  created_at timestamptz not null default now()
);

create index if not exists run_events_consumption_id_id_idx
  on public.run_events (consumption_id, id);

create table if not exists public.run_output_files (
  id uuid primary key default gen_random_uuid(),
  consumption_id uuid not null references public.run_consumptions (id) on delete cascade,
  filename text not null,
  storage_path text not null unique,
  content_type text,
  size_bytes bigint not null check (size_bytes >= 0),
  is_image boolean not null default false,
  created_at timestamptz not null default now()
);

create table if not exists public.admin_audit_logs (
  id uuid primary key default gen_random_uuid(),
  actor_user_id uuid not null references public.profiles (id) on delete restrict,
  action text not null,
  details jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id, email)
  values (new.id, new.email)
  on conflict (id) do update set email = excluded.email;

  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
after insert on auth.users
for each row execute function public.handle_new_user();

create or replace function public.redeem_run_token(
  p_user_id uuid,
  p_user_email text,
  p_code_hash text
)
returns table (
  success boolean,
  message text,
  remaining_runs integer
)
language plpgsql
security definer
set search_path = public
as $$
declare
  v_token public.run_tokens%rowtype;
begin
  select *
  into v_token
  from public.run_tokens
  where code_hash = p_code_hash
  for update;

  if not found then
    return query select false, 'Invalid token code.', 0;
    return;
  end if;

  if v_token.status <> 'unused' then
    return query select false, 'This token has already been used.', 0;
    return;
  end if;

  if v_token.expires_at is not null and v_token.expires_at < now() then
    return query select false, 'This token has expired.', 0;
    return;
  end if;

  if v_token.assigned_email is not null
     and lower(v_token.assigned_email) <> lower(coalesce(p_user_email, '')) then
    return query select false, 'This token is assigned to a different email address.', 0;
    return;
  end if;

  update public.run_tokens
  set status = 'redeemed',
      redeemed_by = p_user_id,
      redeemed_at = now()
  where id = v_token.id
    and status = 'unused';

  return query
  select true, 'Run token redeemed successfully.', count(*)::integer
  from public.run_tokens
  where redeemed_by = p_user_id
    and status = 'redeemed';
end;
$$;

create or replace function public.consume_redeemed_run_token(
  p_user_id uuid,
  p_prompt_excerpt text default null
)
returns table (
  success boolean,
  message text,
  remaining_runs integer,
  token_id uuid,
  consumption_id uuid
)
language plpgsql
security definer
set search_path = public
as $$
declare
  v_token public.run_tokens%rowtype;
  v_consumption_id uuid;
begin
  select *
  into v_token
  from public.run_tokens
  where redeemed_by = p_user_id
    and status = 'redeemed'
  order by redeemed_at asc nulls first, created_at asc
  for update skip locked
  limit 1;

  if not found then
    return query select false, 'No redeemed run tokens are available.', 0, null::uuid, null::uuid;
    return;
  end if;

  update public.run_tokens
  set status = 'consumed',
      consumed_at = now()
  where id = v_token.id;

  insert into public.run_consumptions (user_id, run_token_id, prompt_excerpt, status)
  values (p_user_id, v_token.id, p_prompt_excerpt, 'starting')
  returning id into v_consumption_id;

  return query
  select true, 'Run token consumed.', count(*)::integer, v_token.id, v_consumption_id
  from public.run_tokens
  where redeemed_by = p_user_id
    and status = 'redeemed';
end;
$$;

alter table public.profiles enable row level security;
alter table public.run_tokens enable row level security;
alter table public.run_consumptions enable row level security;
alter table public.run_events enable row level security;
alter table public.run_output_files enable row level security;

create policy "profiles_select_own"
on public.profiles
for select
to authenticated
using (auth.uid() = id);

create policy "run_tokens_select_own"
on public.run_tokens
for select
to authenticated
using (redeemed_by = auth.uid());

create policy "run_consumptions_select_own"
on public.run_consumptions
for select
to authenticated
using (user_id = auth.uid());

create policy "run_output_files_select_own"
on public.run_output_files
for select
to authenticated
using (
  exists (
    select 1
    from public.run_consumptions
    where public.run_consumptions.id = public.run_output_files.consumption_id
      and public.run_consumptions.user_id = auth.uid()
  )
);

create policy "run_events_select_own"
on public.run_events
for select
to authenticated
using (
  exists (
    select 1
    from public.run_consumptions
    where public.run_consumptions.id = public.run_events.consumption_id
      and public.run_consumptions.user_id = auth.uid()
  )
);

insert into storage.buckets (id, name, public)
values ('run-outputs', 'run-outputs', false)
on conflict (id) do nothing;

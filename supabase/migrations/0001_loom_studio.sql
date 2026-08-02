-- Loom Studio: programs, builds, build_events.
-- Showcase rows (user_id null) are world-readable; user rows are owner-only.

create table if not exists public.programs (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid references auth.users (id) on delete cascade,
  name        text not null,
  source      text not null,
  created_at  timestamptz not null default now()
);
create index if not exists programs_user_idx on public.programs (user_id);
alter table public.programs enable row level security;
drop policy if exists "programs owner all" on public.programs;
create policy "programs owner all" on public.programs
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
drop policy if exists "programs showcase read" on public.programs;
create policy "programs showcase read" on public.programs
  for select using (user_id is null);

create table if not exists public.builds (
  id           uuid primary key default gen_random_uuid(),
  program_id   uuid not null references public.programs (id) on delete cascade,
  user_id      uuid references auth.users (id) on delete cascade,
  target_model text not null,
  status       text not null default 'submitted'
    check (status in ('submitted','queued','running','passed','failed','error')),
  report       jsonb,
  hf_repo      text,
  created_at   timestamptz not null default now()
);
create index if not exists builds_program_idx on public.builds (program_id);
create index if not exists builds_status_idx on public.builds (status);
alter table public.builds enable row level security;
drop policy if exists "builds owner all" on public.builds;
create policy "builds owner all" on public.builds
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
drop policy if exists "builds showcase read" on public.builds;
create policy "builds showcase read" on public.builds
  for select using (user_id is null);

create table if not exists public.build_events (
  id        uuid primary key default gen_random_uuid(),
  build_id  uuid not null references public.builds (id) on delete cascade,
  seq       integer not null,
  stage     text not null,
  payload   jsonb not null default '{}'::jsonb,
  ts        timestamptz not null default now()
);
create index if not exists build_events_build_idx
  on public.build_events (build_id, seq);
alter table public.build_events enable row level security;
drop policy if exists "build_events readable with build" on public.build_events;
create policy "build_events readable with build" on public.build_events
  for select using (
    exists (select 1 from public.builds b where b.id = build_events.build_id
            and (b.user_id = auth.uid() or b.user_id is null)));

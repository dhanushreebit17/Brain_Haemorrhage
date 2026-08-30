-- ============================================================================
-- Brain Hemorrhage Detection System — Supabase schema
-- Run this in: Supabase Dashboard -> SQL Editor -> New query -> Run
-- ============================================================================

-- 1. Table: scans
create table if not exists public.scans (
  id uuid primary key default gen_random_uuid(),
  patient_id text not null,
  scan_date date not null default current_date,
  hemorrhage_detected boolean not null default false,
  hemorrhage_type text,                 -- epidural / intracerebral / intraventricular / subarachnoid / subdural
  confidence numeric(5,4),              -- 0.0000 - 1.0000
  ct_scan_url text,
  heatmap_url text,
  report_generated boolean not null default false,
  created_by uuid references auth.users(id),
  created_at timestamptz not null default now()
);

create index if not exists scans_patient_id_idx on public.scans (patient_id);
create index if not exists scans_created_at_idx on public.scans (created_at desc);

-- 2. Row Level Security
-- This app is an internal single-role (radiologist) tool: any authenticated
-- user can read every scan and create new ones. Tighten this later if you
-- introduce multiple roles or per-hospital data isolation.
alter table public.scans enable row level security;

create policy "Authenticated users can read all scans"
  on public.scans for select
  to authenticated
  using (true);

create policy "Authenticated users can insert scans"
  on public.scans for insert
  to authenticated
  with check (auth.uid() = created_by);

create policy "Authenticated users can update scans"
  on public.scans for update
  to authenticated
  using (true);

-- 3. Storage buckets
-- Create these in Dashboard -> Storage -> New bucket (or via SQL below).
-- Public = true so <img src> tags can load them directly for this demo.
-- For production, make them private and serve via signed URLs instead.
insert into storage.buckets (id, name, public)
values ('ct-scans', 'ct-scans', true)
on conflict (id) do nothing;

insert into storage.buckets (id, name, public)
values ('heatmaps', 'heatmaps', true)
on conflict (id) do nothing;

create policy "Authenticated users can upload CT scans"
  on storage.objects for insert
  to authenticated
  with check (bucket_id = 'ct-scans');

create policy "Anyone can view CT scans"
  on storage.objects for select
  using (bucket_id = 'ct-scans');

create policy "Authenticated users can upload heatmaps"
  on storage.objects for insert
  to authenticated
  with check (bucket_id = 'heatmaps');

create policy "Anyone can view heatmaps"
  on storage.objects for select
  using (bucket_id = 'heatmaps');

-- 4. (Optional) create a radiologist login for testing, via
--    Dashboard -> Authentication -> Users -> Add user
--    Then use that email/password on the login page.

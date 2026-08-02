# Loom Studio architecture (from kundali + ACD recon, 2026-08-02)

## Template 1 — kundali (~/projects/kundali)
Next.js 14 App Router + TypeScript + Tailwind; Supabase via @supabase/ssr with magic-link
PKCE (middleware.ts refreshes session cookies; app/auth/callback/route.ts exchanges the
code). Server client in lib/supabase/server.ts, browser client in lib/supabase/client.ts.
RLS scoped to auth.uid(); child tables via FK joins. vercel.json maps /api/py/* to a
Python function. gb10-gateway/main.py: thin FastAPI auth proxy in front of a local model
server, run as a systemd user service, exposed via tunnel, validated by shared secret OR
Supabase JWT.

## Template 2 — ACD tunnel chain (~/projects/ActiveCIrcuitDiscovery)
Browser → Vercel route `app/api/dgx/[...path]/route.ts` (reads DGX_TUNNEL_URL env,
attaches X-ACD-Key, streams SSE without buffering) → permanent Cloudflare Worker
(dgx-server/worker.js; reads current tunnel URL from KV key `gateway_url`; maps tunnel
530/502/504 to clean 503 JSON) → cloudflared quick tunnel → FastAPI on the GB10
(dgx-server/server.py). Orchestrated by dgx-server/golive.sh: start backend, start
cloudflared, poll until reachable, `cf.sh set <url>` writes KV, verify the public chain.
`cf.sh clear` flips the public site to OFFLINE/replay gracefully. Vercel env never
changes; only KV does.

## Loom Studio (this repo, `studio/` + `worker/`)
- studio/: Next.js app — editor (program textarea + refusal-aware explain panel),
  builds list, build page with live event stream (SSE via Supabase realtime or the
  worker tunnel), results view rendering the report (gaps, searches, gates,
  expectations) and HF artifact link. Replay mode renders committed reports when the
  worker is offline, labelled REPLAY.
- worker/: FastAPI daemon on the GB10. Endpoints: /health, /explain (parse+plan+space,
  no GPU), /build (enqueue), internal loop executes builds one at a time via
  loom.app.build_open.build, writing build_events rows as stages complete, dispatching
  training-heavy stages to the RTX 5090 (rtx5090-connect pattern) when configured,
  pushing passing artifacts to HF (qbz506/loom-<app>-<family>), updating builds row.
- supabase/: migrations — programs(id, user_id nullable for anon showcase, name,
  source, created_at), builds(id, program_id, target_model, status
  submitted|queued|running|passed|failed|error, report jsonb, hf_repo, timings),
  build_events(build_id, seq, stage, payload jsonb, ts). RLS: owners all; anon SELECT
  on showcase rows (user_id is null).
- tunnel/: golive.sh + worker.js + cf.sh adapted from ACD (new KV namespace + worker
  name loom-studio-gw).

Security: programs are data, never executed as code — the worker only parses them with
the Loom parser (parse-time refusals returned as diagnostics); builds are rate-limited
(one at a time, queue depth cap); no secrets in the repo.

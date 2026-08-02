# Loom Studio — dynamic spec with gate closures

The contract for the web app. Gates are closed one at a time, in order, each with a
measurable check; a gate that cannot be measured is not a gate. This file is updated as
gates close (dynamic spec) — the current state is always the top table.

## Objective

A person opens a web page, writes a `.loom` program in consequences (knows / speaks /
always / never / refuses / expect), presses Build, and watches a real compiler bring a
real model into existence on real GPUs — gap measurements, searches, refusals,
escalations, gates — ending with a verified artifact published on Hugging Face. When
the GPUs are offline the page replays committed builds honestly, labelled as replay.

## Architecture (from kundali + ACD recon, docs/design/loom_studio_architecture.md)

Browser → Vercel (Next.js, `studio/`) → `/api/gpu/[...path]` proxy →
Cloudflare Worker (permanent URL, KV holds current tunnel) →
cloudflared quick tunnel → worker daemon on GB10 (FastAPI, port 8town) →
Loom compiler; training-heavy stages dispatched to the RTX 5090.
Supabase: programs, builds, build_events (SSE-streamed), artifacts, RLS by auth.uid()
with anonymous read of showcase builds. HF: adapter + report + source per passing
build, public under qbz506.

## Gates

| # | Gate | Check | Status |
|---|------|-------|--------|
| G1 | Four industrial programs compile end-to-end locally | 4 build reports, all expectations pass, committed under results/ | open |
| G2 | Artifacts on HF | 4+ public repos under qbz506 with adapter + report + .loom source; round-trip load verified | open |
| G3 | Studio scaffold runs locally | `npm run dev` serves editor; explain round-trips against worker | open |
| G4 | Supabase schema live | migrations applied; RLS verified (anon can read showcase, cannot write) | CLOSED 2026-08-02 — project lupnomulqaifhqcdwqsd, anon insert refused by RLS, 5 showcase programs seeded |
| G5 | Worker executes a queued build | row in builds → worker picks up → events stream → report row written | open |
| G6 | Tunnel chain up | Vercel /api/gpu/health returns GB10 health through Worker+KV | open |
| G7 | Deployed on Vercel | public URL serves; replay mode works with worker offline | open |
| G8 | Live build from the browser | a user-submitted program builds on GB10, streamed to the page, artifact on HF | open |
| G9 | 5090 in the loop | a training-heavy stage measurably dispatched to the RTX 5090 and results merged | open |
| G10 | Science ledger current | site + VALIDATION.md reflect final-compiler rebuilds of all families/domains | open |

## Non-negotiables (inherited from the project)

No fabricated numbers; no silent fallbacks; replay is labelled replay; a claim without
an artifact renders as not-yet-measured; user-submitted programs run under the same
refusal rules as the examples (parse-time refusals surface in the editor).

# Loom Studio

A Next.js frontend for writing, compiling, and verifying LLM programs in Loom — a declarative language for programming LLM behavior.

## Getting Started

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Environment Variables

Set these in `.env.local` (copy from `.env.example`):

- `LOOM_GPU_URL`: Base URL to the GPU worker (e.g., a Cloudflare Worker URL). Required for live builds.
- `LOOM_GPU_KEY`: Authentication header value for X-Loom-Key. Required if LOOM_GPU_URL is set.
- `NEXT_PUBLIC_SUPABASE_URL`: Supabase project URL (optional; builds list uses replay mode if not set).
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`: Supabase anonymous key (optional).

If no GPU URL is configured, the app runs in **REPLAY mode**: it displays committed build reports from the repository as read-only examples.

## Pages

- **Home** (`/`): Landing page with objective and example programs.
- **Editor** (`/studio`): Write a Loom program, select an example, explain the compiler's search plan, or start a build.
- **Builds** (`/builds`): List committed builds (replay mode) or live builds from Supabase (if configured).
- **Build Detail** (`/builds/[id]`): View full report: capabilities, expectations, margins, and evidence.

## Architecture

### Proxy

`app/api/gpu/[...path]/route.ts` proxies POST `/explain` and `/build` requests to the GPU worker, attaching the X-Loom-Key header. Translates fetch failures into 503 `{offline: true}` responses.

### Examples

`scripts/embed-examples.mjs` runs at prebuild time: it reads `.loom` files from `../examples/` and embeds them as JSON in `lib/examples.json`. The `/api/examples` route serves them.

### Showcase

The same script also reads committed build reports (e.g., `../results/loom_clinic_build_*.json`) and embeds them as JSON in `lib/showcase.json`. Builds are assigned replay IDs (`replay-0`, `replay-1`, etc.) so they can be fetched and rendered without live polling.

## Styling

Calm paper-like design with off-white backgrounds, serif headings, and deep navy accents. Tailwind CSS + custom color palette matching the project site.

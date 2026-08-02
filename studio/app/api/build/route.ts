import { NextRequest } from "next/server";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

// Submit a build durably: create the Supabase row first (so the build has a public
// record and an event stream), then hand the same id to the GPU worker. If Supabase
// is not configured the worker still builds — the id is just not durable.
export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => null);
  if (!body?.source) {
    return Response.json({ error: "source required" }, { status: 422 });
  }
  const target = body.target ?? "meta-llama/Llama-3.2-1B-Instruct";
  const name = (body.name ?? "untitled").slice(0, 80);

  const sbUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const sbKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  let buildId: string | null = null;
  if (sbUrl && sbKey) {
    const r = await fetch(`${sbUrl}/rest/v1/rpc/rpc_submit_build`, {
      method: "POST",
      headers: {
        apikey: sbKey,
        Authorization: `Bearer ${sbKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ p_name: name, p_source: body.source, p_target: target }),
    });
    if (r.ok) buildId = (await r.json()) as string;
    else {
      const detail = await r.text();
      return Response.json({ error: "refused", detail }, { status: 429 });
    }
  }

  const gpu = process.env.LOOM_GPU_URL;
  if (!gpu) {
    return Response.json({ offline: true, detail: "LOOM_GPU_URL not configured" },
      { status: 503 });
  }
  const upstream = await fetch(`${gpu.replace(/\/$/, "")}/build`, {
    method: "POST",
    headers: {
      "X-Loom-Key": process.env.LOOM_GPU_KEY ?? "",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ source: body.source, target, build_id: buildId }),
    signal: AbortSignal.timeout(30_000),
  }).catch(() => null);
  if (!upstream) {
    return Response.json({ offline: true, detail: "GPU worker unreachable" },
      { status: 503 });
  }
  const data = await upstream.json().catch(() => ({}));
  return Response.json({ ...data, build_id: buildId ?? data.build_id },
    { status: upstream.status });
}

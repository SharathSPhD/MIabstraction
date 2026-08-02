import { NextRequest } from "next/server";

export const dynamic = "force-dynamic";
export const maxDuration = 120;

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => null);
  if (!body?.artifact || !body?.message) {
    return Response.json(
      { error: "artifact and message required" },
      { status: 422 }
    );
  }

  const gpu = process.env.LOOM_GPU_URL;
  if (!gpu) {
    return Response.json(
      { offline: true, detail: "LOOM_GPU_URL not configured" },
      { status: 503 }
    );
  }

  try {
    const upstream = await fetch(
      `${gpu.replace(/\/$/, "")}/chat`,
      {
        method: "POST",
        headers: {
          "X-Loom-Key": process.env.LOOM_GPU_KEY ?? "",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ artifact: body.artifact, message: body.message }),
        signal: AbortSignal.timeout(90_000),
      }
    );

    if (!upstream.ok) {
      // Unwrap the worker's JSON rather than passing its body through as a string:
      // the page was rendering {"detail":"..."} verbatim at the user.
      const raw = await upstream.text();
      let detail = raw;
      try {
        const parsed = JSON.parse(raw);
        detail = parsed?.detail ?? parsed?.error ?? raw;
      } catch {
        /* not JSON: show it as-is */
      }
      return Response.json(
        { offline: true, detail },
        { status: upstream.status }
      );
    }

    const data = await upstream.json();
    return Response.json(data);
  } catch (e) {
    return Response.json(
      { offline: true, detail: "GPU worker unreachable or timed out" },
      { status: 503 }
    );
  }
}

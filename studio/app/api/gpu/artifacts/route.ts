import { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const gpu = process.env.LOOM_GPU_URL;
  if (!gpu) {
    return Response.json(
      { offline: true, detail: "LOOM_GPU_URL not configured" },
      { status: 503 }
    );
  }

  try {
    const upstream = await fetch(
      `${gpu.replace(/\/$/, "")}/artifacts`,
      {
        headers: {
          "X-Loom-Key": process.env.LOOM_GPU_KEY ?? "",
        },
        signal: AbortSignal.timeout(10_000),
        cache: "no-store" as RequestCache,
      }
    );

    if (!upstream.ok) {
      return Response.json(
        { offline: true, detail: "The GPU is busy or briefly offline — models will be listed again in a moment." },
        { status: upstream.status }
      );
    }

    const data = await upstream.json();
    return Response.json(data);
  } catch (e) {
    return Response.json(
      { offline: true, detail: "GPU worker unreachable" },
      { status: 503 }
    );
  }
}

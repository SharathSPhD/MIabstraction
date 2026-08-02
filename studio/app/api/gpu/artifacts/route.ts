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
      }
    );

    if (!upstream.ok) {
      return Response.json(
        { offline: true, detail: "GPU worker returned error" },
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

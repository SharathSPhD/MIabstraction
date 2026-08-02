import type { BuildResponse, BuildStatus, ExplainResponse } from "./types";

export async function explainProgram(
  source: string
): Promise<{ ok: boolean; text?: string; error?: string }> {
  try {
    const response = await fetch("/api/gpu/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source }),
    });

    const data = (await response.json()) as ExplainResponse;

    if (!response.ok) {
      return { ok: false, error: data.detail || "Request failed" };
    }

    return { ok: data.ok, text: data.explain };
  } catch (e) {
    return { ok: false, error: String(e) };
  }
}

export async function buildProgram(
  source: string,
  target: string,
  token?: string
): Promise<{ ok: boolean; id?: string; error?: string }> {
  try {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }

    const response = await fetch("/api/build", {
      method: "POST",
      headers,
      body: JSON.stringify({ source, target }),
    });

    const data = (await response.json()) as BuildResponse;

    if (!response.ok) {
      return { ok: false, error: data.detail || "Build request failed" };
    }

    return { ok: data.ok, id: data.build_id };
  } catch (e) {
    return { ok: false, error: String(e) };
  }
}

export async function getBuildStatus(id: string): Promise<BuildStatus> {
  const response = await fetch(`/api/gpu/build/${id}`);
  if (!response.ok) {
    return {
      build_id: id,
      status: "failed",
      report_ready: false,
    };
  }
  return (await response.json()) as BuildStatus;
}

export async function getBuildReport(id: string) {
  const response = await fetch(`/api/gpu/build/${id}/report`);
  if (!response.ok) {
    return null;
  }
  return await response.json();
}

"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Loader, CheckCircle2, XCircle } from "lucide-react";
import type { BuildReport, Capability, Expectation } from "@/lib/types";
import { getBuildStatus, getBuildReport } from "@/lib/gpu";

async function getShowcase() {
  try {
    const res = await fetch("/api/showcase");
    return (await res.json()) as BuildReport[];
  } catch {
    return [];
  }
}

function MarginBar({
  before,
  after,
}: {
  before?: number;
  after?: number;
}) {
  if (before === undefined || after === undefined) return null;
  const maxVal = Math.max(Math.abs(before), Math.abs(after), 0.1);
  const beforePct = (before / maxVal) * 50 + 50;
  const afterPct = (after / maxVal) * 50 + 50;

  return (
    <div className="flex gap-2 items-center text-xs">
      <div className="flex-1 bg-panel h-6 border border-hairline border-gray-300 rounded overflow-hidden flex items-center">
        <div
          className="h-full bg-accent"
          style={{ width: `${Math.max(0, beforePct)}%` }}
        />
      </div>
      <div className="flex-1 bg-panel h-6 border border-hairline border-gray-300 rounded overflow-hidden flex items-center">
        <div
          className="h-full bg-verified"
          style={{ width: `${Math.max(0, afterPct)}%` }}
        />
      </div>
    </div>
  );
}

export default function BuildDetailPage() {
  const params = useParams();
  const id = params.id as string;
  const [report, setReport] = useState<BuildReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [isReplay, setIsReplay] = useState(false);
  const [pollingActive, setPollingActive] = useState(false);

  useEffect(() => {
    const loadBuild = async () => {
      // Check if it's a replay build
      if (id?.startsWith("replay-")) {
        const showcase = await getShowcase();
        const index = parseInt(id.split("-")[1], 10);
        if (showcase[index]) {
          setReport({ ...showcase[index], id });
          setIsReplay(true);
          setLoading(false);
          return;
        }
      }

      // Try to fetch live build
      setPollingActive(true);
      let attempts = 0;
      const maxAttempts = 60; // Poll for up to 3 minutes

      const poll = async () => {
        attempts++;
        try {
          const status = await getBuildStatus(id);

          if (status.status === "completed" && status.report_ready) {
            const fullReport = await getBuildReport(id);
            if (fullReport) {
              setReport({ ...fullReport, id });
              setPollingActive(false);
              return;
            }
          } else if (status.status === "failed") {
            setError("Build failed");
            setPollingActive(false);
            return;
          }

          if (attempts < maxAttempts) {
            setTimeout(poll, 3000);
          } else {
            setError("Build is taking too long");
            setPollingActive(false);
          }
        } catch (e) {
          if (attempts < maxAttempts) {
            setTimeout(poll, 3000);
          } else {
            setError(String(e));
            setPollingActive(false);
          }
        }
      };

      setLoading(false);
      poll();
    };

    loadBuild();
  }, [id]);

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-12">
        <h1 className="font-serif text-4xl font-bold mb-8">Build Details</h1>
        <div className="card text-center py-12">
          <Loader className="w-8 h-8 animate-spin mx-auto mb-4" />
          <p className="text-body">Loading build...</p>
        </div>
      </div>
    );
  }

  if (error && !pollingActive) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-12">
        <h1 className="font-serif text-4xl font-bold mb-8">Build Details</h1>
        <div className="card bg-red-50 border-l-2 border-red-400 py-6">
          <div className="flex gap-3">
            <XCircle className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-red-900 mb-1">Error</p>
              <p className="text-red-800">{error}</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-12">
        <h1 className="font-serif text-4xl font-bold mb-8">Build Details</h1>
        {pollingActive ? (
          <div className="card text-center py-12">
            <Loader className="w-8 h-8 animate-spin mx-auto mb-4" />
            <p className="text-body">Waiting for build to complete...</p>
          </div>
        ) : (
          <div className="card text-center py-12">
            <p className="text-muted">Build not found</p>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-6 py-12">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-start justify-between gap-4 mb-4">
          <div>
            <h1 className="font-serif text-4xl font-bold mb-2">{report.app}</h1>
            <p className="text-body font-mono text-sm">
              {report.base_model}
            </p>
          </div>
          <div className="flex-shrink-0">
            {isReplay ? (
              <span className="badge-replay">REPLAY</span>
            ) : (
              <span className="badge-live">LIVE</span>
            )}
          </div>
        </div>

        {pollingActive && (
          <div className="bg-blue-50 border-l-2 border-blue-400 p-4 flex gap-2 items-center">
            <Loader className="w-4 h-4 animate-spin" />
            <p className="text-sm text-blue-900">
              Build is running... polling for updates
            </p>
          </div>
        )}
      </div>

      <div className="divider" />

      {/* Header stats */}
      <div className="grid md:grid-cols-4 gap-4 mb-12">
        <div className="stat-card">
          <p className="stat-label">Result</p>
          <div className="flex items-center gap-2">
            {report.passed ? (
              <CheckCircle2 className="w-6 h-6 text-verified" />
            ) : (
              <XCircle className="w-6 h-6 text-red-600" />
            )}
            <span className="font-mono font-bold">
              {report.passed ? "PASS" : "FAIL"}
            </span>
          </div>
        </div>

        <div className="stat-card">
          <p className="stat-label">Wall Clock</p>
          <p className="stat-value">{report.wall_clock_s.toFixed(1)}s</p>
        </div>

        <div className="stat-card">
          <p className="stat-label">Expectations</p>
          <p className="stat-value">
            {report.expectations_passed || 0}/{report.expectations.length}
          </p>
        </div>

        <div className="stat-card">
          <p className="stat-label">Verified Against</p>
          <p className="stat-value">
            {report.verified_against_recitation_of || 0} samples
          </p>
        </div>
      </div>

      {/* Capabilities */}
      {report.capabilities.length > 0 && (
        <>
          <h2 className="section-heading">Capabilities</h2>
          <div className="space-y-4 mb-12">
            {report.capabilities.map((cap, idx) => (
              <div key={idx} className="card space-y-4">
                <div>
                  <h3 className="font-serif font-bold mb-1">{cap.capability}</h3>
                  <p className="text-xs font-mono uppercase text-muted">
                    {cap.kind}
                  </p>
                </div>

                <div>
                  <p className="stat-label">Strategy</p>
                  <p className="font-mono text-sm">{cap.strategy}</p>
                </div>

                {cap.execution?.autotune && (
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <p className="stat-label">Target Met</p>
                      <div className="flex items-center gap-2">
                        {cap.execution.autotune.target_met ? (
                          <CheckCircle2 className="w-5 h-5 text-verified" />
                        ) : (
                          <XCircle className="w-5 h-5 text-red-600" />
                        )}
                        <span className="text-sm font-mono">
                          {cap.execution.autotune.target_met ? "Yes" : "No"}
                        </span>
                      </div>
                    </div>

                    {cap.execution.autotune.scale?.gap !== undefined && (
                      <div>
                        <p className="stat-label">Gap</p>
                        <p className="font-mono font-bold">
                          {cap.execution.autotune.scale.gap.toFixed(4)}
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {cap.behavioural_gate?.result && (
                  <div>
                    <p className="stat-label mb-3">Margin Before → After</p>
                    <MarginBar
                      before={cap.behavioural_gate.result.margin_before}
                      after={cap.behavioural_gate.result.margin_after}
                    />
                    <div className="grid md:grid-cols-2 gap-4 mt-3 text-xs">
                      {cap.behavioural_gate.result.margin_before !== undefined && (
                        <div>
                          <p className="text-muted">Before</p>
                          <p className="font-mono font-bold">
                            {cap.behavioural_gate.result.margin_before.toFixed(4)}
                          </p>
                        </div>
                      )}
                      {cap.behavioural_gate.result.margin_after !== undefined && (
                        <div>
                          <p className="text-muted">After</p>
                          <p className="font-mono font-bold">
                            {cap.behavioural_gate.result.margin_after.toFixed(4)}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      {/* Expectations */}
      {report.expectations.length > 0 && (
        <>
          <h2 className="section-heading">Expectations</h2>
          <div className="space-y-4 mb-12">
            {report.expectations.map((exp, idx) => (
              <div key={idx} className="card space-y-3">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h3 className="font-serif font-bold">{exp.expectation}</h3>
                    <p className="text-xs font-mono uppercase text-muted mt-1">
                      {exp.kind}
                    </p>
                  </div>
                  {exp.passed ? (
                    <span className="badge-pass flex-shrink-0">PASS</span>
                  ) : (
                    <span className="badge-fail flex-shrink-0">FAIL</span>
                  )}
                </div>

                <div>
                  <p className="stat-label">Evidence</p>
                  <div className="bg-panel border border-hairline border-gray-300 p-3 font-mono text-xs overflow-x-auto">
                    {exp.evidence}
                  </div>
                </div>

                {exp.detail && (
                  <div>
                    <p className="stat-label">Detail</p>
                    <p className="text-sm text-body">{exp.detail}</p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      {/* Navigation */}
      <div className="flex gap-3">
        <Link href="/builds" className="button-secondary">
          ← Back to Builds
        </Link>
        <Link href="/" className="button-secondary">
          Home
        </Link>
      </div>
    </div>
  );
}

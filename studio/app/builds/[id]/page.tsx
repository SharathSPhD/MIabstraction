"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Loader, CheckCircle2, XCircle, AlertCircle, Home } from "lucide-react";
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
        <h2 className="font-serif text-3xl font-bold mb-8">Build Details</h2>
        <div className="card text-center py-12">
          <Loader className="w-8 h-8 animate-spin mx-auto mb-4" />
          <p>Loading build...</p>
        </div>
      </div>
    );
  }

  if (error && !pollingActive) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-12">
        <h2 className="font-serif text-3xl font-bold mb-8">Build Details</h2>
        <div className="card bg-red-50 border-l-4 border-red-400 py-8">
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
        <h2 className="font-serif text-3xl font-bold mb-8">Build Details</h2>
        {pollingActive ? (
          <div className="card text-center py-12">
            <Loader className="w-8 h-8 animate-spin mx-auto mb-4" />
            <p>Waiting for build to complete...</p>
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
      <div className="flex items-center justify-between mb-8">
        <div>
          <h2 className="font-serif text-3xl font-bold mb-2">{report.app}</h2>
          <p className="text-body">{report.base_model}</p>
        </div>
        <div className="text-right">
          {isReplay && (
            <div className="bg-blue-50 border-l-4 border-blue-400 p-3 rounded mb-4">
              <p className="text-xs font-semibold text-blue-900">
                REPLAY — a committed build, not a live run
              </p>
            </div>
          )}
          {!isReplay && (
            <div className="bg-green-50 border-l-4 border-green-400 p-3 rounded mb-4">
              <p className="text-xs font-semibold text-green-900">LIVE</p>
            </div>
          )}
        </div>
      </div>

      {pollingActive && (
        <div className="mb-6 bg-blue-50 border-l-4 border-blue-400 p-4 rounded">
          <div className="flex gap-2 items-center">
            <Loader className="w-4 h-4 animate-spin" />
            <p className="text-sm text-blue-900">Build is running... polling for updates</p>
          </div>
        </div>
      )}

      {/* Header stats */}
      <div className="grid md:grid-cols-4 gap-4 mb-8">
        <div className="card">
          <p className="text-xs text-muted mb-1">Result</p>
          <div className="flex items-center gap-2">
            {report.passed ? (
              <CheckCircle2 className="w-6 h-6 text-green-600" />
            ) : (
              <XCircle className="w-6 h-6 text-red-600" />
            )}
            <span className="font-semibold">
              {report.passed ? "PASS" : "FAIL"}
            </span>
          </div>
        </div>

        <div className="card">
          <p className="text-xs text-muted mb-1">Wall Clock</p>
          <p className="font-semibold">{report.wall_clock_s.toFixed(1)}s</p>
        </div>

        <div className="card">
          <p className="text-xs text-muted mb-1">Expectations</p>
          <p className="font-semibold">
            {report.expectations_passed || 0}/{report.expectations.length}
          </p>
        </div>

        <div className="card">
          <p className="text-xs text-muted mb-1">Control Quality</p>
          <p className="font-semibold">
            {report.verified_against_recitation_of || 0} samples
          </p>
        </div>
      </div>

      {/* Capabilities */}
      <section className="mb-8">
        <h3 className="font-serif text-2xl font-bold mb-6">Capabilities</h3>
        <div className="space-y-4">
          {report.capabilities.map((cap, idx) => (
            <div key={idx} className="card">
              <div className="mb-4">
                <h4 className="font-semibold mb-1">{cap.capability}</h4>
                <p className="text-sm text-body capitalize">{cap.kind}</p>
              </div>

              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-xs text-muted mb-1">Strategy</p>
                  <p className="font-mono">{cap.strategy}</p>
                </div>

                {cap.execution?.autotune && (
                  <>
                    <div>
                      <p className="text-xs text-muted mb-1">Target Met</p>
                      <p>
                        {cap.execution.autotune.target_met ? (
                          <CheckCircle2 className="w-4 h-4 text-green-600" />
                        ) : (
                          <XCircle className="w-4 h-4 text-red-600" />
                        )}
                      </p>
                    </div>

                    {cap.execution.autotune.scale?.gap !== undefined && (
                      <div>
                        <p className="text-xs text-muted mb-1">Gap</p>
                        <p className="font-mono">
                          {cap.execution.autotune.scale.gap.toFixed(4)}
                        </p>
                      </div>
                    )}
                  </>
                )}

                {cap.behavioural_gate?.result && (
                  <>
                    {cap.behavioural_gate.result.margin_before !== undefined && (
                      <div>
                        <p className="text-xs text-muted mb-1">
                          Margin Before
                        </p>
                        <p className="font-mono">
                          {cap.behavioural_gate.result.margin_before.toFixed(4)}
                        </p>
                      </div>
                    )}
                    {cap.behavioural_gate.result.margin_after !== undefined && (
                      <div>
                        <p className="text-xs text-muted mb-1">Margin After</p>
                        <p className="font-mono">
                          {cap.behavioural_gate.result.margin_after.toFixed(4)}
                        </p>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Expectations */}
      <section className="mb-8">
        <h3 className="font-serif text-2xl font-bold mb-6">Expectations</h3>
        <div className="space-y-4">
          {report.expectations.map((exp, idx) => (
            <div key={idx} className="card">
              <div className="flex items-start justify-between gap-4 mb-4">
                <div>
                  <h4 className="font-semibold">{exp.expectation}</h4>
                  <p className="text-sm text-body capitalize">{exp.kind}</p>
                </div>
                {exp.passed ? (
                  <span className="badge-pass flex-shrink-0">PASS</span>
                ) : (
                  <span className="badge-fail flex-shrink-0">FAIL</span>
                )}
              </div>

              <div className="space-y-2">
                <div>
                  <p className="text-xs text-muted mb-1">Evidence</p>
                  <p className="text-sm font-mono bg-panel p-2 rounded">
                    {exp.evidence}
                  </p>
                </div>
                {exp.detail && (
                  <div>
                    <p className="text-xs text-muted mb-1">Detail</p>
                    <p className="text-sm">{exp.detail}</p>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Back button */}
      <div className="flex gap-2">
        <Link href="/builds" className="button-secondary">
          ← Back to Builds
        </Link>
        <Link href="/" className="button-secondary">
          <Home className="w-4 h-4 inline mr-2" />
          Home
        </Link>
      </div>
    </div>
  );
}

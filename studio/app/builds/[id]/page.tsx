"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Loader, CheckCircle2, XCircle, ChevronDown } from "lucide-react";
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
  target,
}: {
  before?: number;
  after?: number;
  target?: number;
}) {
  if (before === undefined || after === undefined) return null;
  const maxVal = Math.max(Math.abs(before), Math.abs(after), Math.abs(target || 0), 0.1);
  const beforePct = (before / maxVal) * 50 + 50;
  const afterPct = (after / maxVal) * 50 + 50;

  return (
    <div className="flex gap-2 items-center text-xs">
      <div className="flex-1 bg-panel h-6 border border-hairline border-gray-300 rounded overflow-hidden flex items-center relative">
        <div
          className="h-full bg-accent"
          style={{ width: `${Math.max(0, beforePct)}%` }}
        />
        {target !== undefined && (
          <div
            className="absolute h-full w-0.5 bg-gray-400"
            style={{ left: `${50 + (target / maxVal) * 50}%` }}
            title={`Target: ${target.toFixed(4)}`}
          />
        )}
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

function ExpandableSection({
  title,
  defaultOpen,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen ?? false);

  return (
    <div className="card">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between gap-4 p-4 hover:bg-panel/50 transition-colors"
      >
        <h3 className="font-serif font-bold">{title}</h3>
        <ChevronDown
          className={`w-5 h-5 text-muted transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>
      {open && <div className="border-t border-hairline border-gray-300 p-4">{children}</div>}
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

      <div className="divider" />

      {/* L3: Program */}
      <h2 className="section-heading" data-layer="L3">L3 · Program</h2>
      <ExpandableSection title="Program Source" defaultOpen={false}>
        {report.program_id ? (
          <div className="space-y-2">
            <p className="text-sm text-muted">Program ID: {report.program_id}</p>
            <p className="text-xs text-muted">
              (View in programs table — contains .loom source)
            </p>
          </div>
        ) : (
          <div className="text-sm text-body">
            <p className="mb-2">App: {report.app}</p>
            <p className="text-xs text-muted">
              Contract: expectations and capabilities the model must satisfy.
            </p>
          </div>
        )}
      </ExpandableSection>

      <div className="divider" />

      {/* L2: Capability Graph */}
      <h2 className="section-heading" data-layer="L2">L2 · Capability Graph</h2>
      {report.capabilities.length > 0 ? (
        <div className="space-y-4 mb-12">
          {report.capabilities.map((cap, idx) => (
            <ExpandableSection
              key={idx}
              title={`${cap.capability}${cap.kind ? ` · ${cap.kind}` : ""}`}
              defaultOpen={idx === 0}
            >
              <div className="space-y-4">
                {cap.clause && (
                  <div>
                    <p className="stat-label mb-2">Clause</p>
                    <p className="text-sm text-body">{cap.clause}</p>
                  </div>
                )}

                <div>
                  <p className="stat-label mb-2">Strategy</p>
                  <p className="font-mono text-sm">{cap.strategy}</p>
                </div>

                {cap.reason && (
                  <div>
                    <p className="stat-label mb-2">Reason</p>
                    <p className="text-sm text-body">{cap.reason}</p>
                  </div>
                )}

                {/* L1: Mech-Interp IR + Search */}
                {cap.execution?.autotune && (
                  <div className="border-t border-hairline border-gray-300 pt-4">
                    <h4 className="font-mono text-xs font-bold mb-3 text-muted">L1: Autotune Search</h4>

                    {cap.execution.autotune.skipped ? (
                      <p className="text-sm text-muted">Skipped: {cap.reason}</p>
                    ) : (
                      <div className="space-y-3">
                        {cap.execution.autotune.scale && (
                          <div className="grid md:grid-cols-2 gap-3 text-xs">
                            {cap.execution.autotune.scale.instructed_cost !== undefined && (
                              <div>
                                <p className="text-muted">Instructed Cost</p>
                                <p className="font-mono">
                                  {cap.execution.autotune.scale.instructed_cost.toFixed(4)} nats
                                </p>
                              </div>
                            )}
                            {cap.execution.autotune.scale.uninstructed_cost !== undefined && (
                              <div>
                                <p className="text-muted">Uninstructed Cost</p>
                                <p className="font-mono">
                                  {cap.execution.autotune.scale.uninstructed_cost.toFixed(4)} nats
                                </p>
                              </div>
                            )}
                            {cap.execution.autotune.scale.gap !== undefined && (
                              <div>
                                <p className="text-muted">Gap</p>
                                <p className="font-mono font-bold">
                                  {cap.execution.autotune.scale.gap.toFixed(4)} nats
                                </p>
                              </div>
                            )}
                            {cap.execution.autotune.scale.target_nats !== undefined && (
                              <div>
                                <p className="text-muted">Target</p>
                                <p className="font-mono">
                                  {cap.execution.autotune.scale.target_nats.toFixed(4)} nats
                                </p>
                              </div>
                            )}
                          </div>
                        )}

                        {(cap.execution.autotune.n_trials || cap.execution.autotune.n_admissible) && (
                          <div className="grid md:grid-cols-2 gap-3 text-xs">
                            <div>
                              <p className="text-muted">Trials</p>
                              <p className="font-mono">
                                {cap.execution.autotune.n_admissible || 0}/{cap.execution.autotune.n_trials || 0}
                              </p>
                            </div>
                            <div>
                              <p className="text-muted">Target Met</p>
                              <p className="font-mono">
                                {cap.execution.autotune.target_met ? "Yes" : "No"}
                              </p>
                            </div>
                          </div>
                        )}

                        {cap.execution.autotune.trials && cap.execution.autotune.trials.length > 0 && (
                          <div className="mt-3 pt-3 border-t border-hairline border-gray-300">
                            <p className="text-xs font-mono text-muted mb-2">Trial Table</p>
                            <div className="overflow-x-auto">
                              <table className="text-xs w-full">
                                <thead>
                                  <tr className="border-b border-hairline border-gray-300">
                                    <th className="text-left px-2 py-1 text-muted">Config</th>
                                    <th className="text-right px-2 py-1 text-muted">Score</th>
                                    <th className="text-left px-2 py-1 text-muted">Status</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {cap.execution.autotune.trials.map((trial, tidx) => (
                                    <tr
                                      key={tidx}
                                      className="border-b border-hairline border-gray-200"
                                    >
                                      <td className="px-2 py-1 font-mono text-muted">
                                        {JSON.stringify(trial.config).slice(0, 30)}...
                                      </td>
                                      <td className="text-right px-2 py-1 font-mono">
                                        {trial.score.toFixed(4)}
                                      </td>
                                      <td className="px-2 py-1 text-muted">
                                        {trial.rejected_reason ? (
                                          <span className="text-red-600 text-xs">
                                            {trial.rejected_reason.slice(0, 20)}
                                          </span>
                                        ) : (
                                          <span className="text-verified">Admissible</span>
                                        )}
                                      </td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* Behavioral gate */}
                {cap.behavioural_gate && (
                  <div className="border-t border-hairline border-gray-300 pt-4">
                    <h4 className="font-mono text-xs font-bold mb-3 text-muted">
                      Behavioral Gate Block
                    </h4>

                    {cap.behavioural_gate.budget !== undefined && (
                      <div className="grid md:grid-cols-3 gap-3 text-xs mb-3">
                        <div>
                          <p className="text-muted">Budget</p>
                          <p className="font-mono">{cap.behavioural_gate.budget}</p>
                        </div>
                        <div>
                          <p className="text-muted">Resolution</p>
                          <p className="font-mono">{cap.behavioural_gate.resolution}</p>
                        </div>
                        {cap.behavioural_gate.note && (
                          <div>
                            <p className="text-muted">Note</p>
                            <p className="text-xs">{cap.behavioural_gate.note}</p>
                          </div>
                        )}
                      </div>
                    )}

                    {cap.behavioural_gate.result && (
                      <>
                        <p className="stat-label mb-3">Margin Before → After</p>
                        <MarginBar
                          before={cap.behavioural_gate.result.margin_before}
                          after={cap.behavioural_gate.result.margin_after}
                          target={cap.behavioural_gate.result.target}
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

                        {cap.behavioural_gate.trials && cap.behavioural_gate.trials.length > 0 && (
                          <div className="mt-3 pt-3 border-t border-hairline border-gray-300">
                            <p className="text-xs font-mono text-muted mb-2">Trial Table</p>
                            <div className="overflow-x-auto">
                              <table className="text-xs w-full">
                                <thead>
                                  <tr className="border-b border-hairline border-gray-300">
                                    <th className="text-left px-2 py-1 text-muted">Config</th>
                                    <th className="text-right px-2 py-1 text-muted">Score</th>
                                    <th className="text-left px-2 py-1 text-muted">Status</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {cap.behavioural_gate.trials.map((trial, tidx) => (
                                    <tr
                                      key={tidx}
                                      className="border-b border-hairline border-gray-200"
                                    >
                                      <td className="px-2 py-1 font-mono text-muted">
                                        {JSON.stringify(trial.config).slice(0, 30)}...
                                      </td>
                                      <td className="text-right px-2 py-1 font-mono">
                                        {trial.score.toFixed(4)}
                                      </td>
                                      <td className="px-2 py-1 text-muted">
                                        {trial.rejected_reason ? (
                                          <span className="text-red-600 text-xs">
                                            {trial.rejected_reason.slice(0, 20)}
                                          </span>
                                        ) : (
                                          <span className="text-verified">Admissible</span>
                                        )}
                                      </td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                )}
              </div>
            </ExpandableSection>
          ))}
        </div>
      ) : (
        <div className="card text-center py-8">
          <p className="text-muted">No capabilities to display</p>
        </div>
      )}

      <div className="divider" />

      {/* L0: Substrate */}
      <h2 className="section-heading" data-layer="L0">L0 · Substrate</h2>
      <ExpandableSection title="Base Model & Configuration" defaultOpen={true}>
        <div className="space-y-4">
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <p className="stat-label mb-1">Base Model</p>
              <p className="font-mono text-sm">{report.base_model}</p>
            </div>
            <div>
              <p className="stat-label mb-1">Device</p>
              <p className="text-sm text-body">GPU</p>
            </div>
          </div>

          {report.search_space?.explained && (
            <div>
              <p className="stat-label mb-2">Search Space</p>
              <pre className="bg-panel p-3 rounded text-xs font-mono overflow-x-auto">
                {report.search_space.explained}
              </pre>
            </div>
          )}

          {report.side_effect_guard && (
            <div>
              <p className="stat-label mb-2">Side Effect Guard</p>
              <div className="grid md:grid-cols-2 gap-3 text-xs">
                {report.side_effect_guard.budget !== undefined && (
                  <div>
                    <p className="text-muted">Budget</p>
                    <p className="font-mono">{report.side_effect_guard.budget}</p>
                  </div>
                )}
                {report.side_effect_guard.resolution !== undefined && (
                  <div>
                    <p className="text-muted">Resolution</p>
                    <p className="font-mono">{report.side_effect_guard.resolution}</p>
                  </div>
                )}
              </div>
              {report.side_effect_guard.note && (
                <p className="text-xs text-muted mt-2">{report.side_effect_guard.note}</p>
              )}
            </div>
          )}
        </div>
      </ExpandableSection>

      <ExpandableSection title="Controls Installed">
        {report.n_controls_installed ? (
          <div className="space-y-3">
            <p className="text-sm font-mono">
              {report.n_controls_installed} control{report.n_controls_installed !== 1 ? "s" : ""} installed
            </p>
            {report.controls && report.controls.length > 0 && (
              <div className="overflow-x-auto">
                <table className="text-xs w-full">
                  <thead>
                    <tr className="border-b border-hairline border-gray-300">
                      <th className="text-left px-2 py-1 text-muted">Name</th>
                      <th className="text-left px-2 py-1 text-muted">Layer</th>
                      <th className="text-right px-2 py-1 text-muted">Strength</th>
                      <th className="text-right px-2 py-1 text-muted">Side Effect</th>
                    </tr>
                  </thead>
                  <tbody>
                    {report.controls.map((ctrl, cidx) => (
                      <tr key={cidx} className="border-b border-hairline border-gray-200">
                        <td className="px-2 py-1 font-mono">{ctrl.name || `control-${cidx}`}</td>
                        <td className="px-2 py-1">{ctrl.layer || "—"}</td>
                        <td className="text-right px-2 py-1 font-mono">
                          {ctrl.strength?.toFixed(3) || "—"}
                        </td>
                        <td className="text-right px-2 py-1 font-mono">
                          {ctrl.side_effect?.toFixed(3) || "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        ) : (
          <p className="text-sm text-muted">No controls installed</p>
        )}
      </ExpandableSection>

      {report.execution?.adapter_saved_to && report.execution.adapter_saved_to.length > 0 && (
        <ExpandableSection title="Adapters">
          <div className="space-y-2">
            {report.execution.adapter_saved_to.map((adapter, aidx) => (
              <p key={aidx} className="font-mono text-sm text-body">
                {adapter}
              </p>
            ))}
          </div>
        </ExpandableSection>
      )}

      {report.hf_repo && (
        <ExpandableSection title="Hugging Face Repository">
          <div>
            <a
              href={`https://huggingface.co/${report.hf_repo}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent hover:underline text-sm font-mono"
            >
              {report.hf_repo} ↗
            </a>
          </div>
        </ExpandableSection>
      )}

      <div className="divider" />

      {/* Expectations */}
      {report.expectations.length > 0 && (
        <>
          <h2 className="section-heading">Expectations Verified</h2>
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

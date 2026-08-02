"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Loader, CheckCircle2, XCircle, ChevronDown } from "lucide-react";
import type { BuildReport, Capability, Expectation } from "@/lib/types";
import { getBuildStatus, getBuildReport } from "@/lib/gpu";
import { LayerWalk } from "@/components/layer-walk";
import examples from "@/lib/examples.json";

async function getShowcase() {
  try {
    const res = await fetch("/api/showcase");
    return (await res.json()) as BuildReport[];
  } catch {
    return [];
  }
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
        className="w-full flex items-center justify-between gap-4 p-4 hover:bg-night-700/70 transition-colors"
      >
        <h3 className="font-display font-bold text-slate-100">{title}</h3>
        <ChevronDown
          className={`w-5 h-5 text-slate-400 transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>
      {open && <div className="border-t border-night-600/50 p-4 space-y-3">{children}</div>}
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

      setPollingActive(true);
      let attempts = 0;
      const maxAttempts = 60;

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
      <main className="bg-night-950 min-h-screen">
        <div className="max-w-7xl mx-auto px-6 py-12">
          <h1 className="font-display text-4xl font-bold text-slate-100 mb-8">
            Build Details
          </h1>
          <div className="card text-center py-12">
            <Loader className="w-8 h-8 animate-spin mx-auto mb-4 text-gold-400" />
            <p className="text-slate-400">Loading build...</p>
          </div>
        </div>
      </main>
    );
  }

  if (error && !pollingActive) {
    return (
      <main className="bg-night-950 min-h-screen">
        <div className="max-w-7xl mx-auto px-6 py-12">
          <h1 className="font-display text-4xl font-bold text-slate-100 mb-8">
            Build Details
          </h1>
          <div className="card border-rose-500/50 bg-rose-500/5 p-6">
            <div className="flex gap-3">
              <XCircle className="w-6 h-6 text-rose-400 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-rose-300 mb-1">Error</p>
                <p className="text-rose-200 text-sm">{error}</p>
              </div>
            </div>
          </div>
        </div>
      </main>
    );
  }

  if (!report) {
    return (
      <main className="bg-night-950 min-h-screen">
        <div className="max-w-7xl mx-auto px-6 py-12">
          <h1 className="font-display text-4xl font-bold text-slate-100 mb-8">
            Build Details
          </h1>
          {pollingActive ? (
            <div className="card text-center py-12">
              <Loader className="w-8 h-8 animate-spin mx-auto mb-4 text-gold-400" />
              <p className="text-slate-400">Waiting for build to complete...</p>
              <p className="text-slate-500 text-sm mt-2">This may take 5-15 minutes</p>
            </div>
          ) : (
            <div className="card text-center py-12">
              <p className="text-slate-400">Build not found</p>
            </div>
          )}
        </div>
      </main>
    );
  }

  const exampleMap = examples as unknown as Record<string, string>;
  const programSource: string | undefined = report
    ? exampleMap[String((report as any).app || "").toLowerCase()]
    : undefined;

  return (
    <main className="bg-night-950 min-h-screen">
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="mb-8">
          <Link href="/builds" className="text-gold-300 hover:text-gold-200 text-sm">
            ← Back to Builds
          </Link>
          <div className="flex items-center justify-between mt-4">
            <div>
              <h1 className="font-display text-4xl font-bold text-slate-100 mb-2">
                {report.app}
              </h1>
              <p className="text-slate-400 text-sm">
                Built on {report.base_model.split("/").pop()}
                {isReplay && <span className="ml-3 badge-gold">Replay</span>}
              </p>
            </div>
            <div>
              {report.passed ? (
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-6 h-6 text-emerald-400" />
                  <span className="badge-emerald">Verified</span>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <XCircle className="w-6 h-6 text-rose-400" />
                  <span className="badge-rose">Failed</span>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="border-t border-night-600/50 my-8" />

        <div className="space-y-4">
          {/* Summary */}
          <div className="grid md:grid-cols-2 gap-4">
            <div className="card p-6">
              <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">
                Wall Time
              </p>
              <p className="text-slate-100 font-mono text-lg">
                {report.wall_clock_s ? `${(report.wall_clock_s / 60).toFixed(1)}m` : "–"}
              </p>
            </div>
            <div className="card p-6">
              <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">
                Capabilities
              </p>
              <p className="text-slate-100 font-mono text-lg">
                {report.capabilities?.length || 0}
              </p>
            </div>
          </div>

          {/* The layer walk: program -> capability graph -> ISA -> substrate */}
          <LayerWalk report={report as any} source={programSource} />

          {/* Expectations — the program's own acceptance tests, on the built model */}
          {(report as any).expectations?.length > 0 && (
            <div className="mt-12">
              <div className="flex items-baseline gap-3 mb-4">
                <span className="font-mono text-gold-500 text-sm">✓</span>
                <h2 className="font-display text-2xl text-slate-100">Expectations</h2>
                <span className="text-xs text-slate-500">
                  measured on the finished model, with its controls attached
                </span>
              </div>
              <div className="space-y-3">
                {(report as any).expectations.map((e: any, i: number) => (
                  <div key={i} className="card p-4">
                    <div className="flex items-start justify-between gap-4">
                      <p className="text-slate-100">{e.expectation}</p>
                      <span
                        className={
                          e.passed
                            ? "shrink-0 text-xs font-mono px-2 py-0.5 rounded-full border border-emerald-500/40 text-emerald-300 bg-emerald-500/10"
                            : "shrink-0 text-xs font-mono px-2 py-0.5 rounded-full border border-rose-500/40 text-rose-300 bg-rose-500/10"
                        }
                      >
                        {e.passed ? "PASS" : "FAIL"}
                      </span>
                    </div>
                    <p className="text-xs text-slate-500 mt-1">{e.detail}</p>
                    {e.evidence && (
                      <blockquote className="mt-2 border-l-2 border-night-600 pl-3 text-sm text-slate-400 font-mono">
                        {e.evidence}
                      </blockquote>
                    )}
                  </div>
                ))}
              </div>
              {(report as any).verified_against_recitation_of ? (
                <p className="mt-3 text-xs text-slate-500">
                  Verified against recitation of{" "}
                  {(report as any).verified_against_recitation_of} training strings: a
                  sample that merely repeats what it was trained on is not counted.
                </p>
              ) : null}
            </div>
          )}

          {Boolean((report as any).notes) && (
            <ExpandableSection title="Build Notes">
              <pre className="text-xs text-slate-300 overflow-x-auto bg-night-900/50 p-3 rounded border border-night-600/50">
                {(report as any).notes}
              </pre>
            </ExpandableSection>
          )}
        </div>

        <div className="mt-12 pt-8 border-t border-night-600/50">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-slate-400 text-sm mb-2">
                {isReplay
                  ? "This is a replay of a committed build."
                  : "Build completed and verified."}
              </p>
            </div>
            <Link href="/use">
              <button className="btn-gold">Try It Out →</button>
            </Link>
          </div>
        </div>
      </div>
    </main>
  );
}

"use client";

import { useEffect, useState } from "react";
import scienceData from "@/lib/science.json";

interface Hypothesis {
  prior: number;
  posterior: number;
  layer: string;
  description: string;
  [key: string]: unknown;
}

interface Claim {
  id: string;
  statement: string;
  value: string;
  artifact: string;
  measured: boolean;
}

export default function SciencePage() {
  const [hypotheses, setHypotheses] = useState<Record<string, Hypothesis>>({});
  const [claims, setClaims] = useState<Claim[]>([]);
  const [filter, setFilter] = useState<string>("");

  useEffect(() => {
    setHypotheses(scienceData.hypotheses);
    setClaims(scienceData.claims);
  }, []);

  const getVerdict = (posterior: number): { text: string; color: string } => {
    if (posterior > 0.8) return { text: "Supported", color: "emerald" };
    if (posterior < 0.2) return { text: "Refuted", color: "rose" };
    return { text: "Undecided", color: "amber" };
  };

  const filteredClaims = claims.filter(
    (c) =>
      filter === "" ||
      c.id.includes(filter) ||
      c.statement.toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <main className="bg-night-950">
      <div className="max-w-6xl mx-auto px-6 py-16">
        <div className="mb-16">
          <h1 className="font-display text-6xl font-bold text-slate-100 mb-4">
            Five Hypotheses, Verified
          </h1>
          <p className="text-lg text-slate-400">
            The compiler's four-layer abstraction is grounded in empirical evidence. Each
            hypothesis has a measured posterior probability and a committed result artifact.
          </p>
        </div>

        {/* Hypotheses cards */}
        <div className="space-y-4 mb-16">
          {Object.entries(hypotheses).map(([key, hyp]) => {
            const verdict = getVerdict(hyp.posterior);
            const colorMap: Record<string, string> = {
              emerald: "border-emerald-500/50 bg-emerald-500/5",
              rose: "border-rose-500/50 bg-rose-500/5",
              amber: "border-amber-500/50 bg-amber-500/5",
            };
            return (
              <div
                key={key}
                className={`card border-2 ${colorMap[verdict.color]} p-6`}
              >
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <span className="font-display text-lg font-bold text-gold-300">
                        {key}
                      </span>
                      <span className="text-xs text-slate-400 font-mono">
                        {hyp.layer}
                      </span>
                    </div>
                    <p className="text-slate-100 font-medium">{hyp.description}</p>
                  </div>
                  <span
                    className={`badge-${verdict.color} whitespace-nowrap`}
                  >
                    {verdict.text}
                  </span>
                </div>

                {/* Prior → Posterior meter */}
                <div className="mt-4 space-y-1">
                  <div className="flex justify-between text-xs text-slate-400">
                    <span>Prior</span>
                    <span>Posterior</span>
                  </div>
                  <div className="flex gap-2">
                    <div className="flex-1 h-2 bg-night-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gold-600/50"
                        style={{ width: `${hyp.prior * 100}%` }}
                      />
                    </div>
                    <div className="flex-1 h-2 bg-night-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gold-400"
                        style={{ width: `${hyp.posterior * 100}%` }}
                      />
                    </div>
                  </div>
                  <div className="flex justify-between text-xs text-slate-300 font-mono">
                    <span>{(hyp.prior * 100).toFixed(0)}%</span>
                    <span>{(hyp.posterior * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <div className="border-t border-night-600/50 my-16" />

        {/* Controls callout */}
        <div className="card p-6 mb-16 border-gold-600/50 bg-gold-600/5">
          <h3 className="font-display text-lg font-bold text-gold-300 mb-4">
            Controls that Caught False Positives
          </h3>
          <ul className="space-y-2 text-slate-300 text-sm">
            <li className="flex gap-2">
              <span className="text-gold-400">→</span>
              <span>
                <strong>Untrained-model control:</strong> Every claim includes a baseline
                from a randomly-initialized network of the same shape. If the untrained
                model also passes, the claim proves nothing.
              </span>
            </li>
            <li className="flex gap-2">
              <span className="text-gold-400">→</span>
              <span>
                <strong>Trivial baselines:</strong> Each hypothesis includes recent-token and
                linear-probe baselines, so we know when a capability is trivial or just
                memorized.
              </span>
            </li>
            <li className="flex gap-2">
              <span className="text-gold-400">→</span>
              <span>
                <strong>Multi-seed variance:</strong> Claims are measured across 3+ random
                seeds with confidence intervals. No single run determines a verdict.
              </span>
            </li>
            <li className="flex gap-2">
              <span className="text-gold-400">→</span>
              <span>
                <strong>Metric validation:</strong> We explicitly check whether a metric can
                vary with different architectures or seeds. A metric that never moves is not
                evidence.
              </span>
            </li>
          </ul>
        </div>

        {/* Claims ledger */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="font-display text-2xl font-bold text-slate-100 mb-1">
                Provenance Ledger
              </h3>
              <p className="text-slate-400 text-sm">
                {claims.length} measured claims, each with an artifact
              </p>
            </div>
            <input
              type="text"
              placeholder="Search claims..."
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="input max-w-xs"
            />
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-night-600/50">
                <th className="text-left py-3 px-3 text-slate-400 font-mono text-xs uppercase tracking-wider">
                  ID
                </th>
                <th className="text-left py-3 px-3 text-slate-400 font-mono text-xs uppercase tracking-wider">
                  Statement
                </th>
                <th className="text-right py-3 px-3 text-slate-400 font-mono text-xs uppercase tracking-wider">
                  Value
                </th>
                <th className="text-left py-3 px-3 text-slate-400 font-mono text-xs uppercase tracking-wider">
                  Artifact
                </th>
                <th className="text-center py-3 px-3 text-slate-400 font-mono text-xs uppercase tracking-wider">
                  Status
                </th>
              </tr>
            </thead>
            <tbody>
              {filteredClaims.map((claim) => (
                <tr
                  key={claim.id}
                  className="border-b border-night-600/30 hover:bg-night-800/50 transition-colors"
                >
                  <td className="py-3 px-3 text-gold-300 font-mono">{claim.id}</td>
                  <td className="py-3 px-3 text-slate-300">{claim.statement}</td>
                  <td className="py-3 px-3 text-right text-slate-300 font-mono">
                    {claim.value}
                  </td>
                  <td className="py-3 px-3 text-slate-400 text-xs font-mono">
                    {claim.artifact ? (
                      <a
                        href={`#${claim.artifact}`}
                        className="text-gold-400 hover:text-gold-300"
                      >
                        {claim.artifact.split("/").pop()}
                      </a>
                    ) : (
                      <span className="text-slate-600">–</span>
                    )}
                  </td>
                  <td className="py-3 px-3 text-center">
                    {claim.measured ? (
                      <span className="badge-emerald">Measured</span>
                    ) : (
                      <span className="chip text-xs">Not yet measured</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="h-8" />
      </div>
    </main>
  );
}

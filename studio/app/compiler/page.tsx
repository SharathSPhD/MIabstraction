"use client";

import { useState } from "react";
import isaData from "@/lib/isa.json";

interface Instruction {
  op: string;
  target: string;
  for_kinds: string[];
  operands: string[];
  verified_by: string;
  note?: string;
}

export default function CompilerPage() {
  const [activeTab, setActiveTab] = useState<"overview" | "instructions" | "strategy">(
    "overview"
  );

  const instructions: Instruction[] = isaData.instructions || [];
  const uniqueOps = [...new Set(instructions.map((i) => i.op))];

  return (
    <main className="bg-night-950">
      <div className="max-w-6xl mx-auto px-6 py-16">
        <div className="mb-16">
          <h1 className="font-display text-6xl font-bold text-slate-100 mb-4">
            The Compiler: ISA and Lowering
          </h1>
          <p className="text-lg text-slate-400">
            How a Loom program becomes a behavioral machine. The compiler lowers your
            declarative specification through four layers, measuring every step, searching the
            parameter space, and verifying each result.
          </p>
        </div>

        {/* Tabs */}
        <div className="flex gap-4 mb-8 border-b border-night-600/50">
          {(["overview", "instructions", "strategy"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-3 px-4 font-medium text-sm transition-colors border-b-2 ${
                activeTab === tab
                  ? "border-gold-400 text-gold-300"
                  : "border-transparent text-slate-400 hover:text-slate-300"
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        {/* Overview Tab */}
        {activeTab === "overview" && (
          <div className="space-y-8">
            <div>
              <h2 className="font-display text-2xl font-bold text-slate-100 mb-6">
                Four Levels
              </h2>
              <div className="space-y-4">
                {isaData.levels.map((level: any) => (
                  <div key={level.id} className="card p-6">
                    <div className="flex items-start gap-4">
                      <div className="w-16 h-16 rounded-lg bg-gold-600/20 border border-gold-600/50 flex items-center justify-center flex-shrink-0">
                        <span className="font-display text-xl font-bold text-gold-300">
                          {level.id}
                        </span>
                      </div>
                      <div>
                        <h3 className="font-display text-lg font-bold text-slate-100 mb-2">
                          {level.name}
                        </h3>
                        <p className="text-slate-400 text-sm mb-2">{level.what}</p>
                        {level.kinds && (
                          <div className="flex flex-wrap gap-2">
                            {level.kinds.map((kind: string) => (
                              <span
                                key={kind}
                                className="chip text-xs"
                              >
                                {kind}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="border-t border-night-600/50 pt-8">
              <h2 className="font-display text-2xl font-bold text-slate-100 mb-4">
                The Lowering Pipeline
              </h2>
              <p className="text-slate-400 mb-6">
                Each layer transforms the previous one, with measurement at every step:
              </p>
              <div className="space-y-3">
                {[
                  {
                    step: "L3 → L2",
                    desc: "Parse program clauses into capability kinds. Check that corpora are manifested and gates are specific.",
                  },
                  {
                    step: "L2 → L1",
                    desc: "Choose strategies per capability kind. Select which mech-interp operations (read/amplify/suppress/install) will work.",
                  },
                  {
                    step: "L1 → L0",
                    desc: "Generate substrate operations. Calibrate doses, install adapters, set up controls. Measure gaps and verify every decision.",
                  },
                  {
                    step: "L0 execution",
                    desc: "Run on GPU. Return the compiled model, the control values, and the full measurement ledger.",
                  },
                ].map((stage) => (
                  <div key={stage.step} className="card p-4">
                    <h3 className="font-mono text-sm font-bold text-gold-300 mb-2">
                      {stage.step}
                    </h3>
                    <p className="text-slate-400 text-sm">{stage.desc}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Instructions Tab */}
        {activeTab === "instructions" && (
          <div className="space-y-6">
            <p className="text-slate-400 mb-6">
              The ISA: operations that read and write on measured internal objects.
            </p>

            {uniqueOps.map((op) => {
              const opsOfType = instructions.filter((i) => i.op === op);
              return (
                <div key={op} className="space-y-3">
                  <h3 className="font-display text-xl font-bold text-gold-300 uppercase">
                    {op}
                  </h3>
                  {opsOfType.map((instr, idx) => (
                    <div key={idx} className="card p-4 border-gold-600/30">
                      <div className="grid md:grid-cols-3 gap-4 text-sm mb-3">
                        <div>
                          <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">
                            Target
                          </p>
                          <p className="font-mono text-slate-100">{instr.target}</p>
                        </div>
                        <div>
                          <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">
                            For Kinds
                          </p>
                          <div className="flex flex-wrap gap-1">
                            {instr.for_kinds.map((k) => (
                              <span key={k} className="chip text-xs">
                                {k}
                              </span>
                            ))}
                          </div>
                        </div>
                        <div>
                          <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">
                            Operands
                          </p>
                          <p className="text-slate-200 text-xs">
                            {instr.operands.length > 0
                              ? instr.operands.join(", ")
                              : "—"}
                          </p>
                        </div>
                      </div>
                      <div>
                        <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">
                          Verified By
                        </p>
                        <p className="text-slate-300 text-sm leading-relaxed">
                          {instr.verified_by}
                        </p>
                      </div>
                      {instr.note && (
                        <div className="mt-3 pt-3 border-t border-night-600/50">
                          <p className="text-slate-400 text-xs italic">
                            {instr.note}
                          </p>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        )}

        {/* Strategy Tab */}
        {activeTab === "strategy" && (
          <div className="space-y-8">
            <div>
              <h2 className="font-display text-2xl font-bold text-slate-100 mb-6">
                Lowering Strategies
              </h2>
              <p className="text-slate-400 mb-6">
                Per capability kind, the compiler chooses from a ranked list of strategies, each
                with a known cost-benefit tradeoff.
              </p>

              <div className="space-y-4">
                {[
                  {
                    kind: "Knowledge",
                    strategies: [
                      "1. Install a retrieval circuit (E2: zero training required)",
                      "2. Train an adapter (lower risk, lower impact)",
                      "3. Continued pretraining (full learning, most expensive)",
                    ],
                  },
                  {
                    kind: "Skill",
                    strategies: [
                      "1. Install a compiled circuit (E2 verified)",
                      "2. Train on demonstrations",
                      "3. Full curriculum (only if gap > measured max for writes)",
                    ],
                  },
                  {
                    kind: "Style",
                    strategies: [
                      "1. Steer via feature amplification (fast, bounded)",
                      "2. Finetune on styled text (riskier, stronger)",
                    ],
                  },
                  {
                    kind: "Guardrail / Prohibition",
                    strategies: [
                      "1. Amplify refusal / suppress topic feature (steering)",
                      "2. Finetune refusals (higher cost)",
                      "3. Output filter (fallback only)",
                    ],
                  },
                ].map((strat) => (
                  <div key={strat.kind} className="card p-6">
                    <h3 className="font-display text-lg font-bold text-gold-300 mb-4">
                      {strat.kind}
                    </h3>
                    <ol className="space-y-2">
                      {strat.strategies.map((s, idx) => (
                        <li key={idx} className="text-slate-300 text-sm flex gap-3">
                          <span className="text-gold-400 flex-shrink-0">{s.split(".")[0]}.</span>
                          <span>{s.split(". ").slice(1).join(". ")}</span>
                        </li>
                      ))}
                    </ol>
                  </div>
                ))}
              </div>
            </div>

            <div className="border-t border-night-600/50 pt-8">
              <h2 className="font-display text-2xl font-bold text-slate-100 mb-4">
                Measurement and Verification
              </h2>
              <p className="text-slate-400 mb-6">
                Every instruction has a <strong>verified_by</strong> field: the empirical evidence
                that this operation delivers what is claimed. No ISA instruction can be emitted
                without that backing.
              </p>
              <div className="card p-6 border-emerald-500/50 bg-emerald-500/5">
                <h3 className="font-display text-lg font-bold text-emerald-400 mb-3">
                  The Load-Bearing Column
                </h3>
                <p className="text-slate-300 text-sm leading-relaxed">
                  The <strong>verified_by</strong> field is the load-bearing evidence that an
                  instruction is real. Every steering operation points to a ledger of measured
                  capacity in nats. Every circuit installation cites E2 (prefix score 0.979,
                  zero training). Feature reads cite E1 and E4 (incremental R² on unseen data).
                  If an operation has not been measured, it does not have an ISA instruction.
                </p>
              </div>
            </div>
          </div>
        )}

        <div className="h-8" />
      </div>
    </main>
  );
}

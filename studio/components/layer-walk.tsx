"use client";

/**
 * The layer walk: one build, opened all the way down.
 *
 * A build report is the compiler's own account of what it did — every gap it
 * measured, every configuration it tried and refused, every dose it installed. This
 * component renders that account at the density a specialist needs, arranged the way
 * the compiler actually walks it: L3 program, L2 capability graph, L1 the ISA
 * instructions and the searches that chose their operands, L0 the substrate.
 *
 * Nothing here computes a number. Every value is read from the report, and a field
 * the report does not carry renders as absent rather than as zero.
 */

import * as React from "react";
import { useState } from "react";
import { ChevronRight } from "lucide-react";

type Any = Record<string, any>;

const n = (v: unknown, d = 4) =>
  typeof v === "number" ? v.toFixed(d).replace(/\.?0+$/, "") : "—";

const KIND_CLAUSE: Record<string, string> = {
  knowledge: "knows from …",
  style: "speaks …",
  invariant: "always says …",
  prohibition: "never …",
  guardrail: "refuses …",
  skill: "can …",
};

/** The ISA instruction a strategy emits — the compiler's own mapping, mirrored. */
const STRATEGY_OPS: Record<string, string> = {
  steer_style_feature: "read(feature=style) · amplify(feature=style, dose)",
  monitor_and_correct: "read(feature) · amplify(feature, dose)",
  suppress_topic_feature: "read(feature=topic) · suppress(feature=topic, dose)",
  amplify_refusal_feature: "read(feature=refusal) · amplify(feature=refusal, dose)",
  install_compiled_circuit: "install(circuit)",
  retrieval_circuit: "install(circuit=retrieval)",
  continued_pretraining: "train(continued_pretrain)",
  knowledge_adapter: "train(train_adapter)",
  pretraining_mixture: "train(mix_corpus, pretrain)",
  finetune_refusals: "train(finetune)",
  finetune_on_demonstrations: "train(finetune)",
  curriculum: "train(build_curriculum, train)",
  output_filter: "install_filter",
};

function Rail({ id, title, sub }: { id: string; title: string; sub: string }): React.ReactElement {
  return (
    <div className="flex items-baseline gap-3 mb-4">
      <span className="font-mono text-gold-500 text-sm tabular-nums">{id}</span>
      <h2 className="font-display text-2xl text-slate-100">{title}</h2>
      <span className="text-xs text-slate-500">{sub}</span>
    </div>
  );
}

function Field({ label, value, mono = true }: { label: string; value: any; mono?: boolean }): React.ReactElement {
  return (
    <div className="flex justify-between gap-6 py-1.5 border-b border-night-700/60 last:border-0">
      <span className="text-xs uppercase tracking-wider text-slate-500">{label}</span>
      <span className={`text-sm text-slate-200 ${mono ? "font-mono tabular-nums" : ""} text-right`}>
        {value === undefined || value === null || value === "" ? (
          <span className="text-slate-600">not recorded</span>
        ) : (
          String(value)
        )}
      </span>
    </div>
  );
}

/** Margin meter with the declared target as a tick — the picture of "how much of
 *  the behaviour the program insisted on, and how much arrived". */
function Meter({ before, after, target }: { before?: number; after?: number; target?: number }): React.ReactElement {
  const max = Math.max(after ?? 0, target ?? 0, before ?? 0, 0.001) * 1.25;
  const pct = (v?: number) => `${Math.min(100, ((v ?? 0) / max) * 100)}%`;
  return (
    <div className="my-3">
      <div className="relative h-2 rounded-full bg-night-700 overflow-visible">
        <div
          className="absolute inset-y-0 left-0 rounded-full bg-gold-500/80"
          style={{ width: pct(after) }}
        />
        {before !== undefined && before > 0 && (
          <div
            className="absolute inset-y-0 left-0 rounded-full bg-slate-500/40"
            style={{ width: pct(before) }}
          />
        )}
        {target !== undefined && (
          <div
            className="absolute -top-1 h-4 w-px bg-emerald-400"
            style={{ left: pct(target) }}
            title={`target ${n(target, 3)}`}
          />
        )}
      </div>
      <div className="flex justify-between mt-1 text-[11px] font-mono text-slate-500">
        <span>before {n(before, 3)}</span>
        <span className="text-emerald-400">target {n(target, 3)}</span>
        <span className="text-gold-300">after {n(after, 3)}</span>
      </div>
    </div>
  );
}

function TrialTable({ trials }: { trials: Any[] }): React.ReactElement | null {
  const [open, setOpen] = useState(false);
  if (!trials?.length) return null;
  const shown = open ? trials : trials.slice(0, 4);
  return (
    <div className="mt-3">
      <table className="w-full text-[12px] font-mono tabular-nums">
        <thead>
          <tr className="text-left text-slate-500 uppercase tracking-wider text-[10px]">
            <th className="pb-1 font-normal">configuration</th>
            <th className="pb-1 font-normal text-right">score</th>
            <th className="pb-1 font-normal pl-4">verdict</th>
          </tr>
        </thead>
        <tbody>
          {shown.map((t, i) => (
            <tr key={i} className="border-t border-night-700/50">
              <td className="py-1 text-slate-300">
                {Object.entries(t.config || {})
                  .map(([k, v]) => `${k}=${typeof v === "number" ? +Number(v).toPrecision(3) : v}`)
                  .join("  ")}
              </td>
              <td className="py-1 text-right text-slate-200">{n(t.score, 4)}</td>
              <td className="py-1 pl-4">
                {t.rejected ? (
                  <span className="text-amber-400/90">{t.rejected}</span>
                ) : (
                  <span className="text-emerald-400">admissible</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {trials.length > 4 && (
        <button
          onClick={() => setOpen(!open)}
          className="mt-2 text-xs text-gold-400 hover:text-gold-300"
        >
          {open ? "show fewer" : `all ${trials.length} trials`}
        </button>
      )}
    </div>
  );
}

function CapabilityBlock({ cap }: { cap: Any }): React.ReactElement {
  const [open, setOpen] = useState(false);
  const at = cap.autotune || cap.execution?.autotune || {};
  const sc = at.scale || {};
  const gate = cap.behavioural_gate?.result || cap.escalation?.result || {};
  const exec = cap.execution || {};
  const best = (exec.autotune?.best?.metrics) || {};

  return (
    <div className="card p-4">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-start justify-between gap-4 text-left"
      >
        <div>
          <div className="text-xs uppercase tracking-wider text-slate-500 mb-1">
            {KIND_CLAUSE[cap.kind] ?? cap.kind}
          </div>
          <div className="text-slate-100">{cap.capability}</div>
          <div className="mt-1 font-mono text-[12px] text-gold-400/90">
            {STRATEGY_OPS[cap.strategy] ?? cap.strategy ?? "no strategy"}
          </div>
        </div>
        <ChevronRight
          className={`w-4 h-4 mt-1 shrink-0 text-slate-500 transition-transform ${open ? "rotate-90" : ""}`}
        />
      </button>

      {open && (
        <div className="mt-4 space-y-4">
          {/* L1 — what the instruction was asked to deliver, and what it did */}
          {sc.gap !== undefined && (
            <div>
              <div className="text-[11px] uppercase tracking-wider text-gold-500/80 mb-2">
                L1 · gap measured before any lever was tried
              </div>
              <Field label="cost with the rule stated" value={n(sc.instructed_cost)} />
              <Field label="cost without it" value={n(sc.uninstructed_cost)} />
              <Field label="gap (nats)" value={n(sc.gap)} />
              <Field
                label="program insists on"
                value={sc.must_recover !== undefined ? `${(sc.must_recover * 100).toFixed(0)}% → ${n(sc.target_nats)} nats` : undefined}
              />
            </div>
          )}

          {at.skipped && (
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3">
              <div className="text-[11px] uppercase tracking-wider text-amber-400/90 mb-1">
                instruction not emitted
              </div>
              <p className="text-sm text-slate-300">{at.skipped}</p>
            </div>
          )}

          {at.trials?.length > 0 && (
            <div>
              <div className="text-[11px] uppercase tracking-wider text-gold-500/80 mb-1">
                L1 · operand search ({at.n_admissible ?? 0}/{at.n_trials ?? at.trials.length} admissible
                {at.target_met ? " · target met" : " · target not met"})
              </div>
              {at.direction_from && (
                <p className="text-xs text-slate-500 mb-1">direction from: {at.direction_from}</p>
              )}
              <TrialTable trials={at.trials} />
            </div>
          )}

          {at.generalization && (
            <div>
              <div className="text-[11px] uppercase tracking-wider text-gold-500/80 mb-1">
                held-out check (a direction that moves only its own probes does not ship)
              </div>
              <Field label="delivered on derivation half" value={n(at.generalization.derive_delivered)} />
              <Field label="delivered on unseen half" value={n(at.generalization.holdout_delivered)} />
              {at.generalization.rejected && (
                <p className="text-xs text-amber-400/90 mt-1">{at.generalization.rejected}</p>
              )}
            </div>
          )}

          {exec.heldout_loss_before !== undefined || best.heldout_loss_before !== undefined ? (
            <div>
              <div className="text-[11px] uppercase tracking-wider text-gold-500/80 mb-1">
                L0 · training, measured on text excluded from it
              </div>
              <Field
                label="held-out loss"
                value={`${n(best.heldout_loss_before ?? exec.heldout_loss_before)} → ${n(best.heldout_loss_after ?? exec.heldout_loss_after)}`}
              />
              <Field label="base weights unchanged" value={String(best.base_weights_unchanged ?? exec.base_weights_unchanged ?? "—")} />
              <Field label="adapter saved to" value={exec.adapter_saved_to?.split("/").pop()} />
              {exec.autotune?.trials && <TrialTable trials={exec.autotune.trials} />}
            </div>
          ) : null}

          {Object.keys(gate).length > 0 && (
            <div>
              <div className="text-[11px] uppercase tracking-wider text-gold-500/80 mb-1">
                behavioural gate · measured on the composed model
              </div>
              {cap.behavioural_gate?.because && (
                <p className="text-xs text-slate-500 mb-1">{cap.behavioural_gate.because}</p>
              )}
              <Meter
                before={gate.margin_before}
                after={gate.margin_after}
                target={gate.target_margin}
              />
              <Field
                label="refused off-domain / in-domain"
                value={
                  gate.rates_after
                    ? `${n(gate.rates_after.refused_off_domain, 3)} / ${n(gate.rates_after.refused_in_domain, 3)}`
                    : undefined
                }
              />
              <Field label="probe resolution" value={n(gate.rates_after?.resolution, 4)} />
              {gate.reason && <p className="text-xs text-amber-400/90 mt-1">{gate.reason}</p>}
              {gate.autotune?.trials && <TrialTable trials={gate.autotune.trials} />}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export const LayerWalk: React.FC<any> = ({ report, source }: { report: Any; source?: string }) => {
  const caps: Any[] = report.capabilities || report.per_capability || [];
  const guard = report.side_effect_guard || {};
  const controls: Any[] = report.controls || [];
  const joint = report.joint_calibration || {};
  const scratch = report.substrate === "scratch" || String(report.base_model || "").startsWith("scratch");

  return (
    <div className="space-y-12">
      {/* ---------------------------------------------------------------- L3 */}
      <section>
        <Rail id="L3" title="Program" sub="what a person wrote, in consequences" />
        {source ? (
          <pre className="card p-4 text-[12px] leading-relaxed font-mono text-slate-300 overflow-x-auto">
            {source}
          </pre>
        ) : (
          <div className="card p-4 text-sm text-slate-400">
            The source is not stored with this build. Its contract survives as the
            expectations below.
          </div>
        )}
      </section>

      {/* ---------------------------------------------------------------- L2 */}
      <section>
        <Rail
          id="L2"
          title="Capability graph"
          sub={`${caps.length} capabilities, each with the strategy the compiler chose and why`}
        />
        <div className="space-y-3">
          {caps.map((c, i) => (
            <CapabilityBlock key={i} cap={c} />
          ))}
        </div>
        <p className="mt-3 text-xs text-slate-500">
          Open a capability to see the L1 instruction it lowered to, the gap the
          compiler measured before choosing, every configuration it tried, and what
          the composed model actually did.
        </p>
      </section>

      {/* ---------------------------------------------------------------- L0 */}
      <section>
        <Rail
          id="L0"
          title="Substrate"
          sub={scratch ? "a model made here" : "a model someone else trained, adapted"}
        />
        <div className="grid md:grid-cols-2 gap-4">
          <div className="card p-4">
            <div className="text-[11px] uppercase tracking-wider text-gold-500/80 mb-2">
              target
            </div>
            <Field label="base" value={report.base_model} mono={false} />
            <Field label="parameters" value={report.params ? report.params.toLocaleString() : undefined} />
            <Field label="device" value={report.device} mono={false} />
            <Field label="wall clock" value={report.wall_clock_s ? `${report.wall_clock_s}s` : undefined} />
            {scratch && (
              <>
                <Field label="architecture chosen" value={report.architecture_choice} mono={false} />
                <Field label="effort" value={report.effort} mono={false} />
              </>
            )}
          </div>

          <div className="card p-4">
            <div className="text-[11px] uppercase tracking-wider text-gold-500/80 mb-2">
              side-effect guard
            </div>
            <Field label="budget" value={guard.budget} />
            <Field label="resolution" value={guard.resolution} />
            {guard.note && <p className="mt-2 text-xs text-slate-500">{guard.note}</p>}
          </div>
        </div>

        {scratch && report.architecture_rationale && (
          <div className="card p-4 mt-4">
            <div className="text-[11px] uppercase tracking-wider text-gold-500/80 mb-2">
              why this architecture
            </div>
            <p className="text-sm text-slate-300">{report.architecture_rationale}</p>
            {report.compute_rationale && (
              <p className="text-xs text-slate-500 mt-2">{report.compute_rationale}</p>
            )}
          </div>
        )}

        {controls.length > 0 && (
          <div className="card p-4 mt-4">
            <div className="text-[11px] uppercase tracking-wider text-gold-500/80 mb-2">
              controls installed — the writes the artifact carries
            </div>
            <table className="w-full text-[12px] font-mono tabular-nums">
              <thead>
                <tr className="text-left text-slate-500 uppercase tracking-wider text-[10px]">
                  <th className="pb-1 font-normal">capability</th>
                  <th className="pb-1 font-normal text-right">layer</th>
                  <th className="pb-1 font-normal text-right">strength</th>
                  <th className="pb-1 font-normal text-right">side-effect</th>
                </tr>
              </thead>
              <tbody>
                {controls.map((c, i) => (
                  <tr key={i} className="border-t border-night-700/50">
                    <td className="py-1 text-slate-300">{c.name}</td>
                    <td className="py-1 text-right text-slate-200">{c.layer}</td>
                    <td className="py-1 text-right text-slate-200">{n(c.strength, 4)}</td>
                    <td className="py-1 text-right text-slate-200">{n(c.side_effect, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {joint.note && <p className="mt-2 text-xs text-slate-500">{joint.note}</p>}
          </div>
        )}

        {report.search_space?.explained && (
          <div className="card p-4 mt-4">
            <div className="text-[11px] uppercase tracking-wider text-gold-500/80 mb-2">
              the space this build searched
            </div>
            <pre className="text-[12px] font-mono text-slate-300 overflow-x-auto whitespace-pre-wrap">
              {report.search_space.explained}
            </pre>
          </div>
        )}

        {report.hf_repo && (
          <a
            href={`https://huggingface.co/${report.hf_repo}`}
            target="_blank"
            rel="noreferrer"
            className="inline-block mt-4 text-sm text-gold-400 hover:text-gold-300"
          >
            model on Hugging Face ↗
          </a>
        )}
      </section>
    </div>
  );
}

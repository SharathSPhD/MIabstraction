"""Provenance ledger: every claim the site makes, tied to the artifact that backs it.

The site is generated from this table. A claim whose artifact is missing renders as
"not yet measured" — it cannot render as a number, because there is no number to render.
That is the mechanism that keeps the page honest: the page cannot say more than the
repository can prove.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Claim:
    id: str
    statement: str
    artifact: str            # results/... path
    field: str               # dotted path into the JSON
    fmt: str = "{:.3g}"
    note: str = ""

    def value(self):
        p = ROOT / self.artifact
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
        # Keys may themselves contain dots (e.g. a mixing weight "0.5"), so a
        # bracket-quoted segment addresses one literally: ops.merge.ppl_by_alpha["0.5"]
        parts = re.findall(r'\["([^"]+)"\]|([^.\[\]]+)', self.field)
        for br, plain in parts:
            k = br or plain
            if isinstance(d, list):
                try:
                    d = d[int(k)]
                    continue
                except (ValueError, IndexError):
                    return None
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d

    def rendered(self) -> str:
        v = self.value()
        if v is None:
            return "—"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, (int, float)):
            return self.fmt.format(v)
        return str(v)

    def measured(self) -> bool:
        return self.value() is not None


CLAIMS = [
    Claim("E1", "A transformer trained on a hidden-Markov process linearly encodes the "
          "process's belief state in its residual stream",
          "results/final/e1_mess3/result.json", "best_r2", "R² = {:.3f}"),
    Claim("E1c", "…and an untrained network of the same shape already scores this, which "
          "is why the incremental measure is the one that counts",
          "results/final/e1_mess3/result.json", "best_r2_control", "R² = {:.3f}"),
    Claim("E1i", "Belief information beyond what the last 8 tokens trivially explain",
          "results/final/e1_mess3/result.json", "best_incremental_r2", "{:.4f}"),
    Claim("E1ic", "The same measure on an untrained control",
          "results/final/e1_mess3/result.json", "best_incremental_r2_control", "{:.4f}"),
    Claim("E2", "Induction (in-context copying) forms as a discrete phase transition",
          "results/e2_induction/result.json", "final_prefix_score", "score {:.3f}"),
    Claim("E2w", "…inside this fraction of training",
          "results/e2_induction/result.json", "window_frac", "{:.1%}"),
    Claim("E3", "SAE reconstruction metrics separate trained from untrained models — in "
          "the wrong direction",
          "results/e3_sae_control/result.json", "fvu_separation_sigma", "{:.1f}σ"),
    Claim("E4r", "A plain logistic probe on raw activations, on a known concept",
          "results/e4_probe_baseline/result.json", "raw_probe_acc_belief", "{:.3f}"),
    Claim("E4s", "The SAE-feature probe on the same concept",
          "results/e4_probe_baseline/result.json", "sae_probe_acc_belief", "{:.3f}"),
    Claim("E5", "Sparse-trained model's circuit size ratio — which turned out to equal "
          "the sparsity knob, making the measure uninformative",
          "results/e5_sparsity/result.json", "circuit_weight_ratio", "{:.4f}"),
    Claim("E5q", "…the imposed sparsity level it echoes",
          "results/e5_sparsity/result.json", "imposed_q", "q = {:.2f}"),
    Claim("C1", "Hand-compiled induction circuit: attention on the correct target, with "
          "zero training",
          "results/loom_constructed_demo.json", "prefix_score", "{:.3f}"),
    Claim("C2", "…its next-token accuracy on the copied region",
          "results/loom_constructed_demo.json", "icl_argmax_acc", "{:.3f}"),
    Claim("C3", "…nonzero weights in the compiled program",
          "results/loom_constructed_demo.json", "nonzero_params", "{:,.0f}"),
    Claim("CU1", "Multi-skill build: in-context copying",
          "results/loom_curriculum_demo.json", "per_skill_metrics.induction.prefix_score",
          "{:.3f}"),
    Claim("CU2", "Multi-skill build: state tracking",
          "results/loom_curriculum_demo.json", "per_skill_metrics.state_tracking.probe_r2",
          "{:.3f}"),
    Claim("CU3", "Multi-skill build: classification",
          "results/loom_curriculum_demo.json", "per_skill_metrics.classify.accuracy",
          "{:.3f}"),
    Claim("L3o", "Host cost of an out-of-band linked unit",
          "results/loom_link_demo.json", "L3_capacity.out_of_band_host_delta",
          "{:.4f} nats"),
    Claim("L3i", "Host cost of reserving 32 residual dimensions instead",
          "results/loom_link_demo.json",
          "L3_capacity.in_band_host_delta_by_reserved_dims.32", "{:.4f} nats"),
    Claim("L1a", "Host's own induction ability before linking",
          "results/loom_link_demo.json", "host.induction_acc_alone", "{:.3f}"),
    Claim("L1b", "…after linking a separately compiled unit it never trained with",
          "results/loom_link_demo.json", "L1_no_clobber.at_relaxed_budget.unit_icl_acc",
          "{:.3f}"),
    Claim("L1g", "…at a write gain the linker solved for automatically",
          "results/loom_link_demo.json", "L1_no_clobber.at_relaxed_budget.gain", "{:.4f}"),
    Claim("L1c", "…and this is what the host paid for it",
          "results/loom_link_demo.json", "L1_no_clobber.at_relaxed_budget.host_delta",
          "{:.3f} nats"),
    Claim("L2a", "Composition: a unit's own score drops when a second independently "
          "compiled unit is linked beside it",
          "results/loom_link_demo.json", "L2_composition.interference.a_shift", "{:+.3f}"),
    Claim("L2b", "…and the second unit is disturbed more",
          "results/loom_link_demo.json", "L2_composition.interference.b_shift", "{:+.3f}"),
    Claim("F1", "Foundation model pretrained from scratch on real text: held-out loss",
          "results/loom_foundation_demo.json", "val_loss", "{:.3f}"),
    Claim("F2", "…held-out perplexity",
          "results/loom_foundation_demo.json", "val_ppl", "{:.1f}"),
    Claim("F3", "…tokens of real English seen",
          "results/loom_foundation_demo.json", "tokens_seen", "{:,.0f}"),
    Claim("X1", "The same program, executed on a hybrid state-space stack built here: "
          "can a declared feature be read off it?",
          "results/loom_two_substrates.json",
          "substrates.A_from_scratch.gates.0.measured", "{:.3f}"),
    Claim("X2", "…and on a downloaded 1.24B open-weight model, same source",
          "results/loom_two_substrates.json",
          "substrates.B_open_weight.gates.0.measured", "{:.3f}"),
    Claim("X3", "…steering that feature on the open-weight substrate",
          "results/loom_two_substrates.json",
          "substrates.B_open_weight.gates.1.measured", "{:.4f}"),
    # One program, both substrates, on genuine specialist material (MedQuAD, published
    # by the U.S. National Library of Medicine and not written for this project).
    Claim("CL1", "Compiling clinic.loom onto a downloaded 1.24B instruct model: held-out "
          "loss on medical text excluded from training, before adaptation",
          "results/loom_clinic_build.json",
          "capabilities.0.execution.autotune.best.metrics.heldout_loss_before",
          "{:.4f}"),
    Claim("CL2", "…and after, with the downloaded weights frozen and the material in an "
          "adapter", "results/loom_clinic_build.json",
          "capabilities.0.execution.autotune.best.metrics.heldout_loss_after", "{:.4f}"),
    Claim("CL3", "…share of the model's parameters the adaptation was allowed to move",
          "results/loom_clinic_build.json",
          "capabilities.0.execution.autotune.best.metrics.adapter_ratio", "{:.3%}"),
    Claim("CL4", "The same program built from scratch instead: vocabulary learned from "
          "that corpus", "results/loom_clinic_scratch.json", "tokenizer_vocab_size",
          "{:,.0f}"),
    Claim("CL5", "…held-out perplexity of the model it trained, on sequences it never saw",
          "results/loom_clinic_scratch.json", "val_ppl", "{:.1f}"),
    Claim("CL6", "…sequences of real medical text held out from its training",
          "results/loom_clinic_scratch.json", "pretraining.heldout_sequences", "{:,.0f}"),
    # What one linear write at one layer can actually buy, measured across four
    # capabilities of the same program on the same model.
    Claim("SC1", "Steering capacity: stating the smallest of four rules outright is "
          "worth this much", "results/steering_capacity.json", "gap_min", "{:.4f}"),
    Claim("SC2", "…and the largest, twenty-three times more",
          "results/steering_capacity.json", "gap_max", "{:.4f}"),
    Claim("SC3", "…yet a searched control delivers at most", 
          "results/steering_capacity.json", "delivered_max", "{:.4f}"),
    Claim("SC4", "…so the smallest behaviour is nearly fully recovered",
          "results/steering_capacity.json", "capabilities.0.recovered_fraction",
          "{:.1%}"),
    Claim("SC5", "…and the largest is barely touched",
          "results/steering_capacity.json", "capabilities.3.recovered_fraction",
          "{:.1%}"),
    Claim("S1", "Supervised fine-tuning on real dialogue: held-out response perplexity "
          "before", "results/loom_sft_real.json", "heldout_ppl_before", "{:.1f}"),
    Claim("S2", "…and after, with loss computed on response tokens only",
          "results/loom_sft_real.json", "heldout_ppl_after", "{:.1f}"),
    Claim("M1", "Merging a base and instruct checkpoint beats both parents",
          "results/loom_surgery_demo.json", 'ops.merge.ppl_by_alpha["0.5"]', "{:.2f}"),
    Claim("M2", "…the better of the two parents it beats",
          "results/loom_surgery_demo.json", "ops.merge.ppl_base", "{:.2f}"),
    Claim("M3", "Dropping 25% of a model's layers costs this much perplexity",
          "results/loom_surgery_demo.json", "ops.prune_layers.ppl_by_kept_fraction.75pct",
          "{:,.0f}"),
    Claim("P0", "Frozen open-weight model programmed without touching its weights",
          "results/loom_port_demo.json", "params", "{:,.0f} params"),
    Claim("P1", "…a monitor reading a concept off its residual stream (leave-one-out)",
          "results/loom_port_demo.json", "monitor.probe_acc_loo", "{:.3f}"),
    Claim("P2", "…control strength at half-maximum effect, measured by dose-response",
          "results/loom_port_demo.json", "dose_response.ec50", "{:.3f}"),
    Claim("P3", "…the largest effect available before over-steering reverses it",
          "results/loom_port_demo.json", "dose_response.max_effect_nats", "{:.4f} nats"),

    # The gap-aware compiler: one program passing whole on two model families.
    Claim("BL", "clinic.loom passes every expectation on Llama-3.2-1B-Instruct",
          "results/loom_clinic_build_Llama-3_2-1B-Instruct.json", "passed"),
    Claim("BLg", "…its guardrail's refusal margin on the composed model, off-domain "
          "minus in-domain",
          "results/loom_clinic_build_Llama-3_2-1B-Instruct.json",
          "capabilities.4.behavioural_gate.result.margin_after", "{:.3f}"),
    Claim("BLk", "…held-out loss on MedQuAD text excluded from training, after",
          "results/loom_clinic_build_Llama-3_2-1B-Instruct.json",
          "capabilities.0.execution.autotune.best.metrics.heldout_loss_after",
          "{:.4f}"),
    Claim("BQ", "the same program, unchanged, passes on Qwen2.5-1.5B-Instruct",
          "results/loom_clinic_build_Qwen2_5-1_5B-Instruct.json", "passed"),
    Claim("BQg", "…its guardrail margin on the composed model",
          "results/loom_clinic_build_Qwen2_5-1_5B-Instruct.json",
          "capabilities.4.behavioural_gate.result.margin_after", "{:.3f}"),
    Claim("SCQ", "the most a steering write delivered on Qwen — the ceiling is "
          "substrate-specific",
          "results/steering_capacity_qwen2_5-1_5b.json", "delivered_max",
          "{:.4f} nats"),

    # Two constructed skills in one weight set.
    Claim("CP1", "succession, hand-compiled beside induction, exact on its alphabet",
          "results/loom_composed_demo.json", "succession_accuracy", "{:.3f}"),
    Claim("CP2", "induction inside the composed weights (0.909 alone)",
          "results/loom_composed_demo.json", "induction_acc_composed", "{:.3f}"),
    Claim("CP3", "largest logit change the second skill causes on traffic it should "
          "never see",
          "results/loom_composed_demo.json", "letter_traffic_max_logit_divergence",
          "{:.1e}"),

    # Follow-up probes: recorded, not scored.
    Claim("E6t", "SAE reconstruction error on the trained real LM (Qwen2.5-0.5B)",
          "results/e6_real_lm_sae/result.json", "separation.fvu.trained_mean",
          "FVU {:.4f}"),
    Claim("E6c", "…and on the same architecture at random initialization",
          "results/e6_real_lm_sae/result.json", "separation.fvu.control_mean",
          "FVU {:.4f}"),
    Claim("E7d", "heads a dense-trained model needs to reach accuracy 0.9",
          "results/e7_causal_size/result.json", 'per_floor.["0.9"].dense_mean',
          "{:.1f}"),
    Claim("E7s", "…and a weight-sparse-trained model, same floor",
          "results/e7_causal_size/result.json", 'per_floor.["0.9"].sparse_mean',
          "{:.1f}"),
    Claim("E7v", "the direction holds at every floor tested",
          "results/e7_causal_size/result.json", "direction_stable_across_floors"),
]


def ledger() -> list[Claim]:
    return CLAIMS


def coverage() -> tuple[int, int]:
    m = sum(1 for c in CLAIMS if c.measured())
    return m, len(CLAIMS)

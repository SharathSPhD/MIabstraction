"""L2 → L1 → L0: choose how to realize each capability on this substrate.

A compiler targeting a processor without an instruction does not stop; it emits a
different sequence with the same meaning. This module is that decision for language
models. Each capability has several possible realizations, ranked by preference; the
selector walks them and takes the first the substrate permits, recording why.

The realizations are expressed in the mechanistic-interpretability layer wherever the
behaviour is one this project validated as a real internal object — a style is a feature
and can be steered, a guardrail is a feature and can be amplified, a skill may be a
circuit and can be installed. Otherwise they fall back to data and training.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .capability import Capability, Kind
from .substrate import SubstrateProfile


@dataclass
class Strategy:
    """One way to realize a capability, and what it requires of the substrate."""
    name: str
    requires: list[str]              # levers that must be available
    mech_ops: list[str] = field(default_factory=list)   # L1 operations
    substrate_ops: list[str] = field(default_factory=list)  # L0 operations
    rationale: str = ""


# Ranked per capability kind: earlier entries are preferred when available.
# Preference order is cheapest-that-works, with mech-interp realizations ahead of
# training wherever the project measured them to be sufficient.
CATALOGUE: dict[Kind, list[Strategy]] = {
    Kind.KNOWLEDGE: [
        Strategy("pretraining_mixture", ["pretrain"],
                 substrate_ops=["mix_corpus", "pretrain"],
                 rationale="the material is in the weights from the start, which is only "
                           "possible when we are the ones training the model"),
        Strategy("continued_pretraining", ["continued_pretrain"],
                 substrate_ops=["continued_pretrain"],
                 rationale="extends what a downloaded model knows without discarding "
                           "what it already knows"),
        Strategy("knowledge_adapter", ["adapter"],
                 substrate_ops=["train_adapter"],
                 rationale="cheap and reversible; the base weights are untouched"),
        Strategy("retrieval_circuit", ["retrieval", "install_circuit"],
                 mech_ops=["install(circuit=retrieval)"],
                 substrate_ops=["build_index"],
                 rationale="the material stays outside the weights and is fetched on "
                           "demand"),
    ],
    Kind.SKILL: [
        Strategy("install_compiled_circuit", ["install_circuit"],
                 mech_ops=["install(circuit)"],
                 substrate_ops=[],
                 rationale="the skill exists as a verified circuit and can be written "
                           "into the weights directly, with no training at all"),
        Strategy("curriculum", ["curriculum"],
                 substrate_ops=["build_curriculum", "train"],
                 rationale="teach it as one skill among several while the model trains"),
        Strategy("finetune_on_demonstrations", ["finetune"],
                 substrate_ops=["finetune"],
                 rationale="show the model examples of the skill"),
    ],
    Kind.STYLE: [
        Strategy("steer_style_feature", ["steer", "monitor"],
                 mech_ops=["read(feature=style)", "amplify(feature=style, dose=ec50)"],
                 rationale="style is a direction in activation space; steering it is a "
                           "runtime write with a measured dose, and costs no training"),
        Strategy("finetune_on_styled_text", ["finetune"],
                 substrate_ops=["finetune"],
                 rationale="bake the style into the weights when steering is not "
                           "available or not sufficient"),
    ],
    Kind.INVARIANT: [
        Strategy("monitor_and_correct", ["monitor", "steer"],
                 mech_ops=["read(feature)", "amplify(feature, dose=ec50)"],
                 rationale="watch the internal signal for the property and push it when "
                           "it slips, with a bounded side-effect"),
        Strategy("finetune_to_comply", ["finetune"],
                 substrate_ops=["finetune"],
                 rationale="train the behaviour in when it cannot be watched"),
    ],
    Kind.PROHIBITION: [
        Strategy("suppress_topic_feature", ["steer", "monitor"],
                 mech_ops=["read(feature=topic)", "suppress(feature=topic, dose=ec50)"],
                 rationale="the topic is a direction; suppressing it is a runtime write, "
                           "bounded by a side-effect budget so it cannot lobotomize the "
                           "model"),
        Strategy("output_filter", ["monitor"],
                 substrate_ops=["install_filter"],
                 rationale="last resort: block at the output when the internal direction "
                           "cannot be isolated"),
    ],
    Kind.GUARDRAIL: [
        Strategy("amplify_refusal_feature", ["steer", "monitor"],
                 mech_ops=["read(feature=refusal)",
                           "amplify(feature=refusal, dose=ec50)"],
                 rationale="refusal is a measured, dose-responsive direction; amplifying "
                           "it hardens the model without retraining"),
        Strategy("finetune_refusals", ["finetune"],
                 substrate_ops=["finetune"],
                 rationale="train refusal behaviour from examples"),
    ],
}


@dataclass
class Choice:
    capability: Capability
    strategy: Strategy | None
    reason: str
    rejected: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.strategy is not None

    def to_dict(self) -> dict:
        return {
            "capability": self.capability.describe(),
            "kind": self.capability.kind.value,
            "strategy": self.strategy.name if self.strategy else None,
            "mech_ops": self.strategy.mech_ops if self.strategy else [],
            "substrate_ops": self.strategy.substrate_ops if self.strategy else [],
            "reason": self.reason,
            "rejected": [{"strategy": n, "because": w} for n, w in self.rejected],
        }


def select(cap: Capability, sub: SubstrateProfile) -> Choice:
    """Pick the first strategy this substrate can actually run."""
    rejected: list[tuple[str, str]] = []
    for strat in CATALOGUE.get(cap.kind, []):
        missing = [lv for lv in strat.requires if not sub.can(lv)]
        if not missing:
            return Choice(
                cap, strat,
                reason=f"{strat.rationale} (chosen on {sub.id}: "
                       f"{', '.join(strat.requires)} available)",
                rejected=rejected)
        rejected.append((strat.name, sub.why_not(missing[0])))
    return Choice(
        cap, None,
        reason=f"No strategy for {cap.kind.value} is available on {sub.id}. "
               "This is a genuine gap in the compiler, not a property of the program.",
        rejected=rejected)


def plan(caps: list[Capability], sub: SubstrateProfile) -> list[Choice]:
    return [select(c, sub) for c in caps]

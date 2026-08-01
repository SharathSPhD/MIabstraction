"""L0 capability tables — what each substrate actually permits.

This is Loom's equivalent of a target description in a compiler backend. A compiler
targeting two processors does not refuse to build because one lacks an instruction; it
consults the target's description and selects a different lowering with the same meaning.

The decisive asymmetry: on a model you train, every lever is available, including the
architecture itself. On a model someone else trained, the architecture is fixed forever —
so any capability that would have been realized by choosing an architecture or a
pretraining mixture must be reached another way.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Lever:
    """One thing a compiler can do to a model on this substrate."""
    name: str
    available: bool
    cost: float          # relative GPU cost, 1.0 = a fine-tuning run
    note: str = ""


@dataclass(frozen=True)
class SubstrateProfile:
    """Everything the strategy selector needs to know about a target."""
    id: str
    family: str
    levers: dict[str, Lever]
    notes: str = ""

    def can(self, lever: str) -> bool:
        lv = self.levers.get(lever)
        return bool(lv and lv.available)

    def cost(self, lever: str) -> float:
        lv = self.levers.get(lever)
        return lv.cost if lv else float("inf")

    def why_not(self, lever: str) -> str:
        lv = self.levers.get(lever)
        if lv is None:
            return f"'{lever}' is not a lever this compiler knows"
        return lv.note or f"'{lever}' is unavailable on {self.id}"


def _lv(name, avail, cost, note=""):
    return name, Lever(name, avail, cost, note)


FROM_SCRATCH = SubstrateProfile(
    id="scratch",
    family="built here",
    levers=dict([
        _lv("choose_architecture", True, 0.0),
        _lv("choose_tokenizer", True, 0.1),
        _lv("pretrain", True, 10.0, "expensive, but the only way to put knowledge in "
                                    "the weights from nothing"),
        _lv("curriculum", True, 3.0),
        _lv("finetune", True, 1.0),
        _lv("adapter", True, 0.4),
        _lv("install_circuit", True, 0.05, "compiled directly into weights"),
        _lv("steer", True, 0.02, "runtime, no training"),
        _lv("monitor", True, 0.02),
        _lv("retrieval", True, 0.1),
    ]),
    notes="Every lever available: the architecture, the tokenizer and the data are all "
          "choices the compiler gets to make.",
)

OPEN_WEIGHT = SubstrateProfile(
    id="open_weight",
    family="downloaded",
    levers=dict([
        _lv("choose_architecture", False, float("inf"),
            "the architecture was fixed by whoever trained these weights and cannot be "
            "changed without discarding them"),
        _lv("choose_tokenizer", False, float("inf"),
            "the tokenizer is baked into the embedding matrix"),
        _lv("pretrain", False, float("inf"),
            "pretraining from nothing would throw away the weights that make this "
            "substrate worth using"),
        _lv("continued_pretrain", True, 4.0, "extends what the model knows without "
                                             "discarding it"),
        _lv("curriculum", False, float("inf"), "subsumed by continued pretraining here"),
        _lv("finetune", True, 1.0),
        _lv("adapter", True, 0.3, "cheap, composable, leaves the base weights intact"),
        _lv("install_circuit", True, 0.05, "within the circuit's verified envelope"),
        _lv("steer", True, 0.02),
        _lv("monitor", True, 0.02),
        _lv("retrieval", True, 0.1),
    ]),
    notes="The architecture and tokenizer are frozen. Everything the compiler achieves "
          "here it achieves through data, adapters, grafted circuits and steering.",
)

PROFILES = {"scratch": FROM_SCRATCH, "open_weight": OPEN_WEIGHT}


def profile_for(target_spec: dict) -> SubstrateProfile:
    return OPEN_WEIGHT if target_spec.get("kind") == "load" else FROM_SCRATCH

"""The Loom standard library — features and circuits, with their evidence attached.

This is the abstraction layer proper. A Loom program does not manipulate layers, losses,
or learning rates; it manipulates the transformer's own semantic units:

    feature   a named direction in activation space   (the variable)
    circuit   a subgraph that implements a behaviour  (the function)
    steering  a write to the residual stream          (the assignment)

Nothing here is aspirational. Every entry carries the measurement that justifies calling
it a real object, the artifact that measurement lives in, and the conditions under which
it was taken. A unit whose evidence is missing cannot be imported — `require()` raises
rather than letting a program build on something unverified.

Why these three primitives and not others: the project tested whether they correspond to
structure that genuinely exists in a transformer, and reported the answers including the
negative ones (see VALIDATION.md). Features and circuits survived; SAE-derived features
did not earn a place over plain probes, so the library uses probes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[2]


class EvidenceMissing(RuntimeError):
    """A library unit was requested whose supporting measurement is absent."""


@dataclass(frozen=True)
class Evidence:
    """What is known about a unit, and where the number lives."""
    claim: str
    metric: str
    value: float | None
    artifact: str
    conditions: str
    source: str = "MIabstraction"

    def present(self) -> bool:
        return (ROOT / self.artifact).exists()

    def measured(self) -> float | None:
        p = ROOT / self.artifact
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
        cur: Any = d
        for k in self.metric.split("."):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return None
        return cur if isinstance(cur, (int, float)) else None


@dataclass(frozen=True)
class FeatureSpec:
    """A named direction in activation space — the language's variable.

    `extract` says how the direction is obtained: contrastive means of paired
    activations (prayoga's method), a fitted probe, or a supplied vector.
    """
    name: str
    description: str
    extract: Literal["contrast", "probe", "given"]
    evidence: Evidence
    dose: dict = field(default_factory=dict)     # calibration: ec50, safe range
    dual_use: bool = False                        # routed by dual_use policy if True


@dataclass(frozen=True)
class CircuitSpec:
    """A subgraph implementing a behaviour — the language's function.

    `realize` says how the compiler can produce it: `construct` writes weights
    directly (no training), `train` learns it from a curriculum, `find` locates it
    in an existing model.
    """
    name: str
    description: str
    realize: Literal["construct", "train", "find"]
    evidence: Evidence
    envelope: dict = field(default_factory=dict)
    gates: dict = field(default_factory=dict)


# ----------------------------------------------------------------- features

FEATURES: dict[str, FeatureSpec] = {
    "belief_state": FeatureSpec(
        name="belief_state",
        description="The model's running estimate of the hidden state of the process "
                    "it is predicting. Readable linearly from the residual stream.",
        extract="probe",
        evidence=Evidence(
            claim="A transformer trained on a hidden-Markov process linearly encodes "
                  "that process's belief simplex; the incremental signal beyond a "
                  "recent-token baseline is ~100x an untrained control",
            metric="best_incremental_r2",
            value=None,
            artifact="results/final/e1_mess3/result.json",
            conditions="Mess3 process, 4-layer model, held-out sequences, probe fitted "
                       "against exact posteriors; compared against an untrained network "
                       "and an 8-token window baseline",
        ),
    ),
    "refusal": FeatureSpec(
        name="refusal",
        description="The direction mediating refusal behaviour. Low-dimensional and "
                    "dose-responsive; amplifying it hardens a model, suppressing it "
                    "does the opposite.",
        extract="contrast",
        evidence=Evidence(
            claim="Refusal is mediated by a low-dimensional residual direction with a "
                  "shared necessary core across model families; single-direction "
                  "addition is sufficient once its coefficient is calibrated",
            metric="dose_response.ec50",
            value=None,
            artifact="results/loom_port_demo.json",
            conditions="difference-of-means over contrastive prompt pairs; dose-response "
                       "swept with a side-effect budget on unrelated text",
            source="prayoga (F1, F2, F23)",
        ),
        dose={"calibrate": "dose_response", "requires_side_effect_bound": True},
        dual_use=True,
    ),
    "register": FeatureSpec(
        name="register",
        description="Formal versus casual writing register — a neutral concept used to "
                    "verify the feature machinery without touching safety directions.",
        extract="contrast",
        evidence=Evidence(
            claim="A linear probe reads the register off a frozen Llama-3.2-1B's "
                  "residual stream; steering it has a measured dose-response curve "
                  "with a therapeutic window that reverses under over-steering",
            metric="monitor.probe_acc_loo",
            value=None,
            artifact="results/loom_port_demo.json",
            conditions="frozen Llama-3.2-1B, layer -6, leave-one-out cross-validation",
        ),
        dose={"calibrate": "dose_response", "requires_side_effect_bound": True},
    ),
}


# ----------------------------------------------------------------- circuits

CIRCUITS: dict[str, CircuitSpec] = {
    "induction": CircuitSpec(
        name="induction",
        description="In-context copying: when the current context has occurred before, "
                    "continue as it continued then. The canonical transformer circuit.",
        realize="construct",
        evidence=Evidence(
            claim="Compiled directly into attention weights with no training, this "
                  "circuit attends to the correct target more reliably than a trained "
                  "model does, using 13,152 nonzero weights",
            metric="prefix_score",
            value=None,
            artifact="results/loom_constructed_demo.json",
            conditions="3-layer attention-only, trigram matching, zero-mean Hadamard "
                       "codes; verified on variable-gap doubled sequences with a "
                       "negative control on non-repeating input",
        ),
        envelope={"vocab_max": 31, "len_max": 63,
                  "distribution": "token-diverse sequences with genuine repeats"},
        gates={"prefix_score": ">0.5", "icl_argmax_acc": ">0.9"},
    ),
    "state_tracking": CircuitSpec(
        name="state_tracking",
        description="Maintain a belief over a hidden process across a sequence.",
        realize="train",
        evidence=Evidence(
            claim="Trained as one skill among several in a single model, reaching the "
                  "declared probe gate",
            metric="per_skill_metrics.state_tracking.probe_r2",
            value=None,
            artifact="results/loom_curriculum_demo.json",
            conditions="multi-skill curriculum, held-out probe with burn-in",
        ),
        gates={"probe_r2": ">0.9"},
    ),
}


# ------------------------------------------------------------------- access

def require_feature(name: str) -> FeatureSpec:
    """Import a feature, refusing if its evidence is absent."""
    if name not in FEATURES:
        raise KeyError(f"std.features.{name} does not exist. Available: "
                       f"{sorted(FEATURES)}")
    f = FEATURES[name]
    if not f.evidence.present():
        raise EvidenceMissing(
            f"std.features.{name} claims '{f.evidence.claim}' but its artifact "
            f"({f.evidence.artifact}) is not in this repository. A program may not "
            "build on a feature whose evidence cannot be produced.")
    return f


def require_circuit(name: str) -> CircuitSpec:
    """Import a circuit, refusing if its evidence is absent."""
    if name not in CIRCUITS:
        raise KeyError(f"std.circuits.{name} does not exist. Available: "
                       f"{sorted(CIRCUITS)}")
    c = CIRCUITS[name]
    if not c.evidence.present():
        raise EvidenceMissing(
            f"std.circuits.{name} claims '{c.evidence.claim}' but its artifact "
            f"({c.evidence.artifact}) is not in this repository.")
    return c


def catalogue() -> dict:
    """Everything importable, with live measurements — what `loom lib` prints."""
    return {
        "features": {
            n: {"description": f.description, "extract": f.extract,
                "dual_use": f.dual_use, "claim": f.evidence.claim,
                "measured": f.evidence.measured(), "artifact": f.evidence.artifact,
                "source": f.evidence.source}
            for n, f in FEATURES.items()},
        "circuits": {
            n: {"description": c.description, "realize": c.realize,
                "claim": c.evidence.claim, "measured": c.evidence.measured(),
                "artifact": c.evidence.artifact, "gates": c.gates,
                "envelope": c.envelope}
            for n, c in CIRCUITS.items()},
    }

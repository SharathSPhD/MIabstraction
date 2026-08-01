"""The Loom language: weave.yaml -> typed, validated WeaveSpec.

A weave declares WHAT the model must do (skills), what runtime write-access exists
(controls), what read-access exists (monitors), and the evidence required before the
build may succeed (gates). It never mentions how.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SKILL_KINDS = {"induction", "state_tracking", "classify"}
CONTROL_KINDS = {"suppress", "amplify"}
MONITOR_KINDS = {"probe"}
MODEL_SIZES = {
    "small": dict(d_model=64, n_layers=4, n_heads=4, max_len=128),
    "medium": dict(d_model=128, n_layers=6, n_heads=8, max_len=256),
}


class WeaveError(ValueError):
    """A weave that cannot be compiled — with a layman-readable message."""


@dataclass
class Skill:
    name: str
    kind: str
    world: str | None = None      # state_tracking
    concept: str | None = None    # classify
    backend: str = "curriculum"   # curriculum | constructed


@dataclass
class Control:
    name: str
    kind: str
    token: int | None = None
    concept: str | None = None
    strength: float = 1.0


@dataclass
class Monitor:
    name: str
    kind: str
    concept: str


@dataclass
class Gate:
    """One promised metric: e.g. prefix_score > 0.5."""
    target: str          # skill/control name
    metric: str
    op: str              # '>' or '<'
    threshold: float

    def holds(self, value: float) -> bool:
        return value > self.threshold if self.op == ">" else value < self.threshold

    def describe(self) -> str:
        return f"{self.target}.{self.metric} {self.op} {self.threshold}"


_GATE_RE = re.compile(r"^\s*([<>])\s*=?\s*([-+0-9.eE]+)\s*$")

# Metric families with known achievable ranges. A gate whose threshold sits at or
# outside the range is vacuous — it passes (or fails) no matter what the model does,
# turning the promise into theater (red-team FINDING 1A). Loom refuses such gates.
_BOUNDED_01 = re.compile(r"(score|ratio|r2|acc|accuracy|frac|precision|recall)$")
_NONNEG = re.compile(r"(loss|effect|nats|error)$")


def _check_not_vacuous(target: str, metric: str, op: str, thr: float) -> None:
    import math

    if not math.isfinite(thr):
        raise WeaveError(
            f"Gate '{target}.{metric} {op} {thr}' has a non-finite threshold."
        )
    m = metric.rsplit("_", 1)[-1]
    if _BOUNDED_01.search(metric) or _BOUNDED_01.search(m):
        if op == ">" and not (0 < thr < 1):
            raise WeaveError(
                f"Gate '{target}.{metric} > {thr}' is vacuous or impossible: "
                f"'{metric}' lives in [0, 1], so the threshold must be strictly "
                "inside (0, 1)."
            )
        if op == "<" and not (0 < thr < 1):
            raise WeaveError(
                f"Gate '{target}.{metric} < {thr}' is vacuous or impossible for a "
                f"[0, 1] metric."
            )
    elif _NONNEG.search(metric) or _NONNEG.search(m):
        if thr <= 0:
            raise WeaveError(
                f"Gate '{target}.{metric} {op} {thr}': '{metric}' is non-negative, "
                "so this gate can never be informative."
            )
    else:
        # Unknown metric family: refuse the obviously-vacuous shapes.
        if op == ">" and thr < 0:
            raise WeaveError(
                f"Gate '{target}.{metric} > {thr}' is vacuous: metrics in Loom are "
                "non-negative, so any measurement would pass. Declare a real bar."
            )


def parse_gate_expr(target: str, metric: str, expr: str) -> Gate:
    m = _GATE_RE.match(str(expr))
    if not m:
        raise WeaveError(
            f"Gate '{target}.{metric}: {expr}' is not understood. "
            "Write gates like '>0.9' or '<0.1'."
        )
    op, thr = m.group(1), float(m.group(2))
    _check_not_vacuous(target, metric, op, thr)
    return Gate(target=target, metric=metric, op=op, threshold=thr)


@dataclass
class WeaveSpec:
    model: dict
    skills: list[Skill] = field(default_factory=list)
    controls: list[Control] = field(default_factory=list)
    monitors: list[Monitor] = field(default_factory=list)
    gates: list[Gate] = field(default_factory=list)
    seed: int = 0
    name: str = "weave"

    def names(self) -> set[str]:
        return (
            {s.name for s in self.skills}
            | {c.name for c in self.controls}
            | {m.name for m in self.monitors}
        )

    def gates_for(self, target: str) -> list[Gate]:
        return [g for g in self.gates if g.target == target]


def _build_items(raw: list | None, cls, kinds: set[str], what: str):
    out = []
    for item in raw or []:
        kind = item.get("kind")
        if kind not in kinds:
            raise WeaveError(
                f"{what} '{item.get('name', '?')}' has kind '{kind}'. "
                f"Loom v0 knows: {sorted(kinds)}."
            )
        try:
            out.append(cls(**item))
        except TypeError as e:
            raise WeaveError(f"{what} '{item.get('name', '?')}': {e}") from e
    return out


def load_weave(path: str | Path) -> WeaveSpec:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise WeaveError("A weave file must be a YAML mapping.")

    model_raw = raw.get("model", {"size": "small"})
    size = model_raw.get("size", "small")
    if size not in MODEL_SIZES:
        raise WeaveError(f"model.size must be one of {sorted(MODEL_SIZES)}.")
    model = dict(MODEL_SIZES[size])

    spec = WeaveSpec(
        model=model,
        skills=_build_items(raw.get("skills"), Skill, SKILL_KINDS, "Skill"),
        controls=_build_items(raw.get("controls"), Control, CONTROL_KINDS, "Control"),
        monitors=_build_items(raw.get("monitors"), Monitor, MONITOR_KINDS, "Monitor"),
        seed=int(raw.get("seed", 0)),
        name=str(raw.get("name", Path(path).stem.split(".")[0])),
    )

    names = spec.names()
    if len(names) < len(spec.skills) + len(spec.controls) + len(spec.monitors):
        raise WeaveError("Every skill, control, and monitor needs a unique name.")

    for target, metrics in (raw.get("gates") or {}).items():
        if target not in names:
            raise WeaveError(
                f"Gate target '{target}' does not match any declared skill/control/"
                f"monitor. Declared: {sorted(names)}."
            )
        for metric, expr in metrics.items():
            spec.gates.append(parse_gate_expr(target, metric, expr))

    # The promise is not optional: every skill and control must be gated.
    ungated = [
        n for n in ({s.name for s in spec.skills} | {c.name for c in spec.controls})
        if not spec.gates_for(n)
    ]
    if ungated:
        raise WeaveError(
            f"No gates declared for: {sorted(ungated)}. Loom refuses to build "
            "capabilities it is not required to verify."
        )
    return spec

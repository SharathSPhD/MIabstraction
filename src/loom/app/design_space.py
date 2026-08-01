"""L1.5 — the design space the compiler searches, and what the program is allowed to say
about it.

Compiling C is deterministic. Bringing a model into existence is not: the same program,
built twice with different learning rates, gives you two different models, and only
measurement tells you which one you wanted. So a Loom build is a search, and this module
is the space that search runs in.

The whole point is the division of labour. There are two kinds of number here and the
language keeps them apart:

  - What the *programmer* knows: how far they are willing to let the model be changed,
    how hard behaviour may be pushed, how long they are willing to wait. These are the
    `tune` and `effort` clauses. They are stated in consequences.

  - What the *compiler* knows: that "willing to change it a fair amount" means a LoRA
    rank near 8 rather than 2, that steering strength should be searched log-spaced
    because its effect is multiplicative, that a behaviour is usually legible in the
    upper-middle third of the stack. These are the lever specs below. They are stated in
    hyperparameters.

A programmer who writes `tune adaptation from 1 to 8` has said something true and
checkable about their intent. They have not said anything about rank, alpha, learning
rate or step count, and they should not have to — the same way `-O2` says how hard to
optimize without naming a register allocator.

Only levers the compiler actually turns are listed. If a knob is not searched, it is not
in this table, because a design space that advertises levers nobody turns is a brochure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LeverSpec:
    """One searchable dimension, and the programmer-facing knob that bounds it."""
    name: str                  # what the compiler calls it
    stage: str                 # steering | adaptation | pretraining
    why: str                   # what turning it does
    low: float
    high: float
    scale: str = "log"         # log | linear | int_log | depth_fraction
    knob: str | None = None    # the `tune` name that may narrow it, if any
    min_points: int = 1        # below this, it is not a search

    def describe(self) -> str:
        bound = f"bounded by `tune {self.knob}`" if self.knob else "the compiler's own"
        return f"{self.name} ({self.stage}): {self.why} — {bound}"


# The levers, grouped by the stage of the build that turns them.
#
# `layer` deserves a word: which layer a behaviour is legible at is not something the
# programmer can know, and it is not something the compiler can derive either. It is
# found by looking. That is why no `tune` clause bounds it — there is no honest way to
# state it in consequences, so the compiler owns it outright.
DESIGN_SPACE: tuple[LeverSpec, ...] = (
    LeverSpec("layer", "steering",
              "where in the stack the behaviour is legible enough to write to",
              low=0.35, high=0.85, scale="depth_fraction", knob=None, min_points=2),
    LeverSpec("multiplier", "steering",
              "how hard to push, as a multiple of the direction's own scale",
              low=0.5, high=4.0, scale="log", knob="steering", min_points=2),

    LeverSpec("rank", "adaptation",
              "how much capacity the adapter has to change behaviour",
              low=1, high=8, scale="int_log", knob="adaptation", min_points=1),
    LeverSpec("lr", "adaptation",
              "how far each step moves the adapter",
              low=2e-5, high=3e-4, scale="log", knob="adaptation", min_points=1),
    LeverSpec("steps", "adaptation",
              "how long to train the adapter",
              low=30, high=240, scale="int_log", knob="patience", min_points=1),

    LeverSpec("lr", "pretraining",
              "how far each step moves the weights themselves",
              low=1e-6, high=1e-5, scale="log", knob="adaptation", min_points=1),
    LeverSpec("steps", "pretraining",
              "how much of the corpus to walk through",
              low=20, high=120, scale="int_log", knob="patience", min_points=1),
)


# What a `tune` name means, as a dial the programmer turns. These are the only numbers a
# program is allowed to contain, and they are deliberately dimensionless: "how far the
# model may be changed" is a thing a person has a view about, and a learning rate is not.
#
# The domain matters because one knob bounds several levers at once — `adaptation` sets
# both the adapter's rank and how far each step moves it. Those live in incommensurate
# units, so the knob cannot be intersected with them directly. Instead the program's
# range is read as a *position* on the dial, and each lever maps that position onto its
# own range. This is why `tune adaptation from 1 to 4` means "at most half as much
# change" on every lever it touches, rather than "a learning rate of 4".
KNOB_DOMAINS: dict[str, tuple[float, float]] = {
    "adaptation": (1.0, 8.0),    # how far the model itself may be moved
    "steering": (0.5, 4.0),      # how hard behaviour may be pushed at runtime
    "patience": (1.0, 10.0),     # how long the build may spend
}


def levers_for(stage: str) -> list[LeverSpec]:
    return [lv for lv in DESIGN_SPACE if lv.stage == stage]


def _position(knob: str, declared) -> tuple[float, float]:
    """Where on the dial the program asked to sit, as a fraction of the knob's domain."""
    lo_d, hi_d = KNOB_DOMAINS[knob]
    lo = max(lo_d, min(hi_d, float(declared[0])))
    hi = max(lo_d, min(hi_d, float(declared[1])))
    if hi < lo:
        lo, hi = hi, lo
    span = hi_d - lo_d
    if span <= 0:
        return 0.0, 1.0
    return (lo - lo_d) / span, (hi - lo_d) / span


def _at(lv: LeverSpec, t: float) -> float:
    """The lever's value at fraction t of its own range, spaced the way it behaves."""
    if lv.scale in ("log", "int_log"):
        a, b = math.log(max(lv.low, 1e-12)), math.log(max(lv.high, 1e-12))
        return math.exp(a + (b - a) * t)
    return lv.low + (lv.high - lv.low) * t


def _points(low: float, high: float, n: int, scale: str) -> list:
    """n values from low to high, spaced the way this lever's effect actually behaves."""
    if n <= 1:
        # One point means "don't search this" — take the conservative end, which for
        # every lever here is the low one: the smallest change that might work.
        return [int(round(low)) if scale == "int_log" else low]
    if scale in ("log", "int_log"):
        lo, hi = math.log(max(low, 1e-12)), math.log(max(high, 1e-12))
        vals = [math.exp(lo + (hi - lo) * i / (n - 1)) for i in range(n)]
    else:
        vals = [low + (high - low) * i / (n - 1) for i in range(n)]
    if scale == "int_log":
        seen, out = set(), []
        for v in vals:
            iv = int(round(v))
            if iv not in seen:
                seen.add(iv)
                out.append(iv)
        return out
    return [round(v, 10) for v in vals]


def _depth_points(low: float, high: float, n: int, n_layers: int) -> list[int]:
    """Fractions of depth resolved to negative layer indices.

    Negative because the program is written once and built on models of different
    depths: "three-quarters of the way up" has to mean the same thing on a 16-layer
    model and a 28-layer one, and counting from the top is how that survives.
    """
    fr = _points(low, high, max(n, 2), "linear")
    idx = sorted({max(1, min(n_layers - 1, int(round(f * n_layers)))) for f in fr})
    return [i - n_layers for i in idx]


def grids(stage: str, budget: dict, n_layers: int = 16) -> dict[str, list]:
    """Turn the program's declared budget into the values this stage will actually try.

    `budget` is what `App.search_budget()` returned: an effort level, which sets how many
    points each lever gets, and any `tune` bounds, which narrow a lever's range. A bound
    the compiler does not recognise is ignored rather than fatal — the program stays
    valid against a compiler that knows about more levers than the one it was written
    for, which is the same reason an unknown pragma is a warning and not an error.
    """
    per = int(budget.get("trials_per_lever", 2))
    bounds = budget.get("bounds", {})
    out: dict[str, list] = {}

    for lv in levers_for(stage):
        low, high = lv.low, lv.high
        if lv.knob and lv.knob in KNOB_DOMAINS and lv.knob in bounds:
            b = bounds[lv.knob]
            if isinstance(b, (tuple, list)) and len(b) == 2 and \
                    all(isinstance(x, (int, float)) for x in b):
                # The program turned this dial down. Read it as a position rather than a
                # value, and clamp to the knob's domain first: a program may ask for less
                # of the space, never for more than the compiler will deliver safely.
                t_lo, t_hi = _position(lv.knob, b)
                low, high = _at(lv, t_lo), _at(lv, t_hi)

        n = max(lv.min_points, per)
        if lv.scale == "depth_fraction":
            out[lv.name] = _depth_points(low, high, n, n_layers)
        else:
            out[lv.name] = _points(low, high, n, lv.scale)
    return out


def unrecognised(budget: dict) -> list[str]:
    """`tune` names in the program that no lever is bound by — reported, not fatal."""
    known = {lv.knob for lv in DESIGN_SPACE if lv.knob} & set(KNOB_DOMAINS)
    return sorted(set(budget.get("bounds", {})) - known)


def explain(budget: dict, n_layers: int = 16) -> str:
    """What the compiler will search, given this program. Printed by `loom explain`."""
    lines = [f"search effort: {budget.get('effort', 'balanced')} "
             f"({budget.get('trials_per_lever', 2)} values per lever)"]
    for stage in ("steering", "adaptation", "pretraining"):
        g = grids(stage, budget, n_layers)
        if not g:
            continue
        lines.append(f"  {stage}:")
        for lv in levers_for(stage):
            vals = g[lv.name]
            shown = ", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in vals)
            src = (f"tune {lv.knob}" if lv.knob and lv.knob in budget.get("bounds", {})
                   else "default")
            lines.append(f"    {lv.name:<11} {shown:<34} [{src}]  {lv.why}")
    for name in unrecognised(budget):
        lines.append(f"  note: `tune {name}` bounds no lever this compiler knows; ignored")
    total = 1
    for lv in levers_for("steering"):
        total *= len(grids("steering", budget, n_layers)[lv.name])
    lines.append(f"  -> {total} configurations per behavioural capability")
    return "\n".join(lines)

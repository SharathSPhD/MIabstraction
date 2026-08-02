"""L2 — the capability graph: what the model must do, with no architecture in sight.

An application program says what a person wants. This module is its meaning: a set of
capabilities the compiler must realize, stated so that nothing about a transformer,
a training procedure, or a substrate has leaked in yet. That separation is what allows
one program to be built on a model you train and a model you download.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Kind(str, Enum):
    KNOWLEDGE = "knowledge"       # must know this material
    SKILL = "skill"               # must be able to do this
    STYLE = "style"               # must speak this way
    INVARIANT = "invariant"       # must always hold
    # Policy. Declared by the program, carried into the artifact, and compiled to
    # NO weight change and NO steering write.
    #
    # TRIZ, asked to resolve "the same output must refuse (off-subject) and must not
    # refuse (in-subject)", ranks separation on condition above every structural
    # answer and returns Taking Out with Intermediary. The measurements agree: every
    # mechanism strong enough to make a model refuse off-subject questions also made
    # it refuse its own — Counsel, built to a 0.5 refusal margin, declined "what does
    # a motion to dismiss test?". A model never trained to refuse cannot refuse a
    # legal question, so the failure disappears by construction rather than by
    # tuning. Scope belongs to an intermediary that reads the request before the
    # model does; here that is a declared policy, and prabodha is the component that
    # enforces it.
    PROHIBITION = "prohibition"   # policy: must never do
    GUARDRAIL = "guardrail"       # policy: must refuse this class of request
    TUNING = "tuning"             # how hard, and within what bounds, to search


@dataclass
class Capability:
    kind: Kind
    name: str
    args: dict = field(default_factory=dict)
    source_line: int = 0

    def describe(self) -> str:
        if self.kind is Kind.KNOWLEDGE:
            return f"know the material in {self.args.get('corpus', '?')}"
        if self.kind is Kind.SKILL:
            return f"be able to {self.name}"
        if self.kind is Kind.STYLE:
            return f"speak {', '.join(self.args.get('traits', [self.name]))}"
        if self.kind is Kind.INVARIANT:
            return f"always {self.name}"
        if self.kind is Kind.PROHIBITION:
            return f"never {self.name}"
        if self.kind is Kind.TUNING:
            if self.name == "effort":
                return f"search {self.args['effort']}"
            if "range" in self.args:
                lo, hi = self.args["range"]
                return f"keep {self.name} between {lo:g} and {hi:g}"
            return f"try {self.name}: {', '.join(self.args.get('choices', []))}"
        return f"refuse {self.name}"


@dataclass
class Expectation:
    """An acceptance test written by the author, measured after the build."""
    kind: str            # "answers" | "refuses" | "knows_better"
    probe: str
    contains: str | None = None
    source_line: int = 0

    @property
    def measured_not_generated(self) -> bool:
        """True when this is checked against a measurement rather than by asking the
        model something and reading the reply."""
        return self.kind == "knows_better"

    def describe(self) -> str:
        if self.kind == "refuses":
            return f'refuses "{self.probe}"'
        if self.kind == "knows_better":
            return "predicts held-out material better than the base model did"
        return f'answering "{self.probe}" mentions "{self.contains}"'


@dataclass
class App:
    name: str
    capabilities: list[Capability] = field(default_factory=list)
    expectations: list[Expectation] = field(default_factory=list)

    def of(self, kind: Kind) -> list[Capability]:
        return [c for c in self.capabilities if c.kind is kind]

    POLICY_KINDS = (Kind.PROHIBITION, Kind.GUARDRAIL)

    def to_realize(self) -> list[Capability]:
        """Capabilities the compiler must build.

        Tuning clauses direct the search rather than being something to realize.
        Policy clauses are not built into the model at all — see Kind.PROHIBITION.
        """
        return [c for c in self.capabilities
                if c.kind is not Kind.TUNING and c.kind not in self.POLICY_KINDS]

    def policies(self) -> list[Capability]:
        """What the program declared about scope, for the intermediary that enforces
        it. These travel with the artifact; they never change a weight."""
        return [c for c in self.capabilities if c.kind in self.POLICY_KINDS]

    def search_budget(self) -> dict:
        """How hard to search, and within what bounds, in the programmer's terms.

        The point of putting this in the language is that the design space really is
        broad and the person building the app has views about it — how much they are
        willing to change the model, how long they are willing to wait — without having
        any view about learning rates. Effort maps to how many configurations are tried;
        a named bound maps to the range of one lever.
        """
        effort, bounds = "balanced", {}
        for c in self.of(Kind.TUNING):
            if c.name == "effort":
                effort = c.args["effort"]
            elif "range" in c.args:
                bounds[c.name] = tuple(c.args["range"])
            elif "choices" in c.args:
                bounds[c.name] = c.args["choices"]
        grid = {"quick": 1, "balanced": 2, "thorough": 4}[effort]
        return {"effort": effort, "trials_per_lever": grid, "bounds": bounds}

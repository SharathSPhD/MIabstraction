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
    PROHIBITION = "prohibition"   # must never do
    GUARDRAIL = "guardrail"       # must refuse this class of request
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
    kind: str            # "answers" | "refuses"
    probe: str
    contains: str | None = None
    source_line: int = 0

    def describe(self) -> str:
        if self.kind == "refuses":
            return f'refuses "{self.probe}"'
        return f'answering "{self.probe}" mentions "{self.contains}"'


@dataclass
class App:
    name: str
    capabilities: list[Capability] = field(default_factory=list)
    expectations: list[Expectation] = field(default_factory=list)

    def of(self, kind: Kind) -> list[Capability]:
        return [c for c in self.capabilities if c.kind is kind]

    def to_realize(self) -> list[Capability]:
        """Capabilities the compiler must build. Tuning clauses are excluded: they
        direct the search rather than being something to realize."""
        return [c for c in self.capabilities if c.kind is not Kind.TUNING]

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

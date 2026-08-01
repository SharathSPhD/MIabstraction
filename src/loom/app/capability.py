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

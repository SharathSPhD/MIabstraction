"""Type system for Loom language."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class Type:
    """Base class for types."""

    pass


@dataclass
class SimpleType(Type):
    """Simple types: target, model, etc."""

    name: str  # 'target', 'corpus', 'tokenizer', 'model', 'evalset', 'unit'

    def __str__(self) -> str:
        return self.name


@dataclass
class CorpusType(Type):
    """Corpus with optional kind: corpus, corpus[chat], corpus[pref]."""

    kind: Optional[str] = None  # None, 'chat', 'pref'

    def __str__(self) -> str:
        if self.kind:
            return f"corpus[{self.kind}]"
        return "corpus"

    def matches(self, other: Type) -> bool:
        """Check if this corpus type matches another (for function args)."""
        if not isinstance(other, CorpusType):
            return False
        # Plain corpus matches anything, specific kind only matches itself
        if self.kind is None:
            return True
        return self.kind == other.kind


@dataclass
class TupleType(Type):
    """Tuple type: (corpus, corpus)."""

    types: list[Type]

    def __str__(self) -> str:
        return "(" + ", ".join(str(t) for t in self.types) + ")"


@dataclass
class UnionType(Type):
    """Union type: corpus | evalset."""

    types: list[Type]

    def __str__(self) -> str:
        return " | ".join(str(t) for t in self.types)


@dataclass
class NumberType(Type):
    """Number type."""

    name: str = "number"

    def __str__(self) -> str:
        return "number"


# Singleton instances for common types
TARGET_TYPE = SimpleType("target")
FEATURE_TYPE = SimpleType("feature")
CIRCUIT_TYPE = SimpleType("circuit")
MONITOR_TYPE = SimpleType("monitor")
CORPUS_TYPE = CorpusType(kind=None)
CORPUS_CHAT_TYPE = CorpusType(kind="chat")
CORPUS_PREF_TYPE = CorpusType(kind="pref")
TOKENIZER_TYPE = SimpleType("tokenizer")
MODEL_TYPE = SimpleType("model")
EVALSET_TYPE = SimpleType("evalset")
UNIT_TYPE = SimpleType("unit")
NUMBER_TYPE = NumberType()


class TypeError(Exception):
    """Type checking error."""

    pass


def types_match(expected: Type, actual: Type) -> bool:
    """Check if actual type matches expected type."""
    if isinstance(expected, CorpusType) and isinstance(actual, CorpusType):
        return expected.matches(actual)
    if isinstance(expected, SimpleType) and isinstance(actual, SimpleType):
        return expected.name == actual.name
    if isinstance(expected, UnionType):
        return any(types_match(t, actual) for t in expected.types)
    return str(expected) == str(actual)

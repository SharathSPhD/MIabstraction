"""Corpus algebra for Loom stages — data as composable first-class values.

This module implements the data layer described in LANGUAGE.md:
  - corpus algebra: text/chat/prefs, mixing with weights, filtering, dedup, split
  - distinct corpus kinds (text, chat, pref) enforced by type
  - contiguous splits to prevent leakage
  - real loaders for cached HF datasets and local JSONL
"""

from __future__ import annotations

import hashlib
import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Optional, Tuple

import numpy as np


# ============================================================================
# Corpus Types (Kind System)
# ============================================================================

CorpusKind = Literal["text", "chat", "pref"]


# ============================================================================
# Core Corpus Protocol
# ============================================================================


class Corpus(ABC):
    """Abstract base for a corpus — a stream of structured data."""

    kind: CorpusKind

    @abstractmethod
    def iter_batches(self, batch_size: int = 1) -> Iterator[list[dict]]:
        """Yield batches of records from this corpus."""
        pass

    @abstractmethod
    def size_estimate(self) -> int:
        """Estimated number of examples (for progress tracking)."""
        pass

    # Corpus algebra operations
    def __add__(self, other: Corpus) -> Corpus:
        """Concatenate two corpora (must be same kind)."""
        if self.kind != other.kind:
            raise TypeError(f"Cannot mix corpus kinds: {self.kind} + {other.kind}")
        return ConcatCorpus([self, other])

    def __mul__(self, weight: float) -> Corpus:
        """Scale corpus by a weight (for mixing)."""
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Weight must be in [0, 1], got {weight}")
        return WeightedCorpus(self, weight)

    def __rmul__(self, weight: float) -> Corpus:
        """Right multiply (weight * corpus)."""
        return self.__mul__(weight)

    def filter(self, pred: Callable[[dict], bool]) -> Corpus:
        """Filter records by predicate."""
        return FilteredCorpus(self, pred)

    def dedup(self, ngram: int = 13) -> Corpus:
        """Deduplicate by n-gram hashing (sliding window)."""
        return DedupCorpus(self, ngram)

    def shuffle(self, seed: int = 0) -> Corpus:
        """Shuffle (in-memory, only for small corpora)."""
        return ShuffledCorpus(self, seed)

    def split(
        self, frac: float = 0.9
    ) -> Tuple[Corpus, Corpus]:
        """Contiguous split (no shuffled leakage).

        Returns (train, eval) with frac fraction going to train.
        The split is deterministic and contiguous: records 0 to N*frac go to train,
        N*frac to N go to eval.
        """
        return ContiguousSplitCorpus.split_corpus(self, frac)


# ============================================================================
# Concrete Corpus Implementations
# ============================================================================


@dataclass
class TextCorpus(Corpus):
    """A corpus of plain text lines (corpus kind: text)."""

    kind: CorpusKind = "text"
    records: list[dict] = None  # list of {"text": ...}

    def __post_init__(self):
        if self.records is None:
            self.records = []

    def iter_batches(self, batch_size: int = 1) -> Iterator[list[dict]]:
        for i in range(0, len(self.records), batch_size):
            yield self.records[i : i + batch_size]

    def size_estimate(self) -> int:
        return len(self.records)


@dataclass
class ChatCorpus(Corpus):
    """A corpus of chat/instruction pairs (corpus kind: chat).

    Records have schema: {"prompt": str, "response": str}
    """

    kind: CorpusKind = "chat"
    records: list[dict] = None  # list of {"prompt": ..., "response": ...}

    def __post_init__(self):
        if self.records is None:
            self.records = []
        # Validate schema
        for i, rec in enumerate(self.records):
            if not ("prompt" in rec and "response" in rec):
                raise ValueError(
                    f"Chat record {i} missing 'prompt' or 'response': {rec}"
                )

    def iter_batches(self, batch_size: int = 1) -> Iterator[list[dict]]:
        for i in range(0, len(self.records), batch_size):
            yield self.records[i : i + batch_size]

    def size_estimate(self) -> int:
        return len(self.records)


@dataclass
class PrefCorpus(Corpus):
    """A corpus of preference pairs (corpus kind: pref).

    Records have schema: {"prompt": str, "chosen": str, "rejected": str}
    """

    kind: CorpusKind = "pref"
    records: list[dict] = None  # list of {"prompt": ..., "chosen": ..., "rejected": ...}

    def __post_init__(self):
        if self.records is None:
            self.records = []
        # Validate schema
        for i, rec in enumerate(self.records):
            required = {"prompt", "chosen", "rejected"}
            if not required.issubset(rec.keys()):
                raise ValueError(
                    f"Preference record {i} missing required fields {required}: {rec}"
                )

    def iter_batches(self, batch_size: int = 1) -> Iterator[list[dict]]:
        for i in range(0, len(self.records), batch_size):
            yield self.records[i : i + batch_size]

    def size_estimate(self) -> int:
        return len(self.records)


# ============================================================================
# Corpus Operations (Lazy Composition)
# ============================================================================


class ConcatCorpus(Corpus):
    """Concatenation of multiple corpora (same kind)."""

    def __init__(self, corpora: list[Corpus]):
        if not corpora:
            raise ValueError("Cannot create empty ConcatCorpus")
        kinds = set(c.kind for c in corpora)
        if len(kinds) > 1:
            raise TypeError(f"Cannot concatenate corpora of different kinds: {kinds}")
        self.corpora = corpora
        self.kind = corpora[0].kind

    def iter_batches(self, batch_size: int = 1) -> Iterator[list[dict]]:
        for corpus in self.corpora:
            for batch in corpus.iter_batches(batch_size):
                yield batch

    def size_estimate(self) -> int:
        return sum(c.size_estimate() for c in self.corpora)


class WeightedCorpus(Corpus):
    """A corpus with a normalization weight (for mixing)."""

    def __init__(self, corpus: Corpus, weight: float):
        self.corpus = corpus
        self.weight = weight
        self.kind = corpus.kind
        self._normalized_weight: Optional[float] = None

    def set_normalized_weight(self, w: float):
        """Called during mixing normalization."""
        self._normalized_weight = w

    def iter_batches(self, batch_size: int = 1) -> Iterator[list[dict]]:
        # Weighted corpus repeats according to its normalized weight
        for _ in range(max(1, int(self._normalized_weight * 100))):
            for batch in self.corpus.iter_batches(batch_size):
                yield batch

    def size_estimate(self) -> int:
        if self._normalized_weight is None:
            return self.corpus.size_estimate()
        return max(1, int(self.corpus.size_estimate() * self._normalized_weight))


class FilteredCorpus(Corpus):
    """A corpus filtered by a predicate."""

    def __init__(self, corpus: Corpus, pred: Callable[[dict], bool]):
        self.corpus = corpus
        self.pred = pred
        self.kind = corpus.kind

    def iter_batches(self, batch_size: int = 1) -> Iterator[list[dict]]:
        batch = []
        for record in self._iter_records():
            batch.append(record)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _iter_records(self) -> Iterator[dict]:
        for batch in self.corpus.iter_batches(1):
            for record in batch:
                if self.pred(record):
                    yield record

    def size_estimate(self) -> int:
        # Rough estimate: assume filter keeps ~50% (can be overridden)
        return self.corpus.size_estimate() // 2


class DedupCorpus(Corpus):
    """Deduplication by n-gram hashing (sliding window)."""

    def __init__(self, corpus: Corpus, ngram: int = 13):
        self.corpus = corpus
        self.ngram = ngram
        self.kind = corpus.kind

    def iter_batches(self, batch_size: int = 1) -> Iterator[list[dict]]:
        seen_hashes = set()
        batch = []
        for record in self._iter_deduplicated_records():
            batch.append(record)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _iter_deduplicated_records(self) -> Iterator[dict]:
        seen_hashes = set()
        for batch in self.corpus.iter_batches(1):
            for record in batch:
                # Extract text from record (handles text, chat, pref)
                text = None
                if "text" in record:
                    text = record["text"]
                elif "prompt" in record:
                    text = record["prompt"] + " " + record["response"]
                if text is None:
                    yield record
                    continue

                # Hash sliding n-grams
                tokens = text.split()
                doc_hashes = set()
                for i in range(max(0, len(tokens) - self.ngram + 1)):
                    ngram_str = " ".join(tokens[i : i + self.ngram])
                    h = hashlib.md5(ngram_str.encode()).digest()
                    doc_hashes.add(h)

                # Keep if n-gram is new
                if not (doc_hashes & seen_hashes):
                    yield record
                    seen_hashes.update(doc_hashes)

    def size_estimate(self) -> int:
        # Assume dedup removes ~20-30% (depends on corpus)
        return int(self.corpus.size_estimate() * 0.75)


class ShuffledCorpus(Corpus):
    """Shuffled corpus (loads all into memory)."""

    def __init__(self, corpus: Corpus, seed: int = 0):
        self.corpus = corpus
        self.seed = seed
        self.kind = corpus.kind
        self._shuffled_records: Optional[list] = None

    def _ensure_shuffled(self):
        if self._shuffled_records is None:
            records = []
            for batch in self.corpus.iter_batches(1):
                records.extend(batch)
            rng = random.Random(self.seed)
            rng.shuffle(records)
            self._shuffled_records = records

    def iter_batches(self, batch_size: int = 1) -> Iterator[list[dict]]:
        self._ensure_shuffled()
        for i in range(0, len(self._shuffled_records), batch_size):
            yield self._shuffled_records[i : i + batch_size]

    def size_estimate(self) -> int:
        return self.corpus.size_estimate()


class ContiguousSplitCorpus:
    """Contiguous (non-shuffled) split of a corpus."""

    @staticmethod
    def split_corpus(corpus: Corpus, frac: float = 0.9) -> Tuple[Corpus, Corpus]:
        """Split corpus into (train, eval) with no shuffled leakage.

        Args:
            corpus: The corpus to split
            frac: Fraction for training (e.g., 0.9 for 90% train, 10% eval)

        Returns:
            (train_corpus, eval_corpus) where eval is a contiguous tail
        """
        if not 0.0 < frac < 1.0:
            raise ValueError(f"Split fraction must be in (0, 1), got {frac}")

        # Load all records
        all_records = []
        for batch in corpus.iter_batches(1):
            all_records.extend(batch)

        # Split at boundary (contiguous)
        split_idx = int(len(all_records) * frac)
        train_records = all_records[:split_idx]
        eval_records = all_records[split_idx:]

        # Return appropriate corpus types
        if corpus.kind == "text":
            return TextCorpus(records=train_records), TextCorpus(records=eval_records)
        elif corpus.kind == "chat":
            return ChatCorpus(records=train_records), ChatCorpus(records=eval_records)
        elif corpus.kind == "pref":
            return PrefCorpus(records=train_records), PrefCorpus(records=eval_records)
        else:
            raise ValueError(f"Unknown corpus kind: {corpus.kind}")


# ============================================================================
# Data Loaders (Real HF Datasets & Local JSONL)
# ============================================================================


def text(spec: str) -> TextCorpus:
    """Load a text corpus from a dataset spec.

    Specs:
      - "babylm:childes" — BabyLM CHILDES subset
      - "babylm:switchboard" — BabyLM Switchboard subset
      - etc.
      - "local:/path/to/file.jsonl" — local JSONL with {"text": ...} records
    """
    if spec.startswith("local:"):
        path = Path(spec[6:])
        return _load_jsonl_text(path)
    elif spec.startswith("babylm:"):
        subset = spec[7:]
        return _load_babylm_subset(subset)
    else:
        raise ValueError(f"Unknown text corpus spec: {spec}")


def chat(path: str) -> ChatCorpus:
    """Load a chat corpus from a JSONL file.

    File format: lines of JSON with {"prompt": ..., "response": ...}
    """
    return _load_jsonl_chat(Path(path))


def prefs(path: str) -> PrefCorpus:
    """Load a preference corpus from a JSONL file.

    File format: lines of JSON with {"prompt": ..., "chosen": ..., "rejected": ...}
    """
    return _load_jsonl_pref(Path(path))


def _load_jsonl_text(path: Path) -> TextCorpus:
    """Load text corpus from JSONL."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records.append(rec)
    return TextCorpus(records=records)


def _load_jsonl_chat(path: Path) -> ChatCorpus:
    """Load chat corpus from JSONL."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records.append(rec)
    return ChatCorpus(records=records)


def _load_jsonl_pref(path: Path) -> PrefCorpus:
    """Load preference corpus from JSONL."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records.append(rec)
    return PrefCorpus(records=records)


def _load_babylm_subset(subset: str) -> TextCorpus:
    """Load a subset of the BabyLM-2026-Strict corpus."""
    import datasets

    dataset = datasets.load_dataset("BabyLM-community/BabyLM-2026-Strict", subset)
    records = []
    for example in dataset["train"]:
        records.append({"text": example["text"]})
    return TextCorpus(records=records)


# ============================================================================
# Mixing (Corpus Algebra with Normalization)
# ============================================================================


class MixedCorpus(Corpus):
    """Mix of weighted corpora (normalized)."""

    def __init__(self, weighted_corpora: list[WeightedCorpus]):
        if not weighted_corpora:
            raise ValueError("Cannot create empty MixedCorpus")

        # Validate all same kind
        kinds = set(c.kind for c in weighted_corpora)
        if len(kinds) > 1:
            raise TypeError(f"Cannot mix corpora of different kinds: {kinds}")

        self.kind = weighted_corpora[0].kind

        # Extract weights and normalize
        weights = [c.weight for c in weighted_corpora]
        weight_sum = sum(weights)
        if weight_sum <= 0:
            raise ValueError(f"Sum of weights must be positive, got {weight_sum}")

        # Normalize and set on each
        normalized_weights = [w / weight_sum for w in weights]
        self.corpora = []
        for wc, nw in zip(weighted_corpora, normalized_weights):
            wc.set_normalized_weight(nw)
            self.corpora.append(wc)

    def iter_batches(self, batch_size: int = 1) -> Iterator[list[dict]]:
        # Concatenate all weighted corpora (each expanded by normalized weight)
        for corpus in self.corpora:
            for batch in corpus.iter_batches(batch_size):
                yield batch

    def size_estimate(self) -> int:
        return sum(c.size_estimate() for c in self.corpora)


def mix(*weighted_corpora_with_weights) -> MixedCorpus:
    """Mix corpora with weights.

    Usage:
      mix(corpus_a * 0.7, corpus_b * 0.3)
      mix(text("babylm:childes") * 0.6, text("babylm:wiki") * 0.4)
    """
    # Extract WeightedCorpus objects
    wc_list = [wc for wc in weighted_corpora_with_weights if isinstance(wc, WeightedCorpus)]
    if not wc_list:
        raise ValueError("mix() requires at least one weighted corpus")
    return MixedCorpus(wc_list)

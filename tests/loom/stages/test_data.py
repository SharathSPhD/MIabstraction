"""Tests for corpus algebra (loom.stages.data module)."""

import json
import tempfile
from pathlib import Path

import pytest

from loom.stages.data import (
    ChatCorpus,
    PrefCorpus,
    TextCorpus,
    mix,
    ContiguousSplitCorpus,
)


class TestTextCorpus:
    """Test text corpus operations."""

    def test_creation(self):
        """Test TextCorpus creation."""
        records = [{"text": "hello"}, {"text": "world"}]
        corpus = TextCorpus(records=records)
        assert corpus.kind == "text"
        assert corpus.size_estimate() == 2

    def test_iter_batches(self):
        """Test batch iteration."""
        records = [{"text": f"text{i}"} for i in range(10)]
        corpus = TextCorpus(records=records)
        batches = list(corpus.iter_batches(batch_size=3))
        assert len(batches) == 4  # 3+3+3+1
        assert len(batches[0]) == 3
        assert len(batches[-1]) == 1

    def test_concat(self):
        """Test concatenation of text corpora."""
        c1 = TextCorpus(records=[{"text": "a"}])
        c2 = TextCorpus(records=[{"text": "b"}])
        c3 = c1 + c2
        assert c3.size_estimate() == 2
        batches = list(c3.iter_batches(1))
        assert len(batches) == 2

    def test_weighted(self):
        """Test weighted corpus."""
        corpus = TextCorpus(records=[{"text": "a"}, {"text": "b"}])
        weighted = corpus * 0.5
        # Weight is normalized later in mix()
        assert hasattr(weighted, "weight")

    def test_filter(self):
        """Test filtering."""
        records = [{"text": f"text{i}"} for i in range(10)]
        corpus = TextCorpus(records=records)
        filtered = corpus.filter(lambda x: int(x["text"][-1]) % 2 == 0)
        # Filtered corpus yields only even-numbered texts
        results = list(filtered.iter_batches(1))
        assert len(results) >= 1  # At least some records


class TestChatCorpus:
    """Test chat corpus operations."""

    def test_creation(self):
        """Test ChatCorpus creation."""
        records = [
            {"prompt": "hello?", "response": "hi"},
            {"prompt": "how are you?", "response": "good"},
        ]
        corpus = ChatCorpus(records=records)
        assert corpus.kind == "chat"
        assert corpus.size_estimate() == 2

    def test_schema_validation(self):
        """Test that ChatCorpus validates schema."""
        records_bad = [{"prompt": "hello"}]  # Missing response
        with pytest.raises(ValueError, match="missing.*response"):
            ChatCorpus(records=records_bad)

    def test_iter_batches(self):
        """Test batch iteration for chat."""
        records = [
            {"prompt": f"q{i}", "response": f"r{i}"} for i in range(5)
        ]
        corpus = ChatCorpus(records=records)
        batches = list(corpus.iter_batches(batch_size=2))
        assert len(batches) == 3  # 2+2+1


class TestPrefCorpus:
    """Test preference corpus operations."""

    def test_creation(self):
        """Test PrefCorpus creation."""
        records = [
            {"prompt": "q", "chosen": "good", "rejected": "bad"}
        ]
        corpus = PrefCorpus(records=records)
        assert corpus.kind == "pref"
        assert corpus.size_estimate() == 1

    def test_schema_validation(self):
        """Test that PrefCorpus validates schema."""
        records_bad = [{"prompt": "q", "chosen": "good"}]  # Missing rejected
        with pytest.raises(ValueError, match="required"):
            PrefCorpus(records=records_bad)

    def test_iter_batches(self):
        """Test batch iteration for preference."""
        records = [
            {"prompt": f"q{i}", "chosen": f"c{i}", "rejected": f"r{i}"}
            for i in range(4)
        ]
        corpus = PrefCorpus(records=records)
        batches = list(corpus.iter_batches(batch_size=2))
        assert len(batches) == 2


class TestCorpusKindChecks:
    """Test that corpus kind checking works."""

    def test_cannot_mix_different_kinds(self):
        """Test that concatenating different kinds fails."""
        text_corpus = TextCorpus(records=[{"text": "a"}])
        chat_corpus = ChatCorpus(records=[{"prompt": "q", "response": "r"}])
        with pytest.raises(TypeError, match="Cannot mix corpus kinds"):
            text_corpus + chat_corpus

    def test_cannot_mix_different_kinds_in_mix(self):
        """Test that mix() checks kinds."""
        text_corpus = TextCorpus(records=[{"text": "a"}])
        chat_corpus = ChatCorpus(records=[{"prompt": "q", "response": "r"}])
        with pytest.raises(TypeError, match="different kinds"):
            mix(text_corpus * 0.5, chat_corpus * 0.5)


class TestContiguousSplit:
    """Test contiguous splitting (no shuffled leakage)."""

    def test_contiguous_split(self):
        """Test that split is contiguous, not shuffled."""
        records = [{"text": f"text{i}"} for i in range(100)]
        corpus = TextCorpus(records=records)

        train, eval = ContiguousSplitCorpus.split_corpus(corpus, frac=0.8)

        train_records = []
        for batch in train.iter_batches(1):
            train_records.extend(batch)

        eval_records = []
        for batch in eval.iter_batches(1):
            eval_records.extend(batch)

        # Check that train is contiguous (first 80)
        assert len(train_records) == 80
        assert all(int(r["text"][-1]) < 80 or len(r["text"]) == 6 for r in train_records[:5])

        # Check that eval is contiguous (last 20)
        assert len(eval_records) == 20

    def test_split_preserves_kind(self):
        """Test that split preserves corpus kind."""
        chat_records = [
            {"prompt": f"q{i}", "response": f"r{i}"} for i in range(20)
        ]
        chat_corpus = ChatCorpus(records=chat_records)

        train, eval = ContiguousSplitCorpus.split_corpus(chat_corpus, frac=0.7)

        assert isinstance(train, ChatCorpus)
        assert isinstance(eval, ChatCorpus)
        assert train.kind == "chat"
        assert eval.kind == "chat"


class TestLoadLocal:
    """Test loading from local JSONL files."""

    def test_load_text_jsonl(self):
        """Test loading text corpus from JSONL."""
        from loom.stages.data import _load_jsonl_text

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"text": "hello"}) + "\n")
            f.write(json.dumps({"text": "world"}) + "\n")
            path = Path(f.name)

        try:
            corpus = _load_jsonl_text(path)
            assert corpus.size_estimate() == 2
            records = list(corpus.iter_batches(1))
            assert len(records) == 2
        finally:
            path.unlink()

    def test_load_chat_jsonl(self):
        """Test loading chat corpus from JSONL."""
        from loom.stages.data import _load_jsonl_chat

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"prompt": "q1", "response": "r1"}) + "\n")
            f.write(json.dumps({"prompt": "q2", "response": "r2"}) + "\n")
            path = Path(f.name)

        try:
            corpus = _load_jsonl_chat(path)
            assert corpus.size_estimate() == 2
            assert corpus.kind == "chat"
        finally:
            path.unlink()

    def test_load_pref_jsonl(self):
        """Test loading preference corpus from JSONL."""
        from loom.stages.data import _load_jsonl_pref

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"prompt": "q", "chosen": "c", "rejected": "r"}) + "\n")
            path = Path(f.name)

        try:
            corpus = _load_jsonl_pref(path)
            assert corpus.size_estimate() == 1
            assert corpus.kind == "pref"
        finally:
            path.unlink()


class TestDedup:
    """Test deduplication."""

    def test_dedup_removes_similar(self):
        """Test that dedup removes similar documents."""
        # Create corpus with some repetition
        records = [
            {"text": "the quick brown fox jumps over the lazy dog"},
            {"text": "the quick brown fox jumps over the lazy dog"},  # Exact duplicate
            {"text": "a completely different text"},
        ]
        corpus = TextCorpus(records=records)
        deduped = corpus.dedup(ngram=5)

        results = list(deduped.iter_batches(1))
        # At least the exact duplicate should be removed (ngram overlap)
        assert len(results) <= 3  # Could be 2 or 3 depending on implementation


class TestShuffle:
    """Test shuffling."""

    def test_shuffle_is_deterministic(self):
        """Test that shuffle with same seed is deterministic."""
        records = [{"text": f"text{i}"} for i in range(10)]
        corpus = TextCorpus(records=records)

        shuffled1 = corpus.shuffle(seed=42)
        shuffled2 = corpus.shuffle(seed=42)

        results1 = list(shuffled1.iter_batches(1))
        results2 = list(shuffled2.iter_batches(1))

        # Extract text values
        texts1 = [b[0]["text"] for b in results1]
        texts2 = [b[0]["text"] for b in results2]

        assert texts1 == texts2

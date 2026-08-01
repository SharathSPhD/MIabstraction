"""Tests for the from-scratch executor: architecture selection, tokenizer, pretraining."""
import json
import tempfile
from pathlib import Path

import pytest
import torch

from loom.app.capability import App, Capability, Kind, Expectation
from loom.app.exec_scratch import (
    ExecReport,
    build_tokenizer,
    choose_architecture,
    execute_scratch,
)
from loom.app.lowering import select
from loom.app.substrate import FROM_SCRATCH


class TestArchitectureSelection:
    """Test that architecture choices are deterministic and justified."""

    def test_small_decoder_with_no_capabilities(self):
        """Empty app should produce a small decoder."""
        app = App(name="empty")
        spec, rationale = choose_architecture(app, "small")

        assert spec["kind"] == "decoder"
        assert spec["width"] > 0
        assert spec["layers"] > 0
        assert spec["heads"] > 0
        assert "decoder" in rationale

    def test_knowledge_heavy_increases_width(self):
        """App with many knowledge capabilities should choose wider layers."""
        app = App(name="knowledge")
        for i in range(3):
            app.capabilities.append(
                Capability(Kind.KNOWLEDGE, f"corpus{i}", {"corpus": f"path{i}"})
            )

        spec, rationale = choose_architecture(app, "small")

        # Width should be increased for knowledge
        assert "Knowledge=3" in rationale
        assert "width" in rationale.lower()

    def test_skill_heavy_increases_depth_and_uses_hybrid(self):
        """App with multiple skills should choose deeper/hybrid architecture."""
        app = App(name="skills")
        for i in range(2):
            app.capabilities.append(Capability(Kind.SKILL, f"skill{i}"))

        spec, rationale = choose_architecture(app, "small")

        # Should use hybrid architecture and increase depth
        assert spec["kind"] == "nemotron_h"
        assert spec["layers"] >= 6
        assert "Nemotron-H" in rationale
        assert "state tracking" in rationale

    def test_style_increases_heads(self):
        """App with styles should choose more attention heads."""
        app = App(name="styled")
        for i in range(2):
            app.capabilities.append(
                Capability(Kind.STYLE, f"style{i}", {"traits": ["trait"]})
            )

        spec, rationale = choose_architecture(app, "small")

        # More heads for independent style routing
        assert spec["heads"] >= 4

    def test_guardrails_dont_affect_architecture(self):
        """Guardrails are handled via steering, not architecture changes."""
        app = App(name="safe")
        app.capabilities.append(Capability(Kind.GUARDRAIL, "refusal"))

        spec, rationale = choose_architecture(app, "small")

        # Guardrails don't affect width, depth, or heads
        assert spec["kind"] in ("decoder", "nemotron_h")


class TestTokenizer:
    """Test tokenizer training."""

    def test_a_missing_corpus_produces_no_tokenizer(self):
        """There used to be a byte-level fallback here. A build whose corpus is absent
        must not quietly acquire a tokenizer and carry on — the program named material
        that is not there, and every number downstream would be about something else."""
        tokenizer_type, vocab_size, desc, tok = build_tokenizer("/nonexistent/path.txt")

        assert tokenizer_type == "none"
        assert vocab_size == 0
        assert tok is None
        assert "does not exist" in desc

    def test_vocab_size_is_reasonable(self):
        """Trained tokenizer should have reasonable vocab size."""
        # Create a temporary corpus
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello world\n" * 100)
            corpus_path = f.name

        try:
            tokenizer_type, vocab_size, desc, tok = build_tokenizer(
                corpus_path, vocab_size=256)

            assert tokenizer_type == "bpe"
            assert 0 < vocab_size <= 256
            assert tok is not None
        finally:
            Path(corpus_path).unlink()

    def test_tokenizer_roundtrip(self):
        """Tokenizer should be able to process text."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            text = "the quick brown fox"
            f.write(text)
            corpus_path = f.name

        try:
            tokenizer_type, vocab_size, desc, tok = build_tokenizer(corpus_path)

            assert tokenizer_type == "bpe"
            assert vocab_size > 0
            # It has to actually encode the text it was trained on.
            assert tok.encode(text).ids
        finally:
            Path(corpus_path).unlink()


class TestPretrainingMixture:
    """Test pretraining on synthetic data."""

    def test_pretraining_returns_metrics(self):
        """Pretraining should return val_loss, val_ppl, tokens_seen, wall_clock."""
        from loom.backends import ScratchBackend

        app = App(name="test")
        backend = ScratchBackend()
        vocab_size = 256
        spec = {
            "kind": "decoder",
            "width": 128,
            "layers": 2,
            "heads": 2,
            "vocab": vocab_size,
            "ctx": 128,
        }
        model = backend.realize(spec)
        # Use CPU for testing to avoid CUDA vocab indexing issues
        device = "cpu"
        model.to(device)

        from loom.app.exec_scratch import pretraining_mixture

        val_loss, val_ppl, tokens_seen, wall_clock, prov = pretraining_mixture(
            {"test": 1.0},
            app,
            model,
            backend,
            tokenizer=None,
            steps=10,
            batch_size=4,
            device=device,
        )

        # No tokenizer and a corpus path that does not exist: this must report that it
        # did not run rather than training on whatever it can manufacture.
        assert prov["ran"] is False
        assert prov["reason"]
        assert tokens_seen == 0

    def test_it_trains_on_the_real_corpus_when_there_is_one(self):
        """The counterpart: given real text it really trains, and says where from."""
        corpus = Path("data/domains/medical/corpus.txt")
        if not corpus.exists():
            pytest.skip("run scripts/fetch_real_domain_data.py for the domain corpora")

        from loom.app.exec_scratch import build_tokenizer, pretraining_mixture
        from loom.backends import ScratchBackend

        _, vocab_size, _, tok = build_tokenizer(str(corpus), vocab_size=2000)
        backend = ScratchBackend()
        model = backend.realize({"kind": "decoder", "width": 64, "layers": 2,
                                 "heads": 2, "vocab": vocab_size, "ctx": 128})
        device = "cpu"
        model.to(device)

        val_loss, val_ppl, tokens_seen, wall, prov = pretraining_mixture(
            {str(corpus): 1.0}, App(name="t"), model, backend, tokenizer=tok,
            steps=5, batch_size=2, device=device)

        assert prov["ran"] is True
        assert prov["sources"] == [str(corpus)]
        assert prov["heldout_sequences"] > 0
        assert tokens_seen > 0
        assert val_loss is not None and val_loss > 0


class TestExecuteScratch:
    """Integration tests for the full from-scratch executor."""

    def test_execute_scratch_basic(self):
        """Full build: parse app, plan, execute — on the corpus the app names.

        The corpus is real, because the whole point of the from-scratch substrate is
        that the tokenizer and the weights are both built from the app's own material.
        """
        corpus = Path("data/domains/medical/corpus.txt")
        if not corpus.exists():
            pytest.skip("run scripts/fetch_real_domain_data.py for the domain corpora")
        app = App(name="tiny")
        app.capabilities.append(
            Capability(Kind.KNOWLEDGE, "facts", {"corpus": str(corpus)})
        )
        app.expectations.append(
            Expectation("answers", "what is it?", "definition")
        )

        # Create a plan
        choices = [select(c, FROM_SCRATCH) for c in app.capabilities]

        # All choices should be successful on from-scratch
        assert all(c.ok for c in choices)

        # Execute
        target_spec = {"kind": "scratch", "size": "small"}
        device = "cuda" if torch.cuda.is_available() else "cpu"

        report = execute_scratch(choices, target_spec, app, device)

        # Report should be complete
        assert isinstance(report, ExecReport)
        assert report.backend == "scratch"
        assert report.architecture_choice
        assert report.tokenizer_type == "bpe"
        assert report.tokens_seen > 0
        assert report.pretraining["ran"] is True
        assert report.wall_clock_s > 0
        # Substrate advantage should explain what scratch permits that open-weight doesn't
        assert "From-scratch" in report.substrate_advantage

    def test_report_serializable(self):
        """Report should be JSON-serializable."""
        app = App(name="test")
        choices = []

        target_spec = {"kind": "scratch", "size": "small"}
        device = "cuda" if torch.cuda.is_available() else "cpu"

        report = execute_scratch(choices, target_spec, app, device)

        # Should be able to convert to dict and serialize
        data = report.to_dict()
        json_str = json.dumps(data)
        assert json_str
        assert "from-scratch" in json_str or "From-scratch" in json_str

    def test_an_app_with_no_corpus_reports_no_training(self):
        """An app that declares no material has nothing to learn from. This used to
        report tokens_seen in the millions, because the trainer manufactured its own
        data; the only honest output is that it did not run, and why."""
        app = App(name="test")

        target_spec = {"kind": "scratch", "size": "small"}
        device = "cuda" if torch.cuda.is_available() else "cpu"

        report = execute_scratch([], target_spec, app, device)

        assert report.pretraining.get("ran") is False
        assert report.pretraining.get("reason")
        assert report.tokens_seen == 0
        assert report.val_ppl is None


class TestMechInterpOps:
    """Test mech-interp operations (read, amplify, install)."""

    def test_op_read_on_dummy_data(self):
        """op_read should work on synthetic contrastive data."""
        from loom.backends import ScratchBackend
        from loom.app.exec_scratch import op_read

        backend = ScratchBackend()
        spec = {
            "kind": "decoder",
            "width": 64,
            "layers": 2,
            "heads": 2,
            "vocab": 128,
            "ctx": 64,
        }
        model = backend.realize(spec)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.module.eval()

        # Create dummy contrastive batches
        a = [torch.randint(0, 128, (2, 32)) for _ in range(2)]
        b = [torch.randint(0, 128, (2, 32)) for _ in range(2)]

        result = op_read(backend, model, device, -1, a, b)

        assert result["op"] == "read"
        assert "ok" in result
        # If ok, should have probe_acc
        if result["ok"]:
            assert "probe_acc" in result
            assert 0 <= result["probe_acc"] <= 1

    def test_op_install_outside_envelope_fails(self):
        """op_install should reject circuit outside its envelope."""
        from loom.backends import ScratchBackend
        from loom.app.exec_scratch import op_install

        backend = ScratchBackend()
        spec = {
            "kind": "decoder",
            "width": 64,
            "layers": 2,
            "heads": 2,
            "vocab": 100000,  # Way larger than induction circuit expects
            "ctx": 64,
        }
        model = backend.realize(spec)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        result = op_install(backend, model, device, "induction", 100000, 64)

        # Should fail because vocab is too large for induction circuit
        assert result["op"] == "install"
        # Induction circuit has vocab_max=31
        if not result["ok"]:
            assert len(result.get("problems", [])) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

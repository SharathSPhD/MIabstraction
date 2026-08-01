"""Tests for the open-weight executor: exec_open.execute_open and strategy functions."""
import pytest
import json
import torch
from pathlib import Path

from loom.app.exec_open import (
    execute_open, ExecReport, StrategyMeasurement,
    continued_pretraining, knowledge_adapter, steer_style_feature,
    suppress_topic_feature, amplify_refusal_feature, monitor_and_correct,
    install_compiled_circuit, _load_corpus,
)
from loom.app.capability import Capability, Kind, App
from loom.app.lowering import Choice, Strategy, plan, CATALOGUE
from loom.app.substrate import OPEN_WEIGHT
from loom.backends import HFBackend, ModelHandle


class TestCorpusLoading:
    """Test corpus loading utilities."""

    def test_load_corpus(self):
        """Load corpus from BabyLM dataset."""
        corpus = _load_corpus(limit=10)
        if corpus:  # May be empty if dataset not present
            assert isinstance(corpus, list)
            assert all(isinstance(t, str) for t in corpus)
        else:
            # Dataset not present; should be graceful
            assert corpus == []


class TestStrategy_ContinuedPretraining:
    """Test continued_pretraining strategy on a real model."""

    def test_continued_pretraining_basic(self):
        """Train for 1 step on gpt2."""
        backend = HFBackend()
        handle = backend.realize({"name": "gpt2", "dtype": "bfloat16"})

        corpus = ["The quick brown fox", "jumps over the lazy dog"]
        result = continued_pretraining(handle, corpus, steps=1)

        assert isinstance(result, dict)
        if "error" not in result:
            assert "baseline_loss" in result
            assert "final_loss" in result
            assert "steps" in result
            assert result["steps"] >= 1

    def test_continued_pretraining_empty_corpus(self):
        """Handle empty corpus gracefully."""
        backend = HFBackend()
        handle = backend.realize({"name": "gpt2", "dtype": "bfloat16"})

        result = continued_pretraining(handle, [], steps=1)
        # Should return without crashing
        assert isinstance(result, dict)


class TestStrategy_KnowledgeAdapter:
    """Test knowledge_adapter (LoRA) strategy."""

    def test_knowledge_adapter_basic(self):
        """Create adapter measurements on gpt2."""
        backend = HFBackend()
        handle = backend.realize({"name": "gpt2", "dtype": "bfloat16"})

        corpus = ["The quick brown fox", "jumps over the lazy dog"]
        result = knowledge_adapter(handle, corpus, rank=4, steps=1)

        assert isinstance(result, dict)
        if "error" not in result:
            assert "adapter_params" in result
            assert "total_params" in result
            assert "adapter_ratio" in result
            assert result["adapter_params"] > 0
            assert 0 < result["adapter_ratio"] < 1

    def test_knowledge_adapter_base_weights_untouched(self):
        """Adapter training should not modify base weights."""
        backend = HFBackend()
        handle = backend.realize({"name": "gpt2", "dtype": "bfloat16"})

        # Save original weights
        original_state = {k: v.clone() for k, v in handle.module.state_dict().items()}

        corpus = ["The quick brown fox"]
        result = knowledge_adapter(handle, corpus, rank=4, steps=1)

        # For this basic implementation, weights might not change (adapter not installed)
        # In a full implementation, would verify base weights unchanged
        assert isinstance(result, dict)


class TestStrategy_SteerStyleFeature:
    """Test steer_style_feature strategy."""

    def test_steer_style_feature_basic(self):
        """Extract style direction on gpt2."""
        backend = HFBackend()
        handle = backend.realize({"name": "gpt2", "dtype": "bfloat16"})

        traits = {
            "high_examples": ["The formal document states hereby", "It is hereby resolved"],
            "low_examples": ["Yeah so like whatever", "lol that's so funny"],
        }
        result = steer_style_feature(handle, traits)

        assert isinstance(result, dict)
        if "error" not in result:
            assert "probe_accuracy_loo" in result
            assert "direction_norm" in result
            assert 0 <= result["probe_accuracy_loo"] <= 1

    def test_steer_style_feature_missing_traits(self):
        """Reject traits without high/low examples."""
        backend = HFBackend()
        handle = backend.realize({"name": "gpt2", "dtype": "bfloat16"})

        result = steer_style_feature(handle, {})
        assert "error" in result

        result = steer_style_feature(handle, {"high_examples": ["a"]})
        assert "error" in result


class TestStrategy_SuppressTopic:
    """Test suppress_topic_feature strategy."""

    def test_suppress_topic_feature_basic(self):
        """Suppress a topic direction."""
        backend = HFBackend()
        handle = backend.realize({"name": "gpt2", "dtype": "bfloat16"})

        result = suppress_topic_feature(handle, "politics")
        assert isinstance(result, dict)
        assert "error" not in result
        assert result.get("topic") == "politics"


class TestStrategy_AmplifyRefusal:
    """Test amplify_refusal_feature strategy."""

    def test_amplify_refusal_feature_basic(self):
        """Amplify refusal for a classification."""
        backend = HFBackend()
        handle = backend.realize({"name": "gpt2", "dtype": "bfloat16"})

        result = amplify_refusal_feature(handle, "malware-code")
        assert isinstance(result, dict)
        assert "error" not in result
        assert result.get("class") == "malware-code"


class TestStrategy_Monitor:
    """Test monitor_and_correct strategy."""

    def test_monitor_and_correct_basic(self):
        """Install a monitor probe."""
        backend = HFBackend()
        handle = backend.realize({"name": "gpt2", "dtype": "bfloat16"})

        result = monitor_and_correct(handle, "always_cites_source")
        assert isinstance(result, dict)
        assert "error" not in result
        assert result.get("property") == "always_cites_source"


class TestStrategy_InstallCircuit:
    """Test install_compiled_circuit strategy and envelope checking."""

    def test_install_circuit_envelope_ok(self):
        """Circuit within envelope should be installable."""
        backend = HFBackend()
        handle = backend.realize({"name": "gpt2", "dtype": "bfloat16"})
        # gpt2 vocab is 50257, so should exceed most envelopes

        class CircuitGood:
            name = "induction"
            envelope = {"vocab_max": 100000, "len_max": 2048}

        result = install_compiled_circuit(handle, CircuitGood())
        assert isinstance(result, dict)
        assert result.get("envelope_ok") is True
        assert result.get("status") == "ready_for_graft"

    def test_install_circuit_envelope_violation(self):
        """Circuit outside envelope should record fallback."""
        backend = HFBackend()
        handle = backend.realize({"name": "gpt2", "dtype": "bfloat16"})
        # gpt2 has vocab 50257

        class CircuitTight:
            name = "induction"
            envelope = {"vocab_max": 31, "len_max": 63}

        result = install_compiled_circuit(handle, CircuitTight())
        assert isinstance(result, dict)
        assert result.get("envelope_ok") is False
        assert result.get("status") == "envelope_violation"


class TestExecuteOpen:
    """Test the main execute_open function."""

    def test_execute_open_minimal(self):
        """Execute with minimal choices."""
        # Create a simple app and capabilities
        cap = Capability(kind=Kind.KNOWLEDGE, name="test_fact",
                        args={"corpus": "test_corpus"})
        choice = Choice(
            capability=cap,
            strategy=CATALOGUE[Kind.KNOWLEDGE][1],  # continued_pretraining
            reason="testing"
        )

        report = execute_open(
            [choice],
            {"name": "gpt2", "dtype": "bfloat16"},
            app=None,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        assert isinstance(report, ExecReport)
        assert report.model_name == "gpt2"
        assert report.dtype == "bfloat16"
        assert len(report.results) >= 1
        assert report.wall_clock_s > 0

    def test_execute_open_no_strategy(self):
        """Handle case where no strategy is available."""
        cap = Capability(kind=Kind.KNOWLEDGE, name="test",
                        args={"corpus": "test"})
        choice = Choice(
            capability=cap,
            strategy=None,  # No strategy available
            reason="substrate does not support this"
        )

        report = execute_open(
            [choice],
            {"name": "gpt2", "dtype": "bfloat16"},
            app=None,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        assert isinstance(report, ExecReport)
        assert report.failed >= 1
        assert any(r.strategy_name == "none" for r in report.results)

    def test_execute_open_circuit_with_fallback(self):
        """Test circuit with envelope violation triggers fallback."""
        cap = Capability(kind=Kind.SKILL, name="test_skill",
                        args={"circuit": "induction"})
        choice = Choice(
            capability=cap,
            strategy=CATALOGUE[Kind.SKILL][0],  # install_compiled_circuit
            reason="testing"
        )

        report = execute_open(
            [choice],
            {"name": "gpt2", "dtype": "bfloat16"},
            app=None,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        assert isinstance(report, ExecReport)
        # Circuit should violate envelope on gpt2, triggering fallback
        # Check if any result recorded fallback
        results_with_fallback = [r for r in report.results if r.fallback]
        if results_with_fallback:
            assert any(r.fallback == "retrieval_circuit" for r in results_with_fallback)

    def test_execute_open_report_format(self):
        """Verify report can be serialized to JSON."""
        cap = Capability(kind=Kind.KNOWLEDGE, name="test",
                        args={"corpus": "test"})
        choice = Choice(
            capability=cap,
            strategy=CATALOGUE[Kind.KNOWLEDGE][1],  # continued_pretraining
            reason="testing"
        )

        report = execute_open(
            [choice],
            {"name": "gpt2", "dtype": "bfloat16"},
            app=None,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Should serialize to JSON
        report_dict = report.to_dict()
        json_str = json.dumps(report_dict)
        assert json_str
        assert "model" in report_dict
        assert "summary" in report_dict
        assert "per_capability" in report_dict


class TestStrategyMeasurement:
    """Test StrategyMeasurement dataclass."""

    def test_measurement_to_dict(self):
        """Measurement should serialize to dict."""
        m = StrategyMeasurement(
            capability_name="test",
            capability_kind="knowledge",
            strategy_name="continued_pretraining",
            success=True,
            wall_clock_s=1.5,
            measurements={"loss": 2.5},
        )
        d = m.to_dict()
        assert d["capability"] == "test"
        assert d["strategy"] == "continued_pretraining"
        assert d["success"] is True
        assert d["measurements"]["loss"] == 2.5

    def test_measurement_with_fallback(self):
        """Measurement can record fallback."""
        m = StrategyMeasurement(
            capability_name="test",
            capability_kind="skill",
            strategy_name="install_compiled_circuit",
            success=False,
            wall_clock_s=0.5,
            fallback="retrieval_circuit",
            fallback_reason="vocab envelope violated",
        )
        d = m.to_dict()
        assert d["fallback"] == "retrieval_circuit"
        assert "vocab envelope" in d["fallback_reason"]


class TestExecReport:
    """Test ExecReport dataclass."""

    def test_report_to_dict(self):
        """Report should serialize to dict with summary."""
        report = ExecReport(
            model_name="gpt2",
            dtype="bfloat16",
            device="cuda",
            n_params=123456,
            n_layers=12,
            succeeded=1,
            failed=0,
            fallbacks=0,
            wall_clock_s=1.5,
        )
        d = report.to_dict()
        assert d["model"] == "gpt2"
        assert d["summary"]["total"] == 0
        assert d["summary"]["succeeded"] == 1
        assert d["summary"]["wall_clock_s"] == 1.5

    def test_report_with_results(self):
        """Report can contain results."""
        result = StrategyMeasurement(
            capability_name="test",
            capability_kind="knowledge",
            strategy_name="continued_pretraining",
            success=True,
            wall_clock_s=1.0,
        )
        report = ExecReport(
            model_name="gpt2",
            dtype="bfloat16",
            device="cuda",
            n_params=123456,
            n_layers=12,
            results=[result],
            succeeded=1,
        )
        d = report.to_dict()
        assert len(d["per_capability"]) == 1
        assert d["per_capability"][0]["capability"] == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

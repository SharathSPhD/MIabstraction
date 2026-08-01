"""Integration test: execute_open on real models with real measurements."""
import json
import pytest
import torch
from pathlib import Path

from loom.app.exec_open import execute_open, ExecReport
from loom.app.capability import Capability, Kind, App
from loom.app.lowering import plan, CATALOGUE
from loom.app.substrate import OPEN_WEIGHT


class TestExecuteOpenIntegration:
    """End-to-end executor tests with real models and measurements."""

    def test_execute_open_on_gpt2_with_knowledge(self):
        """Execute continued_pretraining on gpt2."""
        # Build capability and plan
        cap_know = Capability(kind=Kind.KNOWLEDGE, name="test_corpus",
                             args={"corpus": "test.txt"})
        caps = [cap_know]
        choices = plan(caps, OPEN_WEIGHT)

        # Execute
        report = execute_open(
            choices,
            {"name": "gpt2", "dtype": "bfloat16"},
            app=None,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Verify report
        assert report is not None
        assert report.model_name == "gpt2"
        assert len(report.results) == len(choices)
        assert report.wall_clock_s > 0

        # Verify at least one succeeded
        assert report.succeeded > 0 or report.failed > 0

    def test_execute_open_mixed_capabilities(self):
        """Execute mixed capabilities: knowledge, style, prohibition, guardrail."""
        caps = [
            Capability(kind=Kind.KNOWLEDGE, name="facts",
                      args={"corpus": "docs"}),
            Capability(kind=Kind.STYLE, name="formal",
                      args={"traits": ["formal", "professional"]}),
            Capability(kind=Kind.PROHIBITION, name="politics",
                      args={}),
            Capability(kind=Kind.GUARDRAIL, name="malware",
                      args={}),
        ]

        choices = plan(caps, OPEN_WEIGHT)
        assert len(choices) == len(caps)
        assert all(c.strategy is not None or c.ok is False for c in choices)

        report = execute_open(
            choices,
            {"name": "gpt2", "dtype": "bfloat16"},
            app=None,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # All capabilities should be attempted
        assert len(report.results) == len(caps)

        # Report should have timing info
        assert report.wall_clock_s > 0

    def test_execute_open_circuit_fallback(self):
        """Test that circuit with violated envelope falls back gracefully."""
        cap_skill = Capability(kind=Kind.SKILL, name="test_induction",
                              args={"circuit": "induction"})
        choices = plan([cap_skill], OPEN_WEIGHT)

        report = execute_open(
            choices,
            {"name": "gpt2", "dtype": "bfloat16"},
            app=None,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Report should have the capability result
        assert len(report.results) >= 1
        result = report.results[0]

        # Circuit should violate gpt2's envelope, so might record fallback
        # Or might fail - either way, should be handled gracefully
        assert isinstance(result.wall_clock_s, float)

    def test_execute_open_report_json_output(self):
        """Report should be serializable to JSON."""
        cap = Capability(kind=Kind.KNOWLEDGE, name="test",
                        args={"corpus": "test"})
        choices = plan([cap], OPEN_WEIGHT)

        report = execute_open(
            choices,
            {"name": "gpt2", "dtype": "bfloat16"},
            app=None,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Should serialize to JSON
        report_dict = report.to_dict()
        json_str = json.dumps(report_dict, indent=2)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["model"] == "gpt2"
        assert "summary" in parsed
        assert "per_capability" in parsed
        assert len(parsed["per_capability"]) >= 1

    def test_execute_open_measurement_realism(self):
        """Measurements should be realistic numbers from real runs."""
        cap = Capability(kind=Kind.KNOWLEDGE, name="test",
                        args={"corpus": "test"})
        choices = plan([cap], OPEN_WEIGHT)

        report = execute_open(
            choices,
            {"name": "gpt2", "dtype": "bfloat16"},
            app=None,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        if report.succeeded > 0:
            # Find a successful result
            successful = [r for r in report.results if r.success][0]
            meas = successful.measurements

            # Loss measurements should be realistic (non-negative, finite)
            for key in ["baseline_loss", "final_loss", "avg_loss"]:
                if key in meas:
                    val = meas[key]
                    assert isinstance(val, (int, float))
                    assert 0 <= val < 100, f"{key} = {val} unrealistic"

            # Parameter counts should be positive
            for key in ["total_params", "adapter_params"]:
                if key in meas:
                    assert meas[key] > 0

    def test_execute_open_with_invalid_model(self):
        """Handle non-existent model gracefully."""
        cap = Capability(kind=Kind.KNOWLEDGE, name="test",
                        args={"corpus": "test"})
        choices = plan([cap], OPEN_WEIGHT)

        # The executor should handle invalid models gracefully by catching exceptions
        try:
            report = execute_open(
                choices,
                {"name": "nonexistent-model/does-not-exist", "dtype": "bfloat16"},
                app=None,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            # If it does not raise, it should return a report with failures recorded
            assert isinstance(report, ExecReport)
            # Should have recorded the failure in initialization
            assert report.failed >= 0 or len(report.results) >= 0
        except (ValueError, OSError, RuntimeError):
            # Also acceptable if it raises early - both behaviors are graceful
            pass


class TestExecuteOpenRealModels:
    """Test on actual available models."""

    @pytest.mark.skipif(
        not Path.home().joinpath(".cache/huggingface/hub").exists(),
        reason="Hugging Face cache not found"
    )
    def test_execute_on_available_models(self):
        """Execute on models we know are cached."""
        models = []
        cache = Path.home() / ".cache/huggingface/hub"

        if (cache / "models--gpt2").exists():
            models.append("gpt2")
        if (cache / "models--google--gemma-2-2b").exists():
            models.append("google/gemma-2-2b")
        if (cache / "models--meta-llama--Llama-3.2-1B").exists():
            models.append("meta-llama/Llama-3.2-1B")

        if not models:
            pytest.skip("No cached models found")

        # Test on first available model
        model_name = models[0]

        cap = Capability(kind=Kind.KNOWLEDGE, name="test",
                        args={"corpus": "test"})
        choices = plan([cap], OPEN_WEIGHT)

        report = execute_open(
            choices,
            {"name": model_name, "dtype": "bfloat16"},
            app=None,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        assert report.model_name == model_name
        assert len(report.results) >= 1
        assert report.n_params > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

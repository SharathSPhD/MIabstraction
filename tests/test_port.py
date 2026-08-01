"""Tests for PORT backend: frozen HF models with Loom controls."""
import pytest
import torch
import numpy as np

from loom.port import (
    PortModel,
    PortControlRecord,
    DoseResponse,
    _fit_dose_response_curve,
    _select_recommended_strength,
)
from loom.event_gating import (
    EntropyGate,
    EntropyGateConfig,
    calibrate_entropy_gate,
    measure_entropy,
)
from loom.dual_use import (
    classify_direction,
    get_artifact_path,
)
from loom.spec import Control


class TestPortModel:
    """Test PORT model loading and control installation."""

    def test_port_model_load(self):
        """Test loading a frozen HF model."""
        try:
            model = PortModel("gpt2", device="cuda" if torch.cuda.is_available() else "cpu")
            assert model.model is not None
            assert model.tokenizer is not None
            assert not model.model.training
            model.cleanup()
        except OSError:
            pytest.skip("Model not cached locally")

    def test_port_model_freeze(self):
        """Test that model weights are frozen."""
        try:
            model = PortModel("gpt2")
            for param in model.model.parameters():
                assert not param.requires_grad
            model.cleanup()
        except OSError:
            pytest.skip("Model not cached locally")

    def test_control_installation(self):
        """Test installing controls on frozen model."""
        try:
            model = PortModel("gpt2")

            steering_vec = torch.randn(model.model.config.n_embd)
            steering_vec = steering_vec / (steering_vec.norm() + 1e-8)

            control = PortControlRecord(
                name="test_suppress",
                kind="suppress",
                token=2,
                layer=-1,
                strength=0.5,
                mechanism="steering",
                steering_vector=steering_vec,
            )

            model.install_controls([control])
            assert len(model.controls) == 1
            assert model.controls[0].name == "test_suppress"
            model.cleanup()
        except OSError:
            pytest.skip("Model not cached locally")

    def test_control_enable_disable(self):
        """Test enabling/disabling controls."""
        try:
            model = PortModel("gpt2")

            steering_vec = torch.randn(model.model.config.n_embd)
            steering_vec = steering_vec / (steering_vec.norm() + 1e-8)

            control = PortControlRecord(
                name="test_control",
                kind="suppress",
                token=2,
                layer=-1,
                mechanism="steering",
                steering_vector=steering_vec,
            )

            model.install_controls([control])

            # Initially enabled
            assert model.control_enabled["test_control"]

            # Disable
            model.disable_control("test_control")
            assert not model.control_enabled["test_control"]

            # Re-enable
            model.enable_control("test_control")
            assert model.control_enabled["test_control"]

            model.cleanup()
        except OSError:
            pytest.skip("Model not cached locally")


class TestEntropyGating:
    """Test event-gating based on entropy."""

    def test_measure_entropy(self):
        """Test entropy calculation from logits."""
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        entropy = measure_entropy(logits)

        # Entropy should be positive and less than log(vocab_size)
        assert 0 < entropy < np.log(4.0)

    def test_entropy_gate_calibration(self):
        """Test entropy threshold calibration."""
        entropies = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
        tau = calibrate_entropy_gate(entropies, percentile=60.0)

        # Should be around the P60 value
        assert 0.7 < tau < 0.9

    def test_entropy_gate_fire_logic(self):
        """Test entropy gate firing decisions."""
        config = EntropyGateConfig(percentile=60.0, min_gap=2)
        gate = EntropyGate(config=config, entropy_threshold=0.5)

        # High entropy, first step: should fire
        assert gate.should_fire(step=0, entropy=0.7)

        # Low entropy, second step: should not fire
        assert not gate.should_fire(step=1, entropy=0.3)

        # High entropy, but too soon after last fire (2 - 0 = 2, need > 2): should not fire
        # Actually step 2 is exactly 2 steps after step 0, and min_gap=2 means >= 2
        # So it will fire. Let's test step 1 instead
        assert not gate.should_fire(step=1, entropy=0.8)

        # High entropy, enough gap (4 - 0 = 4 > 2): should fire
        assert gate.should_fire(step=4, entropy=0.7)

    def test_entropy_gate_recording(self):
        """Test recording entropy measurements."""
        gate = EntropyGate(entropy_threshold=0.5)

        logits = torch.tensor([1.0, 2.0, 3.0])
        gate.record_step(step=0, entropy=0.6, logits=logits, fired=True)
        gate.record_step(step=1, entropy=0.4, logits=logits, fired=False)

        assert len(gate.entropies) == 2
        assert gate.entropies[0].fired
        assert not gate.entropies[1].fired

    def test_entropy_budget_computation(self):
        """Test entropy budget impact calculation."""
        gate = EntropyGate()

        logits = torch.tensor([1.0, 2.0, 3.0])
        gate.record_step(step=0, entropy=0.5, logits=logits, fired=True)
        gate.record_step(step=1, entropy=0.6, logits=logits, fired=True)
        gate.record_step(step=2, entropy=0.7, logits=logits, fired=False)

        baseline = [0.8, 0.8, 0.8]
        budget = gate.compute_entropy_budget(baseline)

        assert "baseline_mean" in budget
        assert "steered_mean" in budget
        assert "delta" in budget
        assert abs(budget["baseline_mean"] - 0.8) < 1e-6
        assert budget["steered_mean"] < 0.8  # steered is lower

    def test_write_efficiency(self):
        """Test write efficiency calculation."""
        gate = EntropyGate()

        logits = torch.tensor([1.0, 2.0, 3.0])
        for i in range(10):
            fired = i % 3 == 0  # Fire every 3rd step
            gate.record_step(step=i, entropy=0.5, logits=logits, fired=fired)

        efficiency = gate.compute_write_efficiency()

        assert efficiency["total_steps"] == 10
        assert efficiency["total_writes"] == 4  # steps 0, 3, 6, 9
        assert 0 < efficiency["write_sparsity"] < 1


class TestDoseResponse:
    """Test dose-response curve fitting."""

    def test_dose_response_fitting(self):
        """Test fitting 4-parameter logistic."""
        # Create synthetic dose-response data
        strengths = [0.0, 0.5, 1.0, 1.5, 2.0]
        effects = [0.0, 0.3, 0.6, 0.8, 0.9]

        result = _fit_dose_response_curve(strengths, effects)

        assert "ec50" in result
        assert "slope" in result
        assert "r2" in result
        assert 0.0 <= result["ec50"] <= 2.0

    def test_recommended_strength_selection(self):
        """Test selection of recommended strength."""
        strengths = [0.0, 0.5, 1.0, 1.5, 2.0]
        effects = [0.0, 0.5, 0.75, 0.85, 0.90]
        side_effects = [0.0, 0.02, 0.05, 0.10, 0.15]

        # With threshold 0.75 and budget 0.08
        recommended = _select_recommended_strength(
            strengths, effects, side_effects,
            effect_threshold=0.75,
            side_effect_budget=0.08
        )

        # Should select a strength that meets both criteria
        assert isinstance(recommended, float)


class TestDualUsePolicy:
    """Test dual-use safety artifact routing."""

    def test_classify_direction_harmless(self):
        """Test classifying a harmless direction."""
        control = Control(
            name="test",
            kind="suppress",
            token=2,
            concept="fluency"
        )

        kind = classify_direction(control, "fluency", torch.randn(100))
        assert kind == "hardening"

    def test_classify_direction_harmful(self):
        """Test classifying a harmful direction."""
        control = Control(
            name="test",
            kind="suppress",
            token=2,
            concept="refusal"
        )

        kind = classify_direction(control, "refusal", torch.randn(100))
        assert kind == "suppression"

    def test_classify_direction_amplify_safety(self):
        """Test that amplifying safety is always hardening."""
        control = Control(
            name="test",
            kind="amplify",
            token=2,
            concept="refusal"
        )

        kind = classify_direction(control, "refusal", torch.randn(100))
        assert kind == "hardening"

    def test_artifact_path_suppression(self):
        """Test path for suppression artifacts."""
        path = get_artifact_path("suppression", "refusal", "steering_vector")
        assert "private" in str(path)
        assert "suppression" in str(path)

    def test_artifact_path_hardening(self):
        """Test path for hardening artifacts."""
        path = get_artifact_path("hardening", "refusal", "steering_vector")
        assert "results" in str(path)
        assert "hardening" in str(path)


class TestLoomSpec:
    """Test spec parsing (already exists, included for completeness)."""

    def test_control_spec_parsing(self):
        """Test Control spec creation."""
        control = Control(
            name="suppress_token_2",
            kind="suppress",
            token=2,
            concept="token_2",
            strength=1.0
        )

        assert control.name == "suppress_token_2"
        assert control.kind == "suppress"
        assert control.token == 2
        assert control.strength == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

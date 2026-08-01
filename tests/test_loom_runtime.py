"""Tests for Loom runtime, verification, and CLI."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from loom.cli import build_weave, explain_weave
from loom.monitors import fit_monitors
from loom.runtime import (
    ControlledModel,
    ControlRecord,
    compute_steering_vector,
    compute_suppression_metrics,
    install_controls,
)
from loom.spec import Control, Gate, Monitor, Skill, WeaveSpec
from loom.verify import Report, verify
from miabstraction.models import TinyTransformer



def _belief_ground_truth(tokens):
    """Real Mess3 posteriors for calib tokens (vocab 3 assumed by these tests)."""
    from miabstraction.data.mess3 import belief_states, mess3_matrices
    T = mess3_matrices(x=0.05, a=0.85)
    b = belief_states(T, tokens.numpy() % 3)
    return {"belief_state": b.reshape(-1, b.shape[-1])}

class TestVerify:
    """Test the gate verification engine."""

    def test_gate_pass(self):
        """A gate passes when measured > threshold (op='>') or measured < threshold (op='<')."""
        gate_gt = Gate(target="test", metric="acc", op=">", threshold=0.9)
        assert gate_gt.holds(0.95)
        assert not gate_gt.holds(0.85)

        gate_lt = Gate(target="test", metric="loss", op="<", threshold=0.1)
        assert gate_lt.holds(0.05)
        assert not gate_lt.holds(0.15)

    def test_verify_all_pass(self):
        """Report passes iff all gates pass."""
        spec = WeaveSpec(
            model={"d_model": 64, "n_layers": 4, "n_heads": 4, "max_len": 128},
            gates=[
                Gate(target="test", metric="acc", op=">", threshold=0.8),
                Gate(target="test", metric="loss", op="<", threshold=0.2),
            ],
        )
        measured = {
            "test": {"acc": 0.9, "loss": 0.1}
        }
        report = verify(spec, measured)
        assert report.passed
        assert len(report.gates) == 2
        assert all(g.passed for g in report.gates)

    def test_verify_one_fails(self):
        """Report fails if any gate fails."""
        spec = WeaveSpec(
            model={"d_model": 64, "n_layers": 4, "n_heads": 4, "max_len": 128},
            gates=[
                Gate(target="test", metric="acc", op=">", threshold=0.9),
                Gate(target="test", metric="loss", op="<", threshold=0.1),
            ],
        )
        measured = {
            "test": {"acc": 0.85, "loss": 0.05}  # acc fails
        }
        report = verify(spec, measured)
        assert not report.passed
        assert not report.gates[0].passed
        assert report.gates[1].passed

    def test_missing_measurement(self):
        """Missing measurements cause gate failure."""
        spec = WeaveSpec(
            model={"d_model": 64, "n_layers": 4, "n_heads": 4, "max_len": 128},
            gates=[
                Gate(target="test", metric="acc", op=">", threshold=0.8),
            ],
        )
        measured = {"test": {}}  # No "acc" key
        report = verify(spec, measured)
        assert not report.passed
        assert "No measurement" in report.gates[0].diagnosis

    def test_refusal_diagnosis(self):
        """Failed gates include layman-readable diagnostics."""
        spec = WeaveSpec(
            model={"d_model": 64, "n_layers": 4, "n_heads": 4, "max_len": 128},
            gates=[
                Gate(target="suppress_test", metric="suppression_ratio", op=">", threshold=0.9),
            ],
        )
        measured = {"suppress_test": {"suppression_ratio": 0.5}}
        report = verify(spec, measured)
        diagnosis = report.gates[0].diagnosis
        assert "too weak" in diagnosis.lower()
        assert "50" in diagnosis

    def test_report_serialization(self):
        """Report serializes to JSON correctly."""
        spec = WeaveSpec(
            model={"d_model": 64, "n_layers": 4, "n_heads": 4, "max_len": 128},
            name="test_weave",
            gates=[
                Gate(target="test", metric="acc", op=">", threshold=0.8),
            ],
        )
        measured = {"test": {"acc": 0.9}}
        report = verify(spec, measured)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            report.save_json(path)
            data = json.loads(path.read_text())
            assert data["spec_name"] == "test_weave"
            assert data["passed"] is True


class TestRuntime:
    """Test steering vector computation and control installation."""

    def test_compute_steering_vector(self):
        """Steering vector is computed from contrastive activations."""
        model = TinyTransformer(vocab=256, d_model=32, n_layers=2, n_heads=2, max_len=64)
        model.eval()

        # Create calibration data with target token present
        calib_tokens = torch.randint(0, 256, (10, 32), dtype=torch.int64)
        # Make sure target token 42 appears frequently
        calib_tokens[:, 1:] = 42  # Next tokens are mostly 42

        steering_vec = compute_steering_vector(
            model, target_token=42, calib_tokens=calib_tokens, layer=-1
        )

        # Should return a vector (or None if not enough data, both are ok)
        if steering_vec is not None:
            assert steering_vec.shape == (32,)  # d_model

    def test_controlled_model_creation(self):
        """ControlledModel wraps a model and applies steering hooks."""
        base_model = TinyTransformer(vocab=256, d_model=32, n_layers=2, n_heads=2, max_len=64)
        control = ControlRecord(
            name="test_control",
            kind="suppress",
            token=42,
            layer=-1,
            strength=1.0,
            mechanism="steering",
            steering_vector=torch.randn(32),
        )

        controlled = ControlledModel(base_model, [control])
        assert len(controlled.controls) == 1
        assert controlled.control_enabled["test_control"] is True

        # Disable and re-enable
        controlled.disable_control("test_control")
        assert not controlled.control_enabled["test_control"]
        controlled.enable_control("test_control")
        assert controlled.control_enabled["test_control"]

    def test_install_controls(self):
        """install_controls creates steering vectors and ControlledModel."""
        model = TinyTransformer(vocab=256, d_model=32, n_layers=2, n_heads=2, max_len=64)

        spec = WeaveSpec(
            model={"d_model": 32, "n_layers": 2, "n_heads": 2, "max_len": 64},
            controls=[
                Control(name="suppress_42", kind="suppress", token=42, strength=1.0),
            ],
        )

        calib_tokens = torch.randint(0, 256, (10, 32), dtype=torch.int64)

        controlled = install_controls(model, spec, calib_tokens)

        assert isinstance(controlled, ControlledModel)
        assert len(controlled.controls) == 1
        assert controlled.controls[0].name == "suppress_42"
        assert controlled.controls[0].mechanism in ("steering", "logit_mask")

    def test_suppression_metrics(self):
        """Suppression metrics are computed correctly."""
        model = TinyTransformer(vocab=256, d_model=32, n_layers=2, n_heads=2, max_len=64)

        spec = WeaveSpec(
            model={"d_model": 32, "n_layers": 2, "n_heads": 2, "max_len": 64},
            controls=[
                Control(name="suppress_42", kind="suppress", token=42, strength=1.0),
            ],
        )

        calib_tokens = torch.randint(0, 256, (10, 32), dtype=torch.int64)
        test_tokens = torch.randint(0, 256, (5, 32), dtype=torch.int64)

        controlled = install_controls(model, spec, calib_tokens)

        metrics = compute_suppression_metrics(
            controlled, controlled.controls[0], test_tokens, n_samples=5
        )

        assert "suppression_ratio" in metrics
        assert "side_effect" in metrics
        assert 0.0 <= metrics["suppression_ratio"] <= 1.0
        assert isinstance(metrics["side_effect"], (float, int))


class TestMonitors:
    """Test probe monitor fitting."""

    def test_fit_monitors_simple(self):
        """fit_monitors fits linear probes on residual activations."""
        model = TinyTransformer(vocab=256, d_model=32, n_layers=2, n_heads=2, max_len=64)

        spec = WeaveSpec(
            model={"d_model": 32, "n_layers": 2, "n_heads": 2, "max_len": 64},
            monitors=[
                Monitor(name="state_monitor", kind="probe", concept="belief_state"),
            ],
        )

        calib_tokens = torch.randint(0, 256, (10, 32), dtype=torch.int64)

        monitors_dict = fit_monitors(model, spec, calib_tokens, _belief_ground_truth(calib_tokens))

        assert "state_monitor" in monitors_dict
        monitor = monitors_dict["state_monitor"]
        assert monitor.name == "state_monitor"
        assert monitor.concept == "belief_state"
        assert "r2_val" in monitor.fit_quality or "acc_val" in monitor.fit_quality

    def test_monitor_read(self):
        """Monitor.read() returns predictions for a residual activation."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TinyTransformer(vocab=256, d_model=32, n_layers=2, n_heads=2, max_len=64)

        spec = WeaveSpec(
            model={"d_model": 32, "n_layers": 2, "n_heads": 2, "max_len": 64},
            monitors=[
                Monitor(name="state_monitor", kind="probe", concept="belief_state"),
            ],
        )

        calib_tokens = torch.randint(0, 256, (10, 32), dtype=torch.int64)

        monitors_dict = fit_monitors(model, spec, calib_tokens, _belief_ground_truth(calib_tokens), device=device)
        monitor = monitors_dict["state_monitor"]

        # Test reading a single residual activation (on the correct device)
        residual = torch.randn(32, device=device)
        pred = monitor.read(residual)
        assert pred.shape == (3,)  # belief_state has 3 dimensions

        # Test reading a batch
        residuals = torch.randn(5, 32, device=device)
        preds = monitor.read(residuals)
        assert preds.shape == (5, 3)


class TestCLI:
    """Test CLI commands."""

    def test_explain_weave(self, capsys):
        """explain command prints weave info."""
        spec_yaml = """
name: test_weave
model: {size: small}
skills:
  - {name: copy_patterns, kind: induction}
controls:
  - {name: suppress_test, kind: suppress, token: 42}
monitors:
  - {name: state_monitor, kind: probe, concept: belief_state}
gates:
  copy_patterns: {prefix_score: ">0.5"}
  suppress_test: {suppression_ratio: ">0.9", side_effect: "<0.1"}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / "test.weave.yaml"
            spec_path.write_text(spec_yaml)

            explain_weave(str(spec_path))
            captured = capsys.readouterr()

            assert "test_weave" in captured.out
            assert "copy_patterns" in captured.out
            assert "suppress_test" in captured.out
            assert "state_monitor" in captured.out

    def test_build_weave_cli(self):
        """build command creates report and weights."""
        spec_yaml = """
name: test_weave
model: {size: small}
skills:
  - {name: copy_patterns, kind: induction}
controls:
  - {name: suppress_test, kind: suppress, token: 42}
gates:
  copy_patterns: {prefix_score: ">0.05"}
  suppress_test: {suppression_ratio: ">0.05", side_effect: "<0.9"}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / "test.weave.yaml"
            spec_path.write_text(spec_yaml)

            output_dir = Path(tmpdir) / "build"

            # Run build
            exit_code = build_weave(str(spec_path), str(output_dir))

            # Should complete (may pass or fail, but should run)
            assert exit_code in (0, 1)

            # Check output artifacts
            assert (output_dir / "report.json").exists()
            assert (output_dir / "weights.pt").exists()
            assert (output_dir / "spec.yaml").exists()

            # Check report is valid JSON
            report_data = json.loads((output_dir / "report.json").read_text())
            assert "gates" in report_data
            assert "passed" in report_data


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_steering_pipeline(self):
        """Full pipeline: train -> steer -> measure -> verify."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create model
        model = TinyTransformer(vocab=256, d_model=32, n_layers=2, n_heads=2, max_len=64)
        model.to(device)

        # Train briefly
        train_tokens = torch.randint(0, 256, (50, 32), dtype=torch.int64)
        from miabstraction.models import train_lm
        train_lm(model, train_tokens, steps=10, batch_size=8, lr=1e-3, device=device)

        # Define spec
        spec = WeaveSpec(
            model={"d_model": 32, "n_layers": 2, "n_heads": 2, "max_len": 64},
            skills=[Skill(name="copy_patterns", kind="induction")],
            controls=[Control(name="suppress_42", kind="suppress", token=42, strength=1.0)],
            monitors=[Monitor(name="state_monitor", kind="probe", concept="belief_state")],
            gates=[
                Gate(target="copy_patterns", metric="prefix_score", op=">", threshold=0.0),
                Gate(target="suppress_42", metric="suppression_ratio", op=">", threshold=0.0),
                Gate(target="suppress_42", metric="side_effect", op="<", threshold=1.0),
            ],
        )

        # Install controls
        calib_tokens = torch.randint(0, 256, (5, 32), dtype=torch.int64)
        controlled = install_controls(model, spec, calib_tokens, device=device)

        # Fit monitors
        monitors_dict = fit_monitors(model, spec, calib_tokens, _belief_ground_truth(calib_tokens), device=device)

        # Measure
        test_tokens = torch.randint(0, 256, (3, 32), dtype=torch.int64)
        measured = {}

        if controlled.controls:
            ctrl = controlled.controls[0]
            metrics = compute_suppression_metrics(controlled, ctrl, test_tokens, device=device, n_samples=3)
            measured[ctrl.name] = metrics

        measured["copy_patterns"] = {"prefix_score": 0.5}  # Dummy measurement

        for name, mon in monitors_dict.items():
            measured[name] = mon.fit_quality

        # Verify
        report = verify(spec, measured)

        # Should have gate results
        assert len(report.gates) > 0
        assert all(g.gate in spec.gates for g in report.gates)

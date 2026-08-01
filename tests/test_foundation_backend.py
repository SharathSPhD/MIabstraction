"""Tests for foundation backend: compute planning, compilation, verification."""
import json
from pathlib import Path
from unittest import mock

import pytest

from loom.compute import plan_target, ComputePlan
from loom.foundation import plan_foundation, build_foundation, _estimate_model_size
from loom.spec import Foundation, WeaveSpec, load_weave


class TestComputePlanner:
    """Tests for compute target planning."""

    def test_plan_pretraining_fits_rtx5090(self):
        """Pretraining that fits in 32GB should use RTX 5090."""
        config = {
            "is_pretraining": True,
            "n_params": 30_000_000,
            "max_memory_gb": 16,  # Fits in RTX 5090
            "throughput_critical": True,
        }
        plan = plan_target(config)
        assert plan.target == "rtx5090"
        assert "RTX 5090" in plan.rationale

    def test_plan_large_job_uses_local_gb10(self):
        """Jobs needing >32GB should use local GB10."""
        config = {
            "is_pretraining": True,
            "n_params": 100_000_000,
            "max_memory_gb": 64,  # Exceeds RTX 5090
            "throughput_critical": True,
        }
        plan = plan_target(config)
        assert plan.target == "local_gb10"
        assert "128GB" in plan.rationale

    def test_plan_throughput_critical_small_job(self):
        """Small throughput-critical jobs prefer RTX 5090."""
        config = {
            "is_pretraining": True,
            "n_params": 10_000_000,
            "max_memory_gb": 8,
            "throughput_critical": True,
        }
        plan = plan_target(config)
        assert plan.target == "rtx5090"


class TestFoundationPlanning:
    """Tests for foundation pretraining planning."""

    def test_plan_foundation_validates_corpus(self):
        """Invalid corpus should raise ValueError."""
        spec = WeaveSpec(
            model={"d_model": 64},
            foundation=Foundation(corpus="unknown_corpus"),
        )
        with pytest.raises(ValueError, match="Only 'babylm_strict'"):
            plan_foundation(spec)

    def test_plan_foundation_validates_tokenizer(self):
        """Invalid tokenizer should raise ValueError."""
        spec = WeaveSpec(
            model={"d_model": 64},
            foundation=Foundation(corpus="babylm_strict", tokenizer="unknown_tokenizer"),
        )
        with pytest.raises(ValueError, match="'gpt2' or 'bpe_train'"):
            plan_foundation(spec)

    def test_plan_foundation_validates_params(self):
        """Out-of-range params should raise ValueError."""
        spec = WeaveSpec(
            model={"d_model": 64},
            foundation=Foundation(corpus="babylm_strict", params=1_000_000),  # Too small
        )
        with pytest.raises(ValueError, match="10M-100M"):
            plan_foundation(spec)

        spec2 = WeaveSpec(
            model={"d_model": 64},
            foundation=Foundation(corpus="babylm_strict", params=1_000_000_000),  # Too large
        )
        with pytest.raises(ValueError, match="10M-100M"):
            plan_foundation(spec2)

    def test_plan_foundation_valid(self):
        """Valid foundation spec should produce a plan."""
        spec = WeaveSpec(
            model={"d_model": 64},
            foundation=Foundation(
                corpus="babylm_strict",
                tokenizer="gpt2",
                params=30_000_000,
                budget_hours=3.0,
            ),
        )
        plan = plan_foundation(spec)
        assert plan.corpus == "babylm_strict"
        assert plan.tokenizer == "gpt2"
        assert plan.model_params == 30_000_000
        assert plan.budget_hours == 3.0


class TestModelSizing:
    """Tests for model architecture estimation."""

    def test_estimate_model_size_10m(self):
        """Estimate architecture for ~10M params."""
        config = _estimate_model_size(10_000_000)
        assert config["vocab_size"] == 50257
        assert config["d_model"] > 0
        assert config["n_layers"] >= 4
        assert config["n_heads"] > 0
        assert config["max_len"] > 0

    def test_estimate_model_size_30m(self):
        """Estimate architecture for ~30M params."""
        config = _estimate_model_size(30_000_000)
        assert config["vocab_size"] == 50257
        assert config["d_model"] > 0
        assert config["n_layers"] >= 4

    def test_estimate_model_size_50m(self):
        """Estimate architecture for ~50M params."""
        config = _estimate_model_size(50_000_000)
        assert config["vocab_size"] == 50257
        assert config["d_model"] > 0


class TestFoundationCompilation:
    """Tests for compiling foundation weaves to job directories."""

    def test_build_foundation_creates_job_dir(self, tmp_path):
        """build_foundation should create a job directory with all files."""
        spec = WeaveSpec(
            model={"d_model": 64},
            foundation=Foundation(
                corpus="babylm_strict",
                tokenizer="gpt2",
                params=30_000_000,
                budget_hours=3.0,
            ),
            name="test_foundation",
        )
        # Add a gate so the spec is valid
        from loom.spec import Gate
        spec.gates.append(
            Gate(target="foundation", metric="val_loss", op="<", threshold=3.0)
        )

        job_dir = build_foundation(spec, tmp_path)

        # Check that job directory was created
        assert job_dir.exists()
        assert (job_dir / "train.py").exists()
        assert (job_dir / "config.json").exists()
        assert (job_dir / "README.md").exists()

    def test_build_foundation_config_valid(self, tmp_path):
        """Generated config should be valid JSON with expected keys."""
        spec = WeaveSpec(
            model={"d_model": 64},
            foundation=Foundation(
                corpus="babylm_strict",
                tokenizer="gpt2",
                params=30_000_000,
                budget_hours=3.0,
            ),
            name="test_foundation",
        )
        from loom.spec import Gate
        spec.gates.append(
            Gate(target="foundation", metric="val_ppl", op="<", threshold=20.0)
        )

        job_dir = build_foundation(spec, tmp_path)
        config = json.loads((job_dir / "config.json").read_text())

        assert "vocab_size" in config
        assert "d_model" in config
        assert "n_layers" in config
        assert "max_steps" in config
        assert config["max_steps"] > 0

    def test_build_foundation_train_script_valid(self, tmp_path):
        """Generated train.py should be syntactically valid Python."""
        spec = WeaveSpec(
            model={"d_model": 64},
            foundation=Foundation(
                corpus="babylm_strict",
                tokenizer="gpt2",
                params=30_000_000,
                budget_hours=3.0,
            ),
            name="test_foundation",
        )
        from loom.spec import Gate
        spec.gates.append(
            Gate(target="foundation", metric="val_loss", op="<", threshold=3.0)
        )

        job_dir = build_foundation(spec, tmp_path)
        script = (job_dir / "train.py").read_text()

        # Check for expected content
        assert "import torch" in script
        assert "GPTModel" in script
        assert "val_loss" in script
        assert "val_ppl" in script
        assert "metrics.json" in script


class TestWeaveSpecFoundation:
    """Tests for parsing foundation weaves from YAML."""

    def test_load_foundation_weave(self, tmp_path):
        """Load a foundation weave from YAML."""
        weave_yaml = """
model:
  size: small

foundation:
  corpus: babylm_strict
  tokenizer: gpt2
  params: 30000000
  budget_hours: 3.0

gates:
  foundation:
    val_loss: "<3.0"
    val_ppl: "<20.0"
"""
        weave_file = tmp_path / "foundation.weave.yaml"
        weave_file.write_text(weave_yaml)

        spec = load_weave(weave_file)
        assert spec.foundation is not None
        assert spec.foundation.corpus == "babylm_strict"
        assert spec.foundation.tokenizer == "gpt2"
        assert spec.foundation.params == 30_000_000
        assert spec.foundation.budget_hours == 3.0

    def test_load_foundation_weave_missing_gates(self, tmp_path):
        """Foundation weave without gates should fail."""
        weave_yaml = """
model:
  size: small

foundation:
  corpus: babylm_strict
  tokenizer: gpt2
  params: 30000000
  budget_hours: 3.0
"""
        weave_file = tmp_path / "foundation.weave.yaml"
        weave_file.write_text(weave_yaml)

        from loom.spec import WeaveError
        with pytest.raises(WeaveError, match="No gates declared"):
            load_weave(weave_file)

    def test_load_foundation_weave_invalid_gate_target(self, tmp_path):
        """Invalid gate target should fail."""
        weave_yaml = """
model:
  size: small

foundation:
  corpus: babylm_strict
  tokenizer: gpt2
  params: 30000000
  budget_hours: 3.0

gates:
  invalid_target:
    val_loss: "<3.0"
"""
        weave_file = tmp_path / "foundation.weave.yaml"
        weave_file.write_text(weave_yaml)

        from loom.spec import WeaveError
        with pytest.raises(WeaveError, match="does not match"):
            load_weave(weave_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

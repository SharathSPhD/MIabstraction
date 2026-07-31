"""Tests for E4 probe baseline experiment."""
import json
from pathlib import Path

from miabstraction.experiments.e4_probe_baseline import run
from miabstraction.config import ExperimentConfig


def test_e4_config_loads():
    """Test that E4 config loads successfully."""
    cfg = ExperimentConfig.load("configs/e4_probe_baseline.yaml")
    assert cfg.name == "e4_probe_baseline"
    assert cfg.hypothesis == "H4"


def test_e4_run_returns_dict():
    """Test that E4 run returns proper result dict."""
    cfg = ExperimentConfig.load("configs/e4_probe_baseline.yaml")
    # Use minimal config for fast test
    cfg.data["n_seq"] = 100
    cfg.data["seq_len"] = 16
    cfg.train["steps"] = 100
    cfg.analysis["n_eval_seq"] = 50

    result = run(cfg)

    # Check result structure
    assert isinstance(result, dict)
    assert "hypothesis" in result
    assert "supports" in result
    assert result["hypothesis"] == "H4"

    # Check probe accuracy keys
    probe_keys = ["raw_probe_acc_belief", "sae_probe_acc_belief"]
    for key in probe_keys:
        assert key in result, f"Missing key: {key}"

    assert "raw_probe_acc_entropy" in result
    assert "sae_probe_acc_entropy" in result


def test_e4_result_file_written(tmp_path):
    """Test that E4 writes result.json."""
    cfg = ExperimentConfig.load("configs/e4_probe_baseline.yaml")
    cfg.out_dir = str(tmp_path)
    cfg.data["n_seq"] = 50
    cfg.data["seq_len"] = 16
    cfg.train["steps"] = 50
    cfg.analysis["n_eval_seq"] = 25

    result = run(cfg)

    result_file = tmp_path / "e4_probe_baseline" / "result.json"
    assert result_file.exists()

    # Verify JSON is valid
    with open(result_file) as f:
        data = json.load(f)
    assert data["hypothesis"] == "H4"
    assert "supports" in data

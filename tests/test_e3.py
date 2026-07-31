"""Tests for E3 SAE random-transformer control experiment."""
import json
from pathlib import Path

import numpy as np
import torch

from miabstraction.experiments.e3_sae_control import run
from miabstraction.config import ExperimentConfig


def test_e3_config_loads():
    """Test that E3 config loads successfully."""
    cfg = ExperimentConfig.load("configs/e3_sae_control.yaml")
    assert cfg.name == "e3_sae_control"
    assert cfg.hypothesis == "H3"


def test_e3_run_returns_dict():
    """Test that E3 run returns proper result dict."""
    cfg = ExperimentConfig.load("configs/e3_sae_control.yaml")
    # Use minimal config for fast test
    cfg.data["n_seq"] = 100
    cfg.data["seq_len"] = 16
    cfg.train["steps"] = 100
    cfg.analysis["n_sae_seeds"] = 2
    cfg.analysis["sae_steps"] = 50

    result = run(cfg)

    # Check result structure
    assert isinstance(result, dict)
    assert "hypothesis" in result
    assert "supports" in result
    assert result["hypothesis"] == "H3"

    # Check metric keys
    metrics_keys = ["fvu_mean", "fvu_std", "l0_mean", "l0_std", "dead_frac_mean", "dead_frac_std"]
    for key in metrics_keys:
        assert key in result, f"Missing key: {key}"
        assert result[key] is not None

    # Check trained vs random comparison
    assert "fvu_mean_random" in result
    assert "fvu_mean_trained" in result
    assert "fvu_separation_sigma" in result


def test_e3_result_file_written(tmp_path):
    """Test that E3 writes result.json."""
    cfg = ExperimentConfig.load("configs/e3_sae_control.yaml")
    cfg.out_dir = str(tmp_path)
    cfg.data["n_seq"] = 50
    cfg.data["seq_len"] = 16
    cfg.train["steps"] = 50
    cfg.analysis["n_sae_seeds"] = 1
    cfg.analysis["sae_steps"] = 30

    result = run(cfg)

    result_file = tmp_path / "e3_sae_control" / "result.json"
    assert result_file.exists()

    # Verify JSON is valid
    with open(result_file) as f:
        data = json.load(f)
    assert data["hypothesis"] == "H3"
    assert "supports" in data

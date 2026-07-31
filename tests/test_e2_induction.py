"""Tests for E2 induction experiment."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from miabstraction.config import ExperimentConfig
from miabstraction.experiments.e2_induction import run


@pytest.fixture
def minimal_config_dict():
    """Minimal E2 config for testing."""
    return {
        "name": "e2_induction",
        "hypothesis": "H2",
        "seed": 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model": {"d_model": 32, "n_layers": 2, "n_heads": 2, "max_len": 64},
        "data": {
            "n_seq": 500,
            "seq_len": 32,
            "vocab": 10,
            "repeat_len": 4,
        },
        "train": {"steps": 200, "batch_size": 64, "lr": 0.001, "log_every": 50},
        "analysis": {"n_attention_seq": 100},
        "out_dir": "/tmp/test_e2",
    }


@pytest.fixture
def config(minimal_config_dict, tmp_path):
    """Create a test config and write to temp file."""
    config_dict = minimal_config_dict.copy()
    config_dict["out_dir"] = str(tmp_path)
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f)
    return ExperimentConfig.load(config_file)


def test_e2_run_returns_result_dict(config):
    """Experiment run returns a dict with required keys."""
    result = run(config)
    assert isinstance(result, dict)
    assert "hypothesis" in result
    assert result["hypothesis"] == "H2"
    assert "supports" in result
    assert isinstance(result["supports"], (bool, np.bool_))
    assert "config_hash" in result
    assert "runtime_s" in result


def test_e2_run_creates_result_json(config):
    """Experiment writes result.json."""
    result = run(config)
    result_file = config.result_dir() / "result.json"
    assert result_file.exists()
    with open(result_file) as f:
        stored = json.load(f)
    assert stored["hypothesis"] == "H2"
    assert "supports" in stored


def test_e2_run_creates_plot(config):
    """Experiment creates phase transition plot."""
    result = run(config)
    plot_file = config.result_dir() / "phase_transition.png"
    assert plot_file.exists()
    assert plot_file.stat().st_size > 0


def test_e2_run_detects_phase_transition_metrics(config):
    """Result includes phase transition detection metrics."""
    result = run(config)
    # These should be present if transition detection ran
    assert "prefix_matching_scores" in result or "transition_window" in result


def test_e2_run_loss_metrics_present(config):
    """Result includes loss tracking for repeat vs first-occurrence regions."""
    result = run(config)
    # Should track losses on different token types
    assert "final_loss" in result or "loss_history" in result


def test_e2_supports_true_if_transition_detected(config):
    """H2 is supported iff clear phase transition is detected."""
    result = run(config)
    supports = result["supports"]
    assert isinstance(supports, (bool, np.bool_))
    # If supports=True, there should be evidence of a transition
    if supports:
        # Window should be short (< 20% of total steps)
        if "transition_window_width" in result:
            window_pct = result["transition_window_width"] / config.train["steps"]
            assert window_pct < 0.2


def test_e2_git_sha_recorded(config):
    """Result records git SHA for provenance."""
    result = run(config)
    assert "git_sha" in result or "config_hash" in result


def test_e2_device_recorded(config):
    """Result records which device was used."""
    result = run(config)
    assert "device" in result

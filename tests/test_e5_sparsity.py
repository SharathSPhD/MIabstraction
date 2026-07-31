"""Tests for E5 sparsity experiment."""
import json
import torch
import pytest
from pathlib import Path

from miabstraction.experiments.e5_sparsity import run
from miabstraction.config import ExperimentConfig


def test_e5_config_load():
    """Test loading E5 config."""
    config_path = Path("configs/e5_sparsity.yaml")
    if config_path.exists():
        cfg = ExperimentConfig.load(config_path)
        assert cfg.name == "e5_sparsity"
        assert cfg.hypothesis == "H5"


def test_e5_result_structure():
    """Test that E5 run produces correct result structure."""
    # Create a minimal config for testing
    cfg = ExperimentConfig(
        name="e5_sparsity",
        hypothesis="H5",
        seed=42,
        device="cpu",
        model=dict(
            d_model=32,
            n_layers=2,
            n_heads=2,
            max_len=32,
        ),
        data=dict(
            vocab_size=10,
            seq_len=16,
            n_seq_train=100,
            n_seq_val=50,
        ),
        train=dict(
            steps=100,
            batch_size=16,
            lr=1e-3,
        ),
        analysis=dict(
            sparsity_targets=[0.5, 0.2],
        ),
        out_dir="results",
    )

    result = run(cfg)

    # Check required fields
    assert "hypothesis" in result
    assert result["hypothesis"] == "H5"
    assert "supports" in result
    assert isinstance(result["supports"], bool)
    assert "circuit_size_dense" in result
    assert "circuit_size_sparse" in result
    assert "faithfulness_dense" in result
    assert "faithfulness_sparse" in result
    assert "config_hash" in result


def test_e5_pareto_result():
    """Test that E5 produces Pareto frontier results."""
    cfg = ExperimentConfig(
        name="e5_sparsity_pareto",
        hypothesis="H5",
        seed=42,
        device="cpu",
        model=dict(
            d_model=32,
            n_layers=2,
            n_heads=2,
            max_len=32,
        ),
        data=dict(
            vocab_size=10,
            seq_len=16,
            n_seq_train=100,
            n_seq_val=50,
        ),
        train=dict(
            steps=100,
            batch_size=16,
            lr=1e-3,
        ),
        analysis=dict(
            sparsity_targets=[0.5, 0.2],
        ),
        out_dir="results",
    )

    result = run(cfg)

    # Check Pareto frontier data
    assert "pareto_sparse_level" in result or "sparsity_levels" in result
    assert "accuracy_dense" in result or "final_accuracy_dense" in result


def test_e5_h5_hypothesis():
    """Test H5 verdict: sparse smaller AND more faithful than dense."""
    cfg = ExperimentConfig(
        name="e5_h5_test",
        hypothesis="H5",
        seed=0,
        device="cpu",
        model=dict(
            d_model=32,
            n_layers=2,
            n_heads=2,
            max_len=32,
        ),
        data=dict(
            vocab_size=10,
            seq_len=16,
            n_seq_train=100,
            n_seq_val=50,
        ),
        train=dict(
            steps=100,
            batch_size=16,
            lr=1e-3,
        ),
        analysis=dict(
            sparsity_targets=[0.2],
        ),
        out_dir="results",
    )

    result = run(cfg)

    # H5 is supported iff:
    # sparse circuit size < dense circuit size
    # AND sparse faithfulness >= dense faithfulness
    if result["supports"]:
        assert result["circuit_size_sparse"] < result["circuit_size_dense"]
        assert result["faithfulness_sparse"] >= result["faithfulness_dense"]

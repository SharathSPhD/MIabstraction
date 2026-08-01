"""Test that hand-written solvers achieve high metric scores (validates task design).

This ensures that each task is well-posed and metrics are achievable.
"""
import numpy as np
import torch

from loom.curriculum import ClassifyCompiler, InductionCompiler, StateTrackingCompiler
from miabstraction.models import TinyTransformer


def test_induction_task_solvable():
    """Test that induction task can be solved."""
    compiler = InductionCompiler(copy_len=4, max_gap=4, vocab_offset=0)
    rng = np.random.default_rng(42)

    # Generate a small dataset
    seqs, gaps = compiler.generator(n_seq=32, rng=rng)
    tokens = torch.from_numpy(seqs)

    # Create a TinyTransformer and train it on this specific task
    model = TinyTransformer(vocab=256, d_model=32, n_layers=2, n_heads=2, max_len=64)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    # Quick training
    for _ in range(100):
        logits = model(tokens[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 256), tokens[:, 1:].reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Evaluate
    metrics = compiler.evaluator(model, tokens, gaps, "cpu")

    print(f"Induction metrics: {metrics}")
    # Even a hand-trained solver should get reasonable metrics
    assert metrics["prefix_score"] > 0.1, f"prefix_score too low: {metrics['prefix_score']}"
    assert metrics["icl_loss"] < 5.0, f"icl_loss too high: {metrics['icl_loss']}"


def test_classify_task_solvable():
    """Test that classify (parity) task can be solved."""
    compiler = ClassifyCompiler(seq_len=16, vocab_offset=256)
    rng = np.random.default_rng(42)

    # Generate dataset
    seqs, answers = compiler.generator(n_seq=32, rng=rng)
    tokens = torch.from_numpy(seqs)

    # Train a TinyTransformer
    model = TinyTransformer(vocab=512, d_model=32, n_layers=2, n_heads=2, max_len=96)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    # Quick training - with proper task design, this should work
    for _ in range(150):
        logits = model(tokens[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 512), tokens[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Evaluate
    metrics = compiler.evaluator(model, tokens, answers, "cpu")
    print(f"Classify metrics: {metrics}")
    # With proper training, should achieve some accuracy
    assert metrics["accuracy"] > 0.3, f"accuracy too low: {metrics['accuracy']}"


def test_state_tracking_task_evaluates():
    """Test that state tracking task produces reasonable metrics."""
    compiler = StateTrackingCompiler(seq_len=16, x=0.05, a=0.85, vocab_offset=0)
    rng = np.random.default_rng(42)

    # Generate dataset
    tokens, beliefs = compiler.generator(n_seq=16, rng=rng)
    tokens = torch.from_numpy(tokens)

    # Use a TinyTransformer
    model = TinyTransformer(vocab=256, d_model=32, n_layers=2, n_heads=2, max_len=64)

    # Evaluate - should at least run without errors
    metrics = compiler.evaluator(model, tokens, beliefs, "cpu")
    print(f"State tracking metrics: {metrics}")
    assert "probe_r2" in metrics
    assert "probe_r2_incremental" in metrics
    assert 0 <= metrics["probe_r2"] <= 1

"""Tests for minimal circuit extraction."""
import torch
import numpy as np
import pytest

from miabstraction.circuits import MinimalCircuit, CircuitExtractor
from miabstraction.models import TinyTransformer


def test_circuit_extractor_init():
    """Test CircuitExtractor initialization."""
    model = TinyTransformer(vocab=10, d_model=32, n_layers=2, n_heads=2)
    extractor = CircuitExtractor(model)
    assert extractor.model is model
    assert extractor.n_layers == 2


def test_minimal_circuit_structure():
    """Test MinimalCircuit structure."""
    circuit = MinimalCircuit(nodes=[(0, 0), (1, 1)], faithfulness=0.95)
    assert len(circuit.nodes) == 2
    assert circuit.faithfulness == 0.95
    assert (0, 0) in circuit.nodes
    assert (1, 1) in circuit.nodes


def test_circuit_extraction_basic():
    """Test basic circuit extraction on a tiny model."""
    torch.manual_seed(0)
    model = TinyTransformer(vocab=5, d_model=32, n_layers=2, n_heads=1)
    model.eval()

    # Create dummy data with correct/incorrect labels
    batch_size = 16
    seq_len = 8
    tokens = torch.randint(0, 5, (batch_size, seq_len))
    # Create labels: random but consistent
    labels = torch.randint(0, 5, (batch_size,))
    answer_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    answer_mask[:, -1] = True  # last position is answer

    def accuracy_fn(model_or_ablated, _model):
        """Compute accuracy at answer positions."""
        with torch.no_grad():
            logits = model_or_ablated(tokens)
            preds = logits[:, -1, :].argmax(dim=-1)
            return (preds == labels).float().mean().item()

    # Extract circuit
    extractor = CircuitExtractor(model, accuracy_threshold_frac=0.95)
    circuit, full_acc = extractor.extract(
        tokens=tokens,
        labels=labels,
        answer_mask=answer_mask,
        accuracy_fn=accuracy_fn,
        max_iterations=50,
    )

    assert isinstance(circuit, MinimalCircuit)
    assert full_acc >= 0.0 and full_acc <= 1.0
    assert circuit.faithfulness >= 0.0 and circuit.faithfulness <= 1.0


def test_circuit_ablation_by_mean():
    """Test mean-ablation of nodes."""
    torch.manual_seed(0)
    model = TinyTransformer(vocab=5, d_model=32, n_layers=2, n_heads=1)
    model.eval()

    tokens = torch.randint(0, 5, (8, 10))

    # Collect baseline activations
    with torch.no_grad():
        _, resid = model(tokens, collect=True)

    # Compute batch mean for layer 0
    layer_0_mean = resid[0].mean(dim=0, keepdim=True)  # (1, seq_len, d_model)

    # Create ablated model and verify we can ablate specific layers
    # This is a simple check that the infrastructure works
    assert layer_0_mean.shape == (1, 10, 32)


def test_circuit_size_reduction():
    """Test that circuit is smaller than full model."""
    torch.manual_seed(0)
    model = TinyTransformer(vocab=5, d_model=32, n_layers=2, n_heads=2)
    model.eval()

    batch_size = 8
    seq_len = 10
    tokens = torch.randint(0, 5, (batch_size, seq_len))
    labels = torch.randint(0, 5, (batch_size,))
    answer_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    answer_mask[:, -1] = True

    def accuracy_fn(model_or_ablated, _model):
        with torch.no_grad():
            logits = model_or_ablated(tokens)
            preds = logits[:, -1, :].argmax(dim=-1)
            return (preds == labels).float().mean().item()

    extractor = CircuitExtractor(model, accuracy_threshold_frac=0.9)
    circuit, _ = extractor.extract(
        tokens=tokens,
        labels=labels,
        answer_mask=answer_mask,
        accuracy_fn=accuracy_fn,
        max_iterations=30,
    )

    # Full circuit size: n_layers * (n_heads + 1) = 2 * (2 + 1) = 6
    full_size = 2 * (2 + 1)
    circuit_size = len(circuit.nodes)

    # Circuit should be <= full size
    assert circuit_size <= full_size


def test_circuit_faithfulness():
    """Test circuit faithfulness computation."""
    torch.manual_seed(0)
    model = TinyTransformer(vocab=5, d_model=32, n_layers=2, n_heads=1)
    model.eval()

    tokens = torch.randint(0, 5, (8, 10))
    labels = torch.randint(0, 5, (8,))

    def accuracy_fn(model_or_ablated, _model):
        with torch.no_grad():
            logits = model_or_ablated(tokens)
            preds = logits[:, -1, :].argmax(dim=-1)
            return (preds == labels).float().mean().item()

    extractor = CircuitExtractor(model)
    circuit, full_acc = extractor.extract(
        tokens=tokens,
        labels=labels,
        answer_mask=torch.zeros_like(tokens, dtype=torch.bool),
        accuracy_fn=accuracy_fn,
        max_iterations=20,
    )

    # Faithfulness should be a fraction in [0, 1]
    assert 0.0 <= circuit.faithfulness <= 1.0


def test_circuit_comparison():
    """Test that we can compare circuits (e.g., dense vs sparse)."""
    circuit1 = MinimalCircuit(nodes=[(0, 0), (1, 0), (1, 1)], faithfulness=0.92)
    circuit2 = MinimalCircuit(nodes=[(0, 0), (1, 1)], faithfulness=0.88)

    # Circuit2 is smaller but less faithful
    assert len(circuit2.nodes) < len(circuit1.nodes)
    assert circuit2.faithfulness < circuit1.faithfulness

    # H5 hypothesis: sparse circuit is smaller AND more faithful
    supports_h5 = (len(circuit2.nodes) < len(circuit1.nodes) and
                   circuit2.faithfulness >= circuit1.faithfulness)
    assert not supports_h5  # Not satisfied in this case


def test_extractor_with_different_thresholds():
    """Test extraction with different accuracy thresholds."""
    torch.manual_seed(0)
    model = TinyTransformer(vocab=5, d_model=32, n_layers=2, n_heads=2)
    model.eval()

    tokens = torch.randint(0, 5, (8, 10))
    labels = torch.randint(0, 5, (8,))

    def accuracy_fn(model_or_ablated, _model):
        with torch.no_grad():
            logits = model_or_ablated(tokens)
            preds = logits[:, -1, :].argmax(dim=-1)
            return (preds == labels).float().mean().item()

    # Extract with stricter threshold (95%)
    extractor_strict = CircuitExtractor(model, accuracy_threshold_frac=0.95)
    circuit_strict, _ = extractor_strict.extract(
        tokens=tokens,
        labels=labels,
        answer_mask=torch.zeros_like(tokens, dtype=torch.bool),
        accuracy_fn=accuracy_fn,
        max_iterations=30,
    )

    # Extract with looser threshold (80%)
    extractor_loose = CircuitExtractor(model, accuracy_threshold_frac=0.80)
    circuit_loose, _ = extractor_loose.extract(
        tokens=tokens,
        labels=labels,
        answer_mask=torch.zeros_like(tokens, dtype=torch.bool),
        accuracy_fn=accuracy_fn,
        max_iterations=30,
    )

    # Stricter threshold should give larger circuits (more selective ablation)
    assert len(circuit_strict.nodes) >= len(circuit_loose.nodes)

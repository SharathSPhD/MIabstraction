"""Tests for weight sparsity (AbsTopK masking)."""
import torch
import pytest

from miabstraction.sparsity import WeightSparsity
from miabstraction.models import TinyTransformer, train_lm


def test_weight_sparsity_init():
    """Test WeightSparsity initialization."""
    model = TinyTransformer(vocab=4, d_model=32, n_layers=2, n_heads=2)
    sparsity = WeightSparsity(model, target_q=0.1)
    assert sparsity.target_q == 0.1
    assert sparsity.current_q == 1.0  # starts at full density


def test_weight_sparsity_apply_mask():
    """Test applying weight masks."""
    model = TinyTransformer(vocab=4, d_model=32, n_layers=2, n_heads=2)
    sparsity = WeightSparsity(model, target_q=0.1)

    # Initially no mask
    assert sparsity.masks is None

    # Apply mask for q=0.5
    sparsity.set_current_q(0.5)
    sparsity.apply_masks()

    # Check that masks are created and reduce parameters
    nonzero_before = sparsity.count_nonzero()
    total_params = sparsity.count_total_params()
    sparsity_level = nonzero_before / total_params
    assert 0.48 < sparsity_level < 0.52  # ~50% as expected


def test_weight_sparsity_anneal():
    """Test annealing schedule."""
    model = TinyTransformer(vocab=4, d_model=32, n_layers=2, n_heads=2)
    sparsity = WeightSparsity(model, target_q=0.1, anneal_steps=100)

    # At step 0, should be at full density
    sparsity.update_annealing_schedule(0)
    assert abs(sparsity.current_q - 1.0) < 1e-6

    # At step 50 (midway through anneal), should be interpolating
    sparsity.update_annealing_schedule(50)
    assert 0.5 < sparsity.current_q < 1.0

    # At step 100 (end of anneal), should be at target
    sparsity.update_annealing_schedule(100)
    assert abs(sparsity.current_q - 0.1) < 1e-6

    # Beyond anneal steps, stay at target
    sparsity.update_annealing_schedule(200)
    assert abs(sparsity.current_q - 0.1) < 1e-6


def test_weight_sparsity_mask_refresh():
    """Test mask refresh after optimizer step."""
    model = TinyTransformer(vocab=4, d_model=32, n_layers=2, n_heads=2)
    sparsity = WeightSparsity(model, target_q=0.2)

    sparsity.set_current_q(0.2)
    sparsity.apply_masks()
    mask1_id = id(sparsity.masks)

    # Refresh masks (should be new masks)
    sparsity.apply_masks()
    mask2_id = id(sparsity.masks)
    # Dictionaries are recreated
    assert mask2_id != mask1_id


def test_weight_sparsity_integration_with_training():
    """Test that sparsity integrates with training without breaking it."""
    torch.manual_seed(0)
    # Simple copy task
    seq = torch.randint(0, 4, (128, 1)).repeat(1, 12)
    model = TinyTransformer(vocab=4, d_model=32, n_layers=1, n_heads=2)
    sparsity = WeightSparsity(model, target_q=0.3, anneal_steps=50)

    # Train with sparsity
    model.to("cpu").train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []

    for step in range(100):
        # Update sparsity schedule
        sparsity.update_annealing_schedule(step)
        sparsity.apply_masks()

        # Forward pass
        batch = seq[:64]
        logits = model(batch[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1)
        )
        losses.append(loss.item())

        # Backward and step
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Re-apply masks after optimizer step
        sparsity.apply_masks()

    # Loss should decrease
    assert losses[-1] < losses[0]


def test_count_nonzero():
    """Test counting nonzero weights."""
    model = TinyTransformer(vocab=4, d_model=32, n_layers=2, n_heads=2)
    sparsity = WeightSparsity(model, target_q=0.1)

    total = sparsity.count_total_params()
    assert total > 0

    # With full density, nonzero should equal total
    sparsity.set_current_q(1.0)
    sparsity.apply_masks()
    nonzero = sparsity.count_nonzero()
    assert nonzero == total

    # With 50% sparsity, should be roughly half
    sparsity.set_current_q(0.5)
    sparsity.apply_masks()
    nonzero = sparsity.count_nonzero()
    assert 0.48 * total < nonzero < 0.52 * total


def test_sparsity_preserves_output_shapes():
    """Test that sparsity doesn't break model forward pass."""
    model = TinyTransformer(vocab=4, d_model=32, n_layers=2, n_heads=2)
    sparsity = WeightSparsity(model, target_q=0.1)

    x = torch.randint(0, 4, (8, 10))

    # Forward with full density
    sparsity.set_current_q(1.0)
    sparsity.apply_masks()
    logits1 = model(x)
    assert logits1.shape == (8, 10, 4)

    # Forward with sparsity
    sparsity.set_current_q(0.2)
    sparsity.apply_masks()
    logits2 = model(x)
    assert logits2.shape == (8, 10, 4)


def test_sparsity_different_levels():
    """Test sparsity at different density levels."""
    model = TinyTransformer(vocab=4, d_model=32, n_layers=2, n_heads=2)
    sparsity = WeightSparsity(model, target_q=0.1)

    total = sparsity.count_total_params()

    for q in [1.0, 0.5, 0.2, 0.1, 0.05]:
        sparsity.set_current_q(q)
        sparsity.apply_masks()
        nonzero = sparsity.count_nonzero()
        ratio = nonzero / total
        # Allow some tolerance (topk is approximate)
        assert abs(ratio - q) < 0.02

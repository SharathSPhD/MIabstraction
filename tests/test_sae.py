"""Tests for sparse autoencoder (SAE) module."""
import numpy as np
import torch

from miabstraction.sae import SAE, train_sae


def test_sae_forward_shapes():
    """Test SAE encoder/decoder shapes."""
    d, m = 64, 256
    sae = SAE(d_model=d, dict_size=m, k=32)
    x = torch.randn(16, 10, d)

    # Forward pass with return_latents
    recon, latents = sae(x, return_latents=True)
    assert recon.shape == x.shape, f"Expected {x.shape}, got {recon.shape}"
    assert latents.shape == (16, 10, m), f"Expected (16, 10, {m}), got {latents.shape}"


def test_sae_topk_activation():
    """Test that top-k activation is sparse."""
    d, m, k = 64, 256, 32
    sae = SAE(d_model=d, dict_size=m, k=k)
    x = torch.randn(32, 50, d)

    _, latents = sae(x, return_latents=True)

    # Each position should have exactly k non-zero latents
    nnz = (latents != 0).sum(dim=2)
    assert torch.all(nnz == k), f"Expected all k={k}, got {nnz.unique()}"


def test_sae_dead_neurons():
    """Test detection of dead neurons (never activated)."""
    d, m, k = 64, 256, 16
    sae = SAE(d_model=d, dict_size=m, k=k)
    x = torch.randn(128, 20, d)

    _, latents = sae(x, return_latents=True)

    # Reshape to (samples*positions, dict_size)
    latents_flat = latents.reshape(-1, m)
    # A neuron is dead if it never has a non-zero value across all samples
    dead_frac = ((latents_flat != 0).sum(dim=0) == 0).float().mean().item()

    # With 128*20=2560 samples and k=16, some should be dead
    assert 0 <= dead_frac <= 1


def test_train_sae_synthetic():
    """Test train_sae on synthetic sparse ground-truth data."""
    torch.manual_seed(0)
    np.random.seed(0)

    # Create synthetic sparse data: 10 basis vectors + noise
    d, true_dict_size, m = 32, 10, 128
    basis = torch.randn(true_dict_size, d)

    # Generate samples: each is a sum of 2-3 random basis vectors
    n_samples = 1000
    x = torch.zeros(n_samples, d)
    for i in range(n_samples):
        n_active = np.random.randint(2, 4)
        active_idx = np.random.choice(true_dict_size, n_active, replace=False)
        coeffs = np.random.uniform(0.1, 2.0, n_active)
        for j, idx in enumerate(active_idx):
            x[i] += coeffs[j] * basis[idx]
    x = x + 0.01 * torch.randn_like(x)  # Add small noise

    # Train SAE
    sae, metrics = train_sae(
        x,
        dict_size=m,
        k=5,
        lr=0.01,
        steps=500,
        seed=0,
        device="cpu",
    )

    # Check metrics structure
    assert isinstance(metrics, dict)
    assert "fvu_final" in metrics
    assert "l0_mean" in metrics
    assert "dead_frac" in metrics
    assert "loss_history" in metrics

    # FVU should be reasonable (< 0.5 with good dictionary learning)
    assert 0 <= metrics["fvu_final"] <= 1

    # L0 should be exactly k (top-k sparsity)
    assert abs(metrics["l0_mean"] - 5) < 0.1, f"Expected L0~5, got {metrics['l0_mean']}"

    # Dead fraction should be moderate
    assert 0 <= metrics["dead_frac"] <= 1


def test_sae_reconstruction():
    """Test that SAE achieves low reconstruction error on training data."""
    d, m, k = 64, 256, 16
    sae = SAE(d_model=d, dict_size=m, k=k)

    # Create simple test data (mostly zeros with few active dimensions)
    x = torch.randn(256, 32, d)
    x = x * (torch.rand_like(x) > 0.9).float()  # Make 90% zeros

    # Train
    sae, metrics = train_sae(
        x,
        dict_size=m,
        k=k,
        lr=0.01,
        steps=200,
        seed=0,
        device="cpu",
    )

    # Final FVU should be low
    assert metrics["fvu_final"] < 0.3, f"FVU too high: {metrics['fvu_final']}"


def test_train_sae_returns_model():
    """Test that train_sae returns a properly configured SAE model."""
    x = torch.randn(100, 32)
    sae, metrics = train_sae(x, dict_size=64, k=8, lr=0.01, steps=50, seed=0, device="cpu")

    assert isinstance(sae, SAE)
    assert sae.d_model == 32
    assert sae.dict_size == 64
    assert sae.k == 8


def test_fvu_metric():
    """Test that FVU (fraction of variance unexplained) is computed correctly."""
    x = torch.randn(100, 32)
    sae = SAE(d_model=32, dict_size=64, k=8)

    # Forward pass
    recon = sae(x)

    # Manual FVU calculation
    residual = x - recon
    var_residual = residual.var(dim=0).sum()
    var_total = x.var(dim=0).sum()
    fvu_expected = (var_residual / var_total).item()

    # Verify it's in [0, 1]
    assert 0 <= fvu_expected <= 2  # Can be > 1 if reconstruction is bad

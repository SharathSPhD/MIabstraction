import os

import numpy as np
import torch

from miabstraction.seeding import set_determinism


def test_cublas_workspace_configured_at_import():
    """Must be set before CUDA init or deterministic matmuls are impossible."""
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8")


def test_same_seed_reproduces_draws():
    set_determinism(7)
    a = (np.random.rand(4), torch.randn(4))
    set_determinism(7)
    b = (np.random.rand(4), torch.randn(4))
    np.testing.assert_array_equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_different_seeds_differ():
    set_determinism(1)
    a = torch.randn(8)
    set_determinism(2)
    assert not torch.equal(a, torch.randn(8))


def test_training_is_reproducible_under_determinism():
    """Two identically-seeded trainings must produce identical losses."""
    from miabstraction.models import TinyTransformer, train_lm

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seq = torch.randint(0, 4, (128, 1)).repeat(1, 8)

    def run():
        set_determinism(3)
        m = TinyTransformer(vocab=4, d_model=32, n_layers=1, n_heads=2, max_len=16)
        return train_lm(m, seq, steps=30, batch_size=32, lr=1e-3, device=dev)

    assert run() == run()

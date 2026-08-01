import numpy as np
import torch

from miabstraction.experiments.e2_induction import (
    attention_patterns,
    copy_region_losses,
    doubled_sequences,
    find_transition,
    gapped_doubled_sequences,
    prefix_matching_score,
)
from miabstraction.models import TinyTransformer


def test_doubled_sequences_structure():
    rng = np.random.default_rng(0)
    s = doubled_sequences(10, 8, 5, rng)
    assert s.shape == (10, 16)
    np.testing.assert_array_equal(s[:, :8], s[:, 8:])


def test_gapped_doubled_sequences_structure():
    rng = np.random.default_rng(0)
    seqs, gaps = gapped_doubled_sequences(50, 12, 6, 7, rng)
    assert seqs.shape == (50, 30)
    assert gaps.min() >= 0 and gaps.max() <= 6
    for i in range(50):
        g = gaps[i]
        np.testing.assert_array_equal(seqs[i, :12], seqs[i, 12 + g : 24 + g])


def test_attention_patterns_shapes_and_causality():
    m = TinyTransformer(vocab=5, d_model=32, n_layers=2, n_heads=2, attn_only=True)
    x = torch.randint(0, 5, (3, 12))
    pats = attention_patterns(m, x)
    assert len(pats) == 2
    assert pats[0].shape == (3, 2, 12, 12)
    upper = torch.triu(torch.ones(12, 12, dtype=torch.bool), diagonal=1)
    assert pats[0][:, :, upper].abs().max().item() < 1e-6
    assert torch.allclose(pats[0].sum(-1), torch.ones(3, 2, 12), atol=1e-5)


def test_prefix_score_near_uniform_for_untrained():
    torch.manual_seed(0)
    m = TinyTransformer(vocab=5, d_model=32, n_layers=2, n_heads=2, attn_only=True,
                        max_len=64)
    seqs, gaps = gapped_doubled_sequences(16, 12, 6, 5, np.random.default_rng(1))
    s = prefix_matching_score(m, torch.from_numpy(seqs), gaps, 12)
    assert 0.0 <= s < 0.3
    per_layer = prefix_matching_score(m, torch.from_numpy(seqs), gaps, 12,
                                      per_layer=True)
    assert len(per_layer) == 2


def test_copy_region_losses_returns_floats():
    m = TinyTransformer(vocab=5, d_model=32, n_layers=1, n_heads=2, attn_only=True,
                        max_len=64)
    seqs, gaps = gapped_doubled_sequences(8, 12, 6, 5, np.random.default_rng(2))
    lf, ls = copy_region_losses(m, torch.from_numpy(seqs), gaps, 12)
    assert lf > 0 and ls > 0


def test_find_transition():
    steps = [0, 100, 200, 300, 400]
    scores = [0.05, 0.1, 0.15, 0.7, 0.8]
    assert find_transition(steps, scores) == (200, 300)
    assert find_transition(steps, [0.1] * 5) is None

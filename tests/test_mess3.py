import numpy as np
import pytest

from miabstraction.data.mess3 import (
    belief_states,
    mess3_matrices,
    sample_sequences,
    stationary_dist,
)


@pytest.fixture
def T():
    return mess3_matrices()


def test_matrices_are_jointly_stochastic(T):
    # sum over symbols and next states must be 1 per current state
    row_sums = T.sum(axis=(0, 2))
    np.testing.assert_allclose(row_sums, np.ones(3), atol=1e-12)
    assert (T >= 0).all()


def test_stationary_dist_is_fixed_point(T):
    pi = stationary_dist(T)
    M = T.sum(axis=0)
    np.testing.assert_allclose(pi @ M, pi, atol=1e-10)
    assert pi.sum() == pytest.approx(1.0)


def test_sample_sequences_shape_and_alphabet(T):
    rng = np.random.default_rng(0)
    seqs = sample_sequences(T, n_seq=100, seq_len=32, rng=rng)
    assert seqs.shape == (100, 32)
    assert set(np.unique(seqs)) <= {0, 1, 2}


def test_belief_states_on_simplex(T):
    rng = np.random.default_rng(1)
    seqs = sample_sequences(T, 50, 16, rng)
    b = belief_states(T, seqs)
    assert b.shape == (50, 16, 3)
    np.testing.assert_allclose(b.sum(axis=-1), 1.0, atol=1e-9)
    assert (b >= 0).all()


def test_belief_states_are_nontrivial(T):
    """Belief states must actually move (fractal MSP), not collapse to stationary."""
    rng = np.random.default_rng(2)
    seqs = sample_sequences(T, 200, 24, rng)
    b = belief_states(T, seqs)
    late = b[:, -1, :]
    assert late.std(axis=0).max() > 0.05


def test_symbol_marginals_match_process(T):
    """Empirical symbol frequencies should match pi @ T[a] row sums."""
    pi = stationary_dist(T)
    expected = np.array([float(pi @ T[a].sum(axis=1)) for a in range(3)])
    rng = np.random.default_rng(3)
    seqs = sample_sequences(T, 2000, 64, rng)
    counts = np.bincount(seqs.ravel(), minlength=3) / seqs.size
    np.testing.assert_allclose(counts, expected, atol=0.01)

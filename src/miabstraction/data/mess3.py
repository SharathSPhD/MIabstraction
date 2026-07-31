"""Mess3 hidden Markov process and its mixed-state presentation (belief states).

Mess3 (Marzen & Crutchfield 2017; used by Shai et al. 2024 "Transformers represent
belief state geometry in their residual stream") is a 3-state, 3-symbol edge-emitting
HMM whose belief-state geometry is a fractal subset of the 2-simplex.

T[a, i, j] = P(emit symbol a, transition i -> j | hidden state i).
"""
from __future__ import annotations

import numpy as np


def mess3_matrices(x: float = 0.15, a: float = 0.6) -> np.ndarray:
    """Labeled transition matrices, shape (3 symbols, 3 states, 3 states)."""
    b = (1 - a) / 2
    y = 1 - 2 * x
    ay, by, ax, bx = a * y, b * y, a * x, b * x
    T = np.array([
        [[ay, bx, bx], [ax, by, bx], [ax, bx, by]],
        [[by, ax, bx], [bx, ay, bx], [bx, ax, by]],
        [[by, bx, ax], [bx, by, ax], [bx, bx, ay]],
    ])
    return T


def stationary_dist(T: np.ndarray) -> np.ndarray:
    """Stationary distribution of the marginal state chain sum_a T[a]."""
    M = T.sum(axis=0)  # (states, states) row-stochastic
    vals, vecs = np.linalg.eig(M.T)
    i = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.real(vecs[:, i])
    pi = np.abs(pi) / np.abs(pi).sum()
    return pi


def sample_sequences(
    T: np.ndarray, n_seq: int, seq_len: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample token sequences from the HMM, shape (n_seq, seq_len), values in {0..A-1}."""
    A, S, _ = T.shape
    pi = stationary_dist(T)
    # P(emit a, next j | state i) flattened for vectorized sampling
    flat = T.transpose(1, 0, 2).reshape(S, A * S)  # (state, symbol*next)
    states = rng.choice(S, size=n_seq, p=pi)
    out = np.empty((n_seq, seq_len), dtype=np.int64)
    for t in range(seq_len):
        # sample (symbol, next_state) jointly per sequence
        u = rng.random(n_seq)
        cdf = np.cumsum(flat[states], axis=1)
        idx = (u[:, None] < cdf).argmax(axis=1)
        out[:, t] = idx // S
        states = idx % S
    return out


def belief_states(T: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """Ground-truth belief states (mixed-state presentation) after each prefix.

    Returns array (n_seq, seq_len, n_states): b_t = P(hidden state | tokens[:t+1]),
    computed by the normalized forward recursion b' = b @ T[a] / ||.||_1.
    """
    n_seq, L = tokens.shape
    S = T.shape[1]
    pi = stationary_dist(T)
    b = np.tile(pi, (n_seq, 1))
    out = np.empty((n_seq, L, S))
    for t in range(L):
        a = tokens[:, t]
        b = np.einsum("ns,nst->nt", b, T[a])
        b /= b.sum(axis=1, keepdims=True)
        out[:, t] = b
    return out

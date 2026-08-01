"""Layer-by-layer verification of the hand-constructed induction circuit."""
import numpy as np
import pytest
import torch

from loom.constructed import (
    _codes,
    _shift_matrix,
    compile_induction,
)
from miabstraction.experiments.e2_induction import (
    attention_patterns,
    gapped_doubled_sequences,
    prefix_matching_score,
)

VOCAB, COPY, GAP = 20, 22, 16          # total length 60 <= max_len 63
MAXLEN = 60


@pytest.fixture(scope="module")
def model():
    return compile_induction(vocab=VOCAB, max_len=MAXLEN)


@pytest.fixture(scope="module")
def data():
    seqs, gaps = gapped_doubled_sequences(64, COPY, GAP, VOCAB,
                                          np.random.default_rng(0))
    return torch.from_numpy(seqs), gaps


def test_codes_zero_mean_orthogonal():
    c = _codes(20, 32)
    np.testing.assert_allclose(c.sum(axis=1), 0)
    gram = c @ c.T
    np.testing.assert_allclose(gram, np.eye(20) * 32)


def test_shift_matrix_shifts():
    pos = _codes(10, 64)
    S = _shift_matrix(pos, 1)
    np.testing.assert_allclose(S @ pos[3], pos[2], atol=1e-9)
    np.testing.assert_allclose(S @ pos[0], 0, atol=1e-9)


def test_layer0_is_previous_token_head(model, data):
    tokens, _ = data
    w = attention_patterns(model, tokens[:8])[0]  # (B,1,L,L)
    L = tokens.shape[1]
    diag_mass = w[:, 0, torch.arange(1, L), torch.arange(0, L - 1)]
    assert diag_mass.mean().item() > 0.95


def test_layer1_is_shift2_head(model, data):
    tokens, _ = data
    w = attention_patterns(model, tokens[:8])[1]
    L = tokens.shape[1]
    diag_mass = w[:, 0, torch.arange(2, L), torch.arange(0, L - 2)]
    assert diag_mass.mean().item() > 0.95


def test_layer2_attends_to_induction_target(model, data):
    tokens, gaps = data
    score = prefix_matching_score(model, tokens, gaps, COPY)
    assert score > 0.5, f"prefix score {score}"


def test_icl_argmax_accuracy(model, data):
    tokens, gaps = data
    with torch.no_grad():
        logits = model(tokens[:, :-1])
    correct, total = 0, 0
    for b in range(tokens.shape[0]):
        g = int(gaps[b])
        # predictable second-copy targets x[2..]: positions COPY+g+2 .. 2*COPY+g-1
        for p in range(COPY + g + 2, 2 * COPY + g):
            pred = logits[b, p - 1].argmax().item()
            if pred == tokens[b, p].item():
                correct += 1
            total += 1
    acc = correct / total
    assert acc > 0.9, f"icl argmax accuracy {acc}"


def test_negative_control_no_repeats(model):
    """On pure random sequences the circuit has nothing to induce on: the
    prefix-score measured against a FAKE alignment must be ~chance."""
    rng = np.random.default_rng(3)
    seqs = rng.integers(0, VOCAB, size=(32, MAXLEN - 1), dtype=np.int64)
    fake_gaps = np.zeros(32, dtype=int)
    s = prefix_matching_score(model, torch.from_numpy(seqs), fake_gaps, COPY)
    assert s < 0.2

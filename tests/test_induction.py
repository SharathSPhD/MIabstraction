"""Tests for induction data generator.

Induction task: sequences with random segments repeated later, where the optimal
strategy is to attend to previous occurrences of the current token and copy its successor.
"""
import numpy as np
import pytest

from miabstraction.data.induction import generate_induction_sequences


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_generate_induction_sequences_shape(rng):
    """Output shape is (n_seq, seq_len)."""
    sequences, repeat_mask = generate_induction_sequences(
        n_seq=100, seq_len=64, vocab=20, repeat_len=8, rng=rng
    )
    assert sequences.shape == (100, 64)
    assert repeat_mask.shape == (100, 64)
    assert sequences.dtype == np.int64
    assert repeat_mask.dtype == bool


def test_generate_induction_sequences_vocabulary(rng):
    """All tokens are in [0, vocab)."""
    sequences, _ = generate_induction_sequences(
        n_seq=50, seq_len=32, vocab=10, repeat_len=4, rng=rng
    )
    assert (sequences >= 0).all() and (sequences < 10).all()


def test_generate_induction_sequences_repeats_exist(rng):
    """Each sequence has at least one repeat region."""
    sequences, repeat_mask = generate_induction_sequences(
        n_seq=100, seq_len=64, vocab=20, repeat_len=8, rng=rng
    )
    # At least one position should be in a repeat region
    assert (repeat_mask.sum(axis=1) > 0).all()


def test_generate_induction_sequences_actual_repeats(rng):
    """The repeated segment actually appears twice in each sequence."""
    sequences, repeat_mask = generate_induction_sequences(
        n_seq=50, seq_len=64, vocab=20, repeat_len=8, rng=rng
    )
    for i in range(sequences.shape[0]):
        seq = sequences[i]
        mask = repeat_mask[i]
        # Find the repeat region (where mask is True)
        repeat_positions = np.where(mask)[0]
        if len(repeat_positions) > 0:
            # The repeated segment should span some positions
            start_repeat = repeat_positions[0]
            # We should be able to find that segment earlier in the sequence
            repeat_seg = seq[start_repeat : start_repeat + len(repeat_positions)]
            # Check if this segment appears earlier (before the mask region)
            found_earlier = False
            for j in range(start_repeat - len(repeat_positions) + 1):
                if np.array_equal(seq[j : j + len(repeat_positions)], repeat_seg):
                    found_earlier = True
                    break
            assert found_earlier, f"Sequence {i}: repeated segment not found earlier"


def test_generate_induction_sequences_repeat_length_consistent(rng):
    """All repeat regions have the expected length."""
    repeat_len = 6
    sequences, repeat_mask = generate_induction_sequences(
        n_seq=50, seq_len=64, vocab=20, repeat_len=repeat_len, rng=rng
    )
    for i in range(sequences.shape[0]):
        mask = repeat_mask[i]
        # Count contiguous True regions
        transitions = np.diff(mask.astype(int), prepend=0, append=0)
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]
        for start, end in zip(starts, ends):
            repeat_region_len = end - start
            assert (
                repeat_region_len == repeat_len
            ), f"Repeat region length {repeat_region_len} != {repeat_len}"


def test_generate_induction_sequences_repeat_position_varies(rng):
    """Repeat segments appear at different positions across sequences."""
    sequences, repeat_mask = generate_induction_sequences(
        n_seq=100, seq_len=64, vocab=20, repeat_len=8, rng=rng
    )
    # Collect the starting positions of repeat regions
    repeat_starts = []
    for i in range(sequences.shape[0]):
        mask = repeat_mask[i]
        if mask.any():
            first_true = np.where(mask)[0][0]
            repeat_starts.append(first_true)
    # Positions should vary (not all the same)
    assert len(set(repeat_starts)) > 1, "Repeat positions are always the same"


def test_generate_induction_sequences_deterministic_with_seed(rng):
    """Same seed produces same sequences."""
    seq1, mask1 = generate_induction_sequences(
        n_seq=10, seq_len=32, vocab=10, repeat_len=4, rng=np.random.default_rng(42)
    )
    seq2, mask2 = generate_induction_sequences(
        n_seq=10, seq_len=32, vocab=10, repeat_len=4, rng=np.random.default_rng(42)
    )
    np.testing.assert_array_equal(seq1, seq2)
    np.testing.assert_array_equal(mask1, mask2)


def test_generate_induction_sequences_no_repeats_outside_mask(rng):
    """Tokens outside the repeat mask are generally different from repeat segment."""
    sequences, repeat_mask = generate_induction_sequences(
        n_seq=20, seq_len=64, vocab=20, repeat_len=8, rng=rng
    )
    for i in range(min(5, sequences.shape[0])):  # Check a few sequences
        seq = sequences[i]
        mask = repeat_mask[i]
        repeat_region_tokens = seq[mask]
        non_repeat_tokens = seq[~mask]
        # At least some differences should exist (with high vocab, unlikely to match by chance)
        if len(repeat_region_tokens) > 0 and len(non_repeat_tokens) > 0:
            assert not np.array_equal(
                repeat_region_tokens, non_repeat_tokens[:len(repeat_region_tokens)]
            )

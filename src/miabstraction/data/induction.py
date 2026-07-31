"""Induction task data generator: synthetic sequences with repeated subsequences.

Each sequence contains a random segment that is repeated later. The optimal
prediction strategy is induction: when seeing a token t at position p_repeat,
attend to the previous occurrence at p_first and copy the token that followed it.

This is the key capability tested in "Induction Heads Implement In-context Learning"
and related work on mechanistic interpretability.
"""
from __future__ import annotations

import numpy as np


def generate_induction_sequences(
    n_seq: int,
    seq_len: int,
    vocab: int,
    repeat_len: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate sequences with repeated random segments.

    Args:
        n_seq: number of sequences
        seq_len: length of each sequence
        vocab: vocabulary size (tokens in 0..vocab-1)
        repeat_len: length of the segment to repeat
        rng: numpy random generator for determinism

    Returns:
        (sequences, repeat_mask) where:
        - sequences: (n_seq, seq_len) int64 array of token IDs
        - repeat_mask: (n_seq, seq_len) bool array indicating repeat-region positions
    """
    sequences = np.zeros((n_seq, seq_len), dtype=np.int64)
    repeat_mask = np.zeros((n_seq, seq_len), dtype=bool)

    for i in range(n_seq):
        # Generate random tokens for the sequence
        seq = rng.integers(0, vocab, size=seq_len, dtype=np.int64)

        # Pick a random position for the first occurrence of the segment
        # Ensure there's room for: first segment + gap + second segment
        min_gap = 2  # at least 2 tokens between occurrences
        max_first_pos = seq_len - 2 * repeat_len - min_gap
        if max_first_pos < 0:
            # If sequence is too short, just put the first segment early
            first_pos = max(0, seq_len - 2 * repeat_len - min_gap)
        else:
            first_pos = rng.integers(0, max_first_pos + 1)

        # Generate the segment to repeat
        segment = rng.integers(0, vocab, size=repeat_len, dtype=np.int64)

        # Place the first occurrence
        seq[first_pos : first_pos + repeat_len] = segment

        # Pick a position for the second (repeat) occurrence
        # Must be after the first occurrence with some gap
        repeat_pos_min = first_pos + repeat_len + min_gap
        repeat_pos_max = seq_len - repeat_len
        if repeat_pos_max >= repeat_pos_min:
            repeat_pos = rng.integers(repeat_pos_min, repeat_pos_max + 1)
        else:
            # Not enough room for a gap and second segment; place it as late as possible
            repeat_pos = max(first_pos + repeat_len, seq_len - repeat_len)
            if repeat_pos + repeat_len > seq_len:
                # Adjust to fit within sequence
                repeat_pos = seq_len - repeat_len

        # Place the second (repeat) occurrence
        if repeat_pos + repeat_len <= seq_len:
            seq[repeat_pos : repeat_pos + repeat_len] = segment
            repeat_mask[i, repeat_pos : repeat_pos + repeat_len] = True

        sequences[i] = seq

    return sequences, repeat_mask

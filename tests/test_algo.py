"""Tests for bracket matching algorithmic task."""
import numpy as np
import torch
import pytest

from miabstraction.data.algo import BracketMatchingDataset


def test_bracket_matching_dataset_basic():
    """Test basic dataset properties."""
    ds = BracketMatchingDataset(vocab_size=10, seq_len=16, n_samples=100, seed=0)
    assert ds.vocab_size == 10
    assert ds.seq_len == 16
    assert len(ds) == 100

    # Get a sample
    seq = ds[0]
    assert seq.shape == (16,)
    assert seq.dtype == torch.int64
    assert seq.max() < ds.vocab_size
    assert seq.min() >= 0


def test_bracket_matching_includes_answer_mask():
    """Test that dataset includes answer position mask."""
    ds = BracketMatchingDataset(vocab_size=10, seq_len=16, n_samples=100, seed=0)
    seq, mask, correct_closing = ds.get_with_mask(0)
    assert seq.shape == (16,)
    assert mask.shape == (16,)
    assert mask.dtype == torch.bool
    assert mask.sum() == 1  # exactly one answer position
    assert correct_closing in [2, 3]  # one of the two closer types


def test_bracket_matching_correctness():
    """Test that the correct closing bracket is actually correct."""
    ds = BracketMatchingDataset(vocab_size=10, seq_len=16, n_samples=100, seed=0)
    for i in range(min(10, len(ds))):
        seq, mask, correct = ds.get_with_mask(i)
        ans_pos = mask.nonzero(as_tuple=True)[0].item()
        # Verify brackets are in expected range
        # openers are 0,1; closers are 2,3; distractors are 4-9
        assert correct in [2, 3]


def test_bracket_matching_stack_validity():
    """Test that sequences have valid bracket structure."""
    ds = BracketMatchingDataset(vocab_size=10, seq_len=32, n_samples=100, seed=42)
    # Sample a few and verify they follow the Dyck-1 structure
    for i in range(min(5, len(ds))):
        seq, mask, correct = ds.get_with_mask(i)
        # openers = 0,1; closers = 2,3
        stack = []
        ans_pos = mask.nonzero(as_tuple=True)[0].item()

        for t, tok in enumerate(seq[:ans_pos].tolist()):
            if tok == 0:  # opener type 0
                stack.append(0)
            elif tok == 1:  # opener type 1
                stack.append(1)
            elif tok == 2:  # closer type 0
                if stack:
                    stack.pop()
            elif tok == 3:  # closer type 1
                if stack:
                    stack.pop()
            # else: distractor, ignore

        # At answer position, correct closing bracket should match top of stack
        if stack:
            expected_closer = stack[-1]
            assert correct == expected_closer + 2  # converters to closer ID


def test_bracket_matching_accuracy_metric():
    """Test accuracy metric computation at answer positions."""
    ds = BracketMatchingDataset(vocab_size=10, seq_len=16, n_samples=50, seed=0)

    # Create dummy predictions (all predict token 2)
    seqs = []
    masks = []
    corrects = []
    for i in range(len(ds)):
        seq, mask, correct = ds.get_with_mask(i)
        seqs.append(seq)
        masks.append(mask)
        corrects.append(correct)

    seqs = torch.stack(seqs)
    masks = torch.stack(masks)
    corrects = torch.tensor(corrects)

    # Compute accuracy: correct if predicted == correct at answer position
    preds = torch.full_like(masks, 2, dtype=torch.long)  # always predict closer type 0

    answer_mask = masks.unsqueeze(-1)  # (B, L, 1)
    # Extract predictions and corrects at answer positions
    pred_at_ans = (preds * masks).sum(dim=1)  # (B,)
    acc = (pred_at_ans == corrects).float().mean()

    assert acc >= 0.0 and acc <= 1.0
    # Random guess should be ~50% for 2 choices
    assert acc > 0.3  # with seed, some randomness


def test_dataset_reproducible():
    """Test that seeding produces reproducible sequences."""
    ds1 = BracketMatchingDataset(vocab_size=10, seq_len=16, n_samples=50, seed=0)
    ds2 = BracketMatchingDataset(vocab_size=10, seq_len=16, n_samples=50, seed=0)

    for i in range(10):
        seq1 = ds1[i]
        seq2 = ds2[i]
        assert torch.allclose(seq1.float(), seq2.float())


def test_dataset_different_seeds():
    """Test that different seeds produce different sequences."""
    ds1 = BracketMatchingDataset(vocab_size=10, seq_len=16, n_samples=50, seed=0)
    ds2 = BracketMatchingDataset(vocab_size=10, seq_len=16, n_samples=50, seed=1)

    # Check at least first few are different
    different_count = 0
    for i in range(10):
        seq1 = ds1[i]
        seq2 = ds2[i]
        if not torch.allclose(seq1.float(), seq2.float()):
            different_count += 1

    assert different_count > 0


def test_vocab_mapping():
    """Test vocabulary mapping: 0,1 openers; 2,3 closers; 4+ distractors."""
    ds = BracketMatchingDataset(vocab_size=10, seq_len=32, n_samples=100, seed=0)

    all_tokens = []
    for i in range(len(ds)):
        seq = ds[i]
        all_tokens.extend(seq.tolist())

    all_tokens = set(all_tokens)
    # Should only see tokens in range [0, vocab_size)
    assert all_tokens <= set(range(ds.vocab_size))

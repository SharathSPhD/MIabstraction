"""Algorithmic tasks: bracket matching (Dyck-1 with two bracket types + distractors)."""
from __future__ import annotations

import torch
import numpy as np


class BracketMatchingDataset:
    """Bracket matching task: sequences over small vocab with two opener types.

    Vocabulary mapping:
    - 0, 1: opener types (type 0, type 1)
    - 2, 3: closer types (matches type 0, matches type 1)
    - 4-9: distractor tokens (irrelevant to task)

    Query position: marked by answer_mask. Model predicts the closing bracket
    matching the most recent unmatched opener.
    """

    def __init__(
        self,
        vocab_size: int = 10,
        seq_len: int = 16,
        n_samples: int = 1000,
        seed: int = 0,
    ):
        assert vocab_size >= 4, "vocab_size must be >= 4 (2 openers + 2 closers)"
        assert seq_len >= 8, "seq_len must be >= 8"
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Generate all sequences and answer info at init
        self.sequences = []
        self.answer_masks = []
        self.correct_closers = []
        self._generate_sequences()

    def _generate_sequences(self):
        """Generate n_samples valid bracket-matching sequences."""
        for _ in range(self.n_samples):
            seq, mask, closer = self._generate_one_sequence()
            self.sequences.append(seq)
            self.answer_masks.append(mask)
            self.correct_closers.append(closer)

    def _generate_one_sequence(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Generate a single valid bracket sequence.

        Returns:
            seq: (seq_len,) array with tokens
            mask: (seq_len,) boolean mask with single True at answer position
            closer: int in {2, 3}, the correct closing bracket
        """
        seq = np.zeros(self.seq_len, dtype=np.int64)

        # Reserve last few positions for answer
        max_bracket_pos = self.seq_len - 4
        min_brackets = 2
        n_brackets = self.rng.integers(min_brackets, (max_bracket_pos - 2) // 2 + 1)

        # Generate bracket sequence with balanced structure
        stack = []
        bracket_pos = 0

        for _ in range(n_brackets):
            # Randomly choose: open or close (biased toward open in first half)
            if not stack or self.rng.random() < 0.6:
                # Open
                opener_type = self.rng.integers(0, 2)
                seq[bracket_pos] = opener_type
                stack.append(opener_type)
            else:
                # Close
                opener_type = stack.pop()
                seq[bracket_pos] = opener_type + 2  # convert to closer
            bracket_pos += 1

            if bracket_pos >= max_bracket_pos:
                break

        # Ensure we have at least one unmatched opener for the query
        if not stack:
            # Open one more
            opener_type = self.rng.integers(0, 2)
            seq[bracket_pos] = opener_type
            stack.append(opener_type)

        # Fill remaining positions before answer with distractors
        answer_pos = self.rng.integers(bracket_pos + 2, self.seq_len - 1)
        for i in range(bracket_pos, answer_pos):
            seq[i] = self.rng.integers(4, self.vocab_size)  # distractor

        # Set answer position
        correct_closer = stack[-1] + 2  # convert most recent opener to closer
        seq[answer_pos] = correct_closer  # put correct answer at query position
        # (This is for reference; model must predict it)

        # Fill any remaining positions with distractors
        for i in range(answer_pos + 1, self.seq_len):
            seq[i] = self.rng.integers(4, self.vocab_size)

        # Create answer mask: single True at answer_pos
        mask = np.zeros(self.seq_len, dtype=bool)
        mask[answer_pos] = True

        return seq, mask, correct_closer

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a sequence as a tensor."""
        return torch.tensor(self.sequences[idx], dtype=torch.int64)

    def get_with_mask(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return sequence, answer mask, and correct closing token.

        Args:
            idx: sample index

        Returns:
            seq: (seq_len,) int64 tensor
            mask: (seq_len,) bool tensor with single True at answer position
            correct_closer: int in {2, 3}
        """
        return (
            torch.tensor(self.sequences[idx], dtype=torch.int64),
            torch.tensor(self.answer_masks[idx], dtype=torch.bool),
            int(self.correct_closers[idx]),
        )

    def compute_accuracy_at_answer_position(
        self, logits: torch.Tensor, indices: list[int]
    ) -> float:
        """Compute accuracy only at answer positions.

        Args:
            logits: (B, seq_len, vocab_size) model output
            indices: list of B sample indices

        Returns:
            accuracy: fraction correct at answer positions
        """
        correct_count = 0
        for b, idx in enumerate(indices):
            mask = self.answer_masks[idx]
            ans_pos = np.where(mask)[0][0]

            pred = logits[b, ans_pos].argmax().item()
            expected = self.correct_closers[idx]

            if pred == expected:
                correct_count += 1

        return correct_count / len(indices) if indices else 0.0

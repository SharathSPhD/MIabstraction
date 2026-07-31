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

        CRITICAL: Ensures at least one unmatched opener at answer position.
        This prevents ill-defined samples where model has no correct answer.

        Returns:
            seq: (seq_len,) array with tokens
            mask: (seq_len,) boolean mask with single True at answer position
            closer: int in {2, 3}, the correct closing bracket
        """
        seq = np.zeros(self.seq_len, dtype=np.int64)

        # Generate bracket sequence that GUARANTEES unmatched opener at query
        stack = []
        bracket_pos = 0

        # Phase 1: Generate some opening brackets to build stack
        n_opens = self.rng.integers(1, 3)  # 1-2 initial opens
        for _ in range(n_opens):
            opener_type = self.rng.integers(0, 2)
            seq[bracket_pos] = opener_type
            stack.append(opener_type)
            bracket_pos += 1

        # Phase 2: Add more brackets (mix of opens/closes) for complexity
        max_bracket_pos = self.seq_len - 3  # Leave room for answer
        while bracket_pos < max_bracket_pos:
            if not stack or self.rng.random() < 0.4:
                # Open: grow stack
                opener_type = self.rng.integers(0, 2)
                seq[bracket_pos] = opener_type
                stack.append(opener_type)
            else:
                # Close: shrink stack
                opener_type = stack.pop()
                seq[bracket_pos] = opener_type + 2
            bracket_pos += 1

            # CRITICAL: If stack becomes empty, add an opener immediately
            if not stack and bracket_pos < max_bracket_pos:
                opener_type = self.rng.integers(0, 2)
                seq[bracket_pos] = opener_type
                stack.append(opener_type)
                bracket_pos += 1

        # CRITICAL: Verify stack is non-empty (should never fail)
        assert len(stack) > 0, "Stack must have at least one unmatched opener"

        # Phase 3: Add distractors before answer position
        min_ans_pos = bracket_pos + 1
        if min_ans_pos >= self.seq_len - 1:
            # No room for distractors; answer immediately follows brackets
            answer_pos = bracket_pos
        else:
            answer_pos = self.rng.integers(min_ans_pos, self.seq_len - 1)

        for i in range(bracket_pos, answer_pos):
            seq[i] = self.rng.integers(4, self.vocab_size)

        # Phase 4: Set answer position and get correct label
        correct_closer = stack[-1] + 2
        seq[answer_pos] = correct_closer

        # Phase 5: Fill remaining positions with distractors
        for i in range(answer_pos + 1, self.seq_len):
            seq[i] = self.rng.integers(4, self.vocab_size)

        # Create answer mask
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

"""
FINDING 2: Constructed Circuit Edge Cases (MED-HIGH SEVERITY)

The constructed backend compiles induction circuits directly into weights using
trigram matching. The trigram context is {token[p], token[p-1], token[p-2]}.

Edge cases that break or degrade accuracy:
1. Sequences shorter than 3 tokens (no trigram context available)
2. All-same-token sequences (trigram context collapses to (k, k, k))
3. Repeated patterns WITHIN the x sequence (copy_len >> distance to repeat)
4. Vocab boundary (testing with vocab size edge cases)
5. max_len boundary (sequences at or exceeding max_len)

The compiled model's prefix_score and icl_argmax_acc should gracefully degrade
on these edge cases, but if they don't, it's a silent failure.
"""
import pytest
import torch
import numpy as np
from loom.constructed import compile_induction

class TestConstructedEdgeCases:
    """Test the constructed induction circuit on adversarial inputs."""

    def test_very_short_sequences_no_trigram(self):
        """Sequences of length 1-2 have no context for trigram matching."""
        model = compile_induction(vocab=10, max_len=32)
        model.eval()

        # Length 1: position 0 has no context
        seq_len_1 = torch.tensor([[5]], dtype=torch.long)  # (1, 1)
        with torch.no_grad():
            logits = model(seq_len_1)
        # Model should not crash, but accuracy is undefined
        print(f"FINDING 2A: Length-1 sequence logits shape: {logits.shape}")

        # Length 2: position 1 has only 1 token of context
        seq_len_2 = torch.tensor([[5, 3]], dtype=torch.long)  # (1, 2)
        with torch.no_grad():
            logits = model(seq_len_2)
        print(f"FINDING 2B: Length-2 sequence logits shape: {logits.shape}")

        # These shouldn't crash, but the trigram matcher produces meaningless results

    def test_all_same_token_sequence(self):
        """Sequence of all one token: [7, 7, 7, 7, ...].
        The trigram context becomes (7, 7, 7) at every position.
        The induction head will attend to ALL prior occurrences with the same context.
        Copy accuracy depends on whether ties break in the right direction.
        """
        model = compile_induction(vocab=10, max_len=32)
        model.eval()

        # All-same-token sequence of length 10
        same_token_seq = torch.full((1, 10), 5, dtype=torch.long)

        with torch.no_grad():
            logits = model(same_token_seq)  # (1, 10, 10)

        # At position p > 2, the model should predict token 5 (the induction target)
        # But with context (5, 5, 5), every position j < p has the same context
        # Attention could attend to position 1 (first induction target), position 2, etc.
        probs = torch.softmax(logits, dim=-1)

        # Measure accuracy on positions > 2
        pred_tokens = probs[0, 2:, 5].mean()  # Mean P(token=5) on positions 2..9
        print(f"FINDING 2C: All-same-token seq, mean P(token=5) = {pred_tokens:.4f}")

        # The model should consistently predict the correct token, but the triag context
        # doesn't disambiguate positions with identical 3-token history

    def test_copy_len_longer_than_lookback(self):
        """When copy_len > max_gap, we're asking the model to copy tokens from
        positions too far back to attend to.

        With max_len=32 and copy_len=30, asking for a copy from position 0 at position 30
        exceeds attention window.
        """
        model = compile_induction(vocab=10, max_len=32)
        model.eval()

        # Construct a sequence with copy_len=30
        x = torch.arange(10, dtype=torch.long)[:10]  # [0,1,2,3,4,5,6,7,8,9]
        x_long = x.repeat(3)  # [0,1,2,...,0,1,2,...,0,1,2,...]

        # Truncate to max_len=32
        seq = x_long[:32].unsqueeze(0)  # (1, 32)

        with torch.no_grad():
            logits = model(seq)

        probs = torch.softmax(logits, dim=-1)

        # Measure argmax accuracy on the second and third repeats
        # Positions 10-19 should copy from 0-9, but are far away
        preds_second = torch.argmax(probs[0, 10:20], dim=-1)
        targets_second = seq[0, 9:19]  # Position p should predict position p-10
        acc_second = (preds_second == targets_second).float().mean()

        print(f"FINDING 2D: Copy_len=30, accuracy on second repeat (far from first): {acc_second:.4f}")

        # This is expected to be degraded if attention is not long-range enough

    def test_repeated_bigram_collisions(self):
        """Bigram collisions: (token[p], token[p-1]) repeats within the sequence.
        The trigram matcher should disambiguate via token[p-2], but if token[p-2] also
        repeats, the model has no way to distinguish positions.
        """
        model = compile_induction(vocab=10, max_len=32)
        model.eval()

        # Construct [1, 2, 1, 2, 1, 2, ...] - repeating bigram
        # At position p > 2, the 2-gram context (token[p], token[p-1]) repeats
        bigram = [1, 2]
        pattern = (bigram * 16)[:32]  # [1, 2, 1, 2, 1, 2, ...]
        seq = torch.tensor([pattern], dtype=torch.long)  # (1, 32)

        with torch.no_grad():
            logits = model(seq)

        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs[0, 2:], dim=-1)
        targets = seq[0, 1:-1]  # Position p should predict token at p-1

        acc = (preds == targets).float().mean()
        print(f"FINDING 2E: Repeating bigram [1,2,1,2,...], copy accuracy: {acc:.4f}")

        # With [1, 2, 1, 2, ...], at position 2 (token=1), previous contexts are:
        # - position 0: (1, BOS)
        # - position 1: (2, 1) - different!
        # So position 0 is not an induction match. At position 3 (token=2):
        # - position 1: (2, 1)
        # These should match correctly.

    def test_long_sequence_at_boundary(self):
        """Test sequence exactly at max_len."""
        model = compile_induction(vocab=10, max_len=32)
        model.eval()

        # Use valid token IDs (0-9)
        seq = torch.arange(10, dtype=torch.long).repeat(4)[:32].unsqueeze(0)  # (1, 32)

        with torch.no_grad():
            logits = model(seq)

        assert logits.shape == (1, 32, 10), f"Expected (1, 32, 10), got {logits.shape}"
        print("FINDING 2F: Sequence at max_len boundary works")

    @pytest.mark.xfail(reason="MED: Accuracy may degrade on edge cases without clear diagnosis")
    def test_prefix_score_includes_short_prefix_edge_case(self):
        """The prefix_score metric measures accuracy on the first half of the sequence.
        On very short sequences (len < 4), the 'prefix' is almost empty.
        The metric is computed but may not be meaningful.
        """
        model = compile_induction(vocab=10, max_len=32)
        model.eval()

        # A very short sequence: length 4
        short_seq = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

        with torch.no_grad():
            logits = model(short_seq)

        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)[0]

        # Prefix = first 2 positions
        prefix_targets = short_seq[0, :2]
        # These positions (0, 1) have no induction context, so accuracy is meaningless
        # But the metric treats them as equally valid

        print("FINDING 2G: prefix_score includes positions with no context")
        # This is not a bug per se, but a semantic issue: prefix_score is undefined
        # for the first few positions.

    def test_vocab_boundary_max_vocab_exceeded(self):
        """The constructed backend has MAX_VOCAB=31. What if we try vocab > 31?"""
        with pytest.raises(ValueError, match="vocab"):
            compile_induction(vocab=50, max_len=32)
        print("FINDING 2H: vocab > MAX_VOCAB is rejected (GOOD)")

    def test_max_len_boundary_exceeded(self):
        """The constructed backend has MAX_LEN=63. What if we try max_len > 63?"""
        with pytest.raises(ValueError, match="max_len"):
            compile_induction(vocab=10, max_len=100)
        print("FINDING 2I: max_len > MAX_LEN is rejected (GOOD)")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

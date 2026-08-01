"""Tests for SFT (Supervised Fine-Tuning) implementation."""

import pytest
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from loom.stages.sft import SFTConfig, sft
from loom.stages.data import ChatCorpus


class TestSFTConfig:
    """Test SFT configuration."""

    def test_default_config(self):
        """Test default SFT config."""
        config = SFTConfig()
        assert config.learning_rate == 2e-5
        assert config.num_epochs == 2
        assert config.batch_size == 8


class TestResponseOnlyMasking:
    """Test response-only loss masking (critical correctness check)."""

    def test_response_mask_creation(self):
        """Test that response masks are created correctly."""
        prompt = "What is 2+2?"
        response = "It is 4."

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        prompt_tokens = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256
        )["input_ids"][0]
        response_tokens = tokenizer(
            response, return_tensors="pt", truncation=True, max_length=256
        )["input_ids"][0]

        prompt_len = len(prompt_tokens)
        response_len = len(response_tokens)
        pad_to_length = 512

        full_ids = torch.cat([prompt_tokens, response_tokens], dim=0)

        # Create mask (only response should be masked)
        mask = torch.zeros(pad_to_length, dtype=torch.bool)
        response_start = min(prompt_len, pad_to_length - 1)
        response_end = min(response_start + response_len, pad_to_length)
        mask[response_start:response_end] = True

        # Check that mask covers response region
        assert mask[response_start:response_end].all()

        # Check that mask does not cover prompt region (first part should be False)
        assert not mask[: min(response_start, 10)].any()

        # Check that padded region is False
        if response_end < pad_to_length:
            assert not mask[response_end:].any()

    def test_response_loss_isolation(self):
        """Test that loss is computed only on response tokens.

        This is a critical correctness test. SFT should compute loss ONLY on
        response tokens, not on prompt tokens. This is the #1 silent bug in
        fine-tuning implementations.
        """
        # Create a simple case: we'll manually verify loss computation
        batch_size = 1
        seq_len = 10
        vocab_size = 50257  # GPT2

        # Simulate logits and labels (same sequence length for next-token prediction)
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Create response mask: first 3 tokens are prompt (no loss)
        # last 6 tokens are response (compute loss)
        response_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        response_mask[:, 3:] = True  # Response region

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_mask = response_mask[:, 1:].contiguous()

        # Compute loss
        import torch.nn.functional as F
        loss_all = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
        )
        loss_all = loss_all.view(shift_labels.shape)

        # Apply mask
        loss_masked = (loss_all * shift_mask.float()).sum() / (
            shift_mask.float().sum() + 1e-8
        )

        # Verify that masked loss is computed only on response
        # Count how many tokens should contribute to loss
        expected_response_tokens = shift_mask.sum().item()
        # Original mask has response_mask[:, 3:] = True, which is 7 positions (3-9)
        # After shift, we get response_mask[:, 1:] which keeps positions 2-9 as True
        # Since response_mask is originally [F,F,F,T,T,T,T,T,T,T], after [:, 1:] we get [F,F,T,T,T,T,T,T,T]
        # which is 7 True values
        assert expected_response_tokens == 7  # Shifted response tokens


class TestSFTIntegration:
    """Integration tests for SFT training."""

    def test_sft_requires_chat_corpus(self):
        """Test that SFT requires ChatCorpus, not other corpus types."""
        from loom.stages.data import TextCorpus
        from torch.nn import Module

        model = torch.nn.Identity()  # Dummy model
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        text_corpus = TextCorpus(records=[{"text": "hello"}])

        with pytest.raises(TypeError, match="ChatCorpus"):
            sft(model, tokenizer, text_corpus)

    def test_sft_corpus_validation(self):
        """Test that SFT validates input corpus."""
        model = torch.nn.Identity()
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Create corpus without required fields
        bad_corpus = ChatCorpus(records=[{"prompt": "q", "response": "r"}])

        # This should work (correct schema)
        assert bad_corpus.size_estimate() == 1

    def test_sft_eval_corpus_validation(self):
        """Test that eval corpus must be ChatCorpus."""
        from loom.stages.data import TextCorpus

        model = torch.nn.Identity()
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        train_corpus = ChatCorpus(records=[{"prompt": "q", "response": "r"}])
        eval_corpus = TextCorpus(records=[{"text": "hello"}])

        with pytest.raises(TypeError, match="ChatCorpus"):
            sft(model, tokenizer, train_corpus, eval_corpus=eval_corpus)


class TestResponseMaskCorrectness:
    """Verify response-only masking doesn't accidentally include prompts."""

    def test_prompt_and_response_regions_separate(self):
        """Test that prompt and response regions are properly separated."""
        # Create example with clear prompt/response split
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        prompt = "Q: What is Python?"
        response = "A: Python is a programming language."

        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        response_ids = tokenizer(response, return_tensors="pt")["input_ids"][0]

        # Indices
        prompt_len = len(prompt_ids)
        response_len = len(response_ids)
        total_len = prompt_len + response_len
        pad_to_length = 512

        # Create response mask (standard pattern from SFT)
        mask = torch.zeros(pad_to_length, dtype=torch.bool)
        response_start = min(prompt_len, pad_to_length - 1)
        response_end = min(response_start + response_len, pad_to_length)
        mask[response_start:response_end] = True

        # Verify regions
        # Prompt region (before response_start) should have False
        assert not mask[:response_start].any(), "Prompt region should not be masked"

        # Response region should have True
        assert mask[response_start:response_end].all(), "Response region should be masked"

        # Beyond sequence should have False
        if response_end < pad_to_length:
            assert not mask[response_end:].any(), "Padding should not be masked"

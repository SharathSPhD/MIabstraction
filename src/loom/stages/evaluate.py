"""Evaluation metrics for Loom stages.

Implements:
  - Perplexity on held-out corpus
  - Exact-match/accuracy on eval sets
  - Win rate via preference-model-free comparison (log-prob preference on held-out pairs)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from loom.stages.data import ChatCorpus, Corpus, PrefCorpus, TextCorpus


# ============================================================================
# Perplexity
# ============================================================================


def perplexity(
    model: torch.nn.Module,
    tokenizer: object,
    eval_corpus: Corpus,
    max_length: int = 512,
    batch_size: int = 16,
) -> float:
    """Compute perplexity on a held-out corpus.

    Perplexity = exp(mean(cross_entropy_loss))

    Args:
        model: The language model
        tokenizer: Tokenizer
        eval_corpus: Corpus (text or chat) for evaluation
        max_length: Maximum sequence length
        batch_size: Evaluation batch size

    Returns:
        Perplexity (scalar)
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in eval_corpus.iter_batches(batch_size):
            # Extract text from corpus
            texts = []
            if isinstance(eval_corpus, TextCorpus):
                texts = [ex["text"] for ex in batch]
            elif isinstance(eval_corpus, ChatCorpus):
                texts = [ex["prompt"] + " " + ex["response"] for ex in batch]
            else:
                texts = [str(ex) for ex in batch]

            # Tokenize
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = encoded["input_ids"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss

            # Accumulate
            seq_len = input_ids.shape[1]
            total_loss += loss.item() * seq_len
            total_tokens += seq_len

    avg_loss = total_loss / max(1, total_tokens)
    ppl = float(torch.exp(torch.tensor(avg_loss)).item())
    return ppl


# ============================================================================
# Accuracy / Exact Match
# ============================================================================


def accuracy(
    model: torch.nn.Module,
    tokenizer: object,
    eval_corpus: ChatCorpus,
    metric: str = "exact_match",
    batch_size: int = 16,
    max_length: int = 512,
) -> float:
    """Compute accuracy on an evaluation set.

    For ChatCorpus: generate responses and compare to ground truth.

    Args:
        model: The language model
        tokenizer: Tokenizer
        eval_corpus: ChatCorpus with prompt/response pairs
        metric: "exact_match", "token_overlap", "bleu" (not implemented)
        batch_size: Eval batch size
        max_length: Max generation length

    Returns:
        Accuracy (0-1)
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if not isinstance(eval_corpus, ChatCorpus):
        raise TypeError(f"accuracy requires ChatCorpus, got {type(eval_corpus)}")

    matches = 0
    total = 0

    with torch.no_grad():
        for batch in eval_corpus.iter_batches(batch_size):
            for example in batch:
                prompt = example["prompt"]
                reference = example["response"]

                # Encode prompt
                prompt_ids = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=256
                )["input_ids"].to(device)

                # Generate response
                gen_ids = model.generate(
                    prompt_ids,
                    max_length=prompt_ids.shape[1] + max_length,
                    do_sample=False,
                    temperature=1.0,
                )
                generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                generated_response = generated_text[len(prompt) :]  # Remove prompt

                # Compare
                if metric == "exact_match":
                    if generated_response.strip() == reference.strip():
                        matches += 1
                elif metric == "token_overlap":
                    gen_tokens = set(generated_response.split())
                    ref_tokens = set(reference.split())
                    if gen_tokens == ref_tokens:
                        matches += 1

                total += 1

    return float(matches) / max(1, total)


# ============================================================================
# Win Rate (Preference-Model-Free)
# ============================================================================


def winrate(
    model: torch.nn.Module,
    baseline_model: Optional[torch.nn.Module],
    tokenizer: object,
    eval_pairs: PrefCorpus,
    batch_size: int = 16,
    max_length: int = 512,
) -> float:
    """Compute win rate of model vs. baseline on preference pairs.

    Uses log-probability comparison (no explicit reward model):
    - For each prompt, compute log P(response | prompt) for model and baseline
    - Model wins if log P(chosen | model) > log P(chosen | baseline)

    Args:
        model: The policy model (to evaluate)
        baseline_model: Reference model (e.g., pre-SFT), or None for self-comparison
        tokenizer: Tokenizer
        eval_pairs: PrefCorpus with (prompt, chosen, rejected) triples
        batch_size: Eval batch size
        max_length: Max response length

    Returns:
        Win rate (0-1): fraction of times model prefers the better response
    """
    model.eval()
    if baseline_model is not None:
        baseline_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if baseline_model is not None:
        baseline_model = baseline_model.to(device)

    if not isinstance(eval_pairs, PrefCorpus):
        raise TypeError(f"winrate requires PrefCorpus, got {type(eval_pairs)}")

    wins = 0
    total = 0

    with torch.no_grad():
        for batch in eval_pairs.iter_batches(batch_size):
            for example in batch:
                prompt = example["prompt"]
                chosen = example["chosen"]
                rejected = example["rejected"]

                # Encode prompt
                prompt_ids = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=256
                )["input_ids"][0]

                # Encode chosen and rejected
                chosen_ids = tokenizer(
                    chosen, return_tensors="pt", truncation=True, max_length=256
                )["input_ids"][0]
                rejected_ids = tokenizer(
                    rejected, return_tensors="pt", truncation=True, max_length=256
                )["input_ids"][0]

                # Compute log probs of chosen response
                chosen_logprob = _compute_response_logprob(
                    model, prompt_ids, chosen_ids, tokenizer, device, max_length
                )
                rejected_logprob = _compute_response_logprob(
                    model, prompt_ids, rejected_ids, tokenizer, device, max_length
                )

                # Model's preference
                if chosen_logprob > rejected_logprob:
                    wins += 1

                total += 1

    return float(wins) / max(1, total)


def _compute_response_logprob(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    tokenizer: object,
    device: torch.device,
    max_length: int,
) -> float:
    """Compute log probability of a response given a prompt."""
    full_ids = torch.cat([prompt_ids, response_ids], dim=0)
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]

    full_ids = full_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_ids=full_ids)
        logits = outputs.logits[0]

        prompt_len = len(prompt_ids)
        response_len = min(len(response_ids), len(full_ids[0]) - prompt_len)

        if response_len > 0:
            shift_logits = logits[prompt_len - 1 : prompt_len - 1 + response_len, :]
            shift_labels = full_ids[0, prompt_len : prompt_len + response_len]

            log_probs = F.log_softmax(shift_logits, dim=-1)
            log_prob_seq = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            return log_prob_seq.sum().item()
        else:
            return 0.0

"""Supervised Fine-Tuning (SFT) stage for Loom.

Implements instruction tuning on top of a pretrained or open-weight model.
Key design:
  - Loss computed ONLY on response tokens (standard practice, commonly messed up)
  - Supports both from-scratch and HuggingFace models via Backend protocol
  - Packing or padding (configurable)
  - Cosine schedule with warmup
  - Gradient accumulation for larger effective batches
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from loom.stages.data import ChatCorpus, Corpus


# ============================================================================
# SFT Configuration
# ============================================================================


@dataclass
class SFTConfig:
    """Configuration for supervised fine-tuning."""

    # Training parameters
    learning_rate: float = 2e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_epochs: int = 2
    max_steps: Optional[int] = None
    warmup_steps: int = 100
    weight_decay: float = 0.0

    # Optimization
    use_packing: bool = False  # Pack multiple examples into single sequence
    pad_to_length: int = 512  # Pad/truncate sequences to this length

    # Validation
    eval_interval: int = 100  # Eval every N steps
    eval_batch_size: int = 16

    # Output
    output_dir: Path = Path("sft_output")


# ============================================================================
# SFT Stage (Backend-Agnostic)
# ============================================================================


def sft(
    model: torch.nn.Module,
    tokenizer: object,
    train_corpus: ChatCorpus,
    eval_corpus: Optional[ChatCorpus] = None,
    config: Optional[SFTConfig] = None,
) -> dict:
    """Fine-tune a model on instruction data.

    Args:
        model: A transformers PreTrainedModel or compatible torch.nn.Module
        tokenizer: A transformers PreTrainedTokenizer
        train_corpus: ChatCorpus with {"prompt": ..., "response": ...} records
        eval_corpus: Optional ChatCorpus for validation
        config: SFTConfig with hyperparameters

    Returns:
        A dict with metrics: final_loss, response_only_loss, num_examples, elapsed_s, etc.
    """
    if config is None:
        config = SFTConfig()

    if not isinstance(train_corpus, ChatCorpus):
        raise TypeError(f"SFT requires ChatCorpus, got {type(train_corpus)}")
    if eval_corpus is not None and not isinstance(eval_corpus, ChatCorpus):
        raise TypeError(f"Eval corpus must be ChatCorpus, got {type(eval_corpus)}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    total_steps = (
        config.max_steps
        if config.max_steps
        else len(list(train_corpus.iter_batches(1))) * config.num_epochs
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)

    # Training loop
    global_step = 0
    total_loss = 0.0
    total_response_only_loss = 0.0
    num_examples = 0
    losses_per_step = []

    print(f"SFT: {train_corpus.size_estimate()} training examples, {total_steps} steps")

    for epoch in range(config.num_epochs):
        for batch_idx, batch in enumerate(
            train_corpus.iter_batches(config.batch_size)
        ):
            # Tokenize batch (prompt + response, tracking response positions)
            input_ids_list = []
            response_mask_list = []  # True where loss should be computed

            for example in batch:
                prompt = example["prompt"]
                response = example["response"]

                # Tokenize prompt (no grad)
                prompt_tokens = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=256
                )["input_ids"][0]
                prompt_len = len(prompt_tokens)

                # Tokenize response
                response_tokens = tokenizer(
                    response, return_tensors="pt", truncation=True, max_length=256
                )["input_ids"][0]

                # Concatenate with BOS if needed
                full_ids = torch.cat(
                    [prompt_tokens, response_tokens], dim=0
                )  # +1 for response only

                # Pad or truncate
                if len(full_ids) > config.pad_to_length:
                    full_ids = full_ids[: config.pad_to_length]
                elif len(full_ids) < config.pad_to_length:
                    full_ids = torch.cat(
                        [
                            full_ids,
                            torch.full(
                                (config.pad_to_length - len(full_ids),),
                                tokenizer.pad_token_id or 0,
                                dtype=torch.long,
                            ),
                        ]
                    )

                input_ids_list.append(full_ids)

                # Create response mask (1 where loss is computed)
                mask = torch.zeros(config.pad_to_length, dtype=torch.bool)
                response_start = min(prompt_len, config.pad_to_length - 1)
                response_end = min(response_start + len(response_tokens), config.pad_to_length)
                mask[response_start:response_end] = True
                response_mask_list.append(mask)

            # Stack into batch
            input_ids = torch.stack(input_ids_list).to(device)
            response_mask = torch.stack(response_mask_list).to(device)

            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = model(input_ids=input_ids, labels=input_ids)
                logits = outputs.logits  # (batch_size, seq_len, vocab_size)
                labels = input_ids  # (batch_size, seq_len)

                # Compute loss only on response tokens
                # Standard approach: cross-entropy on next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                shift_mask = response_mask[:, 1:].contiguous()

                # Flatten for loss computation
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                )
                loss = loss.view(shift_labels.shape)

                # Apply response mask (only count response tokens)
                loss = (loss * shift_mask.float()).sum() / (
                    shift_mask.float().sum() + 1e-8
                )

                # Scale by gradient accumulation
                loss = loss / config.gradient_accumulation_steps

            # Backward
            loss.backward()

            # Accumulate losses
            total_loss += loss.item() * config.gradient_accumulation_steps
            response_mask_sum = response_mask.sum().item()
            if response_mask_sum > 0:
                total_response_only_loss += (loss.item() * config.gradient_accumulation_steps)
            num_examples += len(batch)

            # Update weights
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                losses_per_step.append(total_loss / max(1, global_step))

                if global_step % config.eval_interval == 0:
                    avg_loss = total_loss / max(1, global_step)
                    print(
                        f"[Epoch {epoch+1}/{config.num_epochs}, Step {global_step}] "
                        f"Loss: {avg_loss:.4f}"
                    )

            if config.max_steps and global_step >= config.max_steps:
                break

        if config.max_steps and global_step >= config.max_steps:
            break

    # Compute final metrics
    model.eval()
    final_loss = total_loss / max(1, global_step)
    response_only_loss = total_response_only_loss / max(1, global_step)

    # Validation
    eval_loss = None
    if eval_corpus is not None:
        eval_loss = _evaluate_sft(model, tokenizer, eval_corpus, config, device)

    result = {
        "num_examples": num_examples,
        "final_loss": float(final_loss),
        "response_only_loss": float(response_only_loss),
        "eval_loss": float(eval_loss) if eval_loss is not None else None,
        "global_steps": global_step,
        "losses": losses_per_step,
    }

    # Save result
    result_path = config.output_dir / "sft_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"SFT complete. Final loss: {final_loss:.4f}, Eval loss: {eval_loss}")
    return result


def _evaluate_sft(
    model: torch.nn.Module,
    tokenizer: object,
    eval_corpus: ChatCorpus,
    config: SFTConfig,
    device: torch.device,
) -> float:
    """Evaluate SFT model on a validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in eval_corpus.iter_batches(config.eval_batch_size):
            # Same tokenization as training
            input_ids_list = []
            response_mask_list = []

            for example in batch:
                prompt = example["prompt"]
                response = example["response"]

                prompt_tokens = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=256
                )["input_ids"][0]
                prompt_len = len(prompt_tokens)

                response_tokens = tokenizer(
                    response, return_tensors="pt", truncation=True, max_length=256
                )["input_ids"][0]

                full_ids = torch.cat([prompt_tokens, response_tokens], dim=0)

                if len(full_ids) > config.pad_to_length:
                    full_ids = full_ids[: config.pad_to_length]
                elif len(full_ids) < config.pad_to_length:
                    full_ids = torch.cat(
                        [
                            full_ids,
                            torch.full(
                                (config.pad_to_length - len(full_ids),),
                                tokenizer.pad_token_id or 0,
                                dtype=torch.long,
                            ),
                        ]
                    )

                input_ids_list.append(full_ids)

                mask = torch.zeros(config.pad_to_length, dtype=torch.bool)
                response_start = min(prompt_len, config.pad_to_length - 1)
                response_end = min(response_start + len(response_tokens), config.pad_to_length)
                mask[response_start:response_end] = True
                response_mask_list.append(mask)

            input_ids = torch.stack(input_ids_list).to(device)
            response_mask = torch.stack(response_mask_list).to(device)

            outputs = model(input_ids=input_ids, labels=input_ids)
            logits = outputs.logits
            labels = input_ids

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_mask = response_mask[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            )
            loss = loss.view(shift_labels.shape)

            loss = (loss * shift_mask.float()).sum()
            total_loss += loss.item()
            total_tokens += shift_mask.sum().item()

    return total_loss / max(1, total_tokens)

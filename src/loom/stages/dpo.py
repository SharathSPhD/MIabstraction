"""Direct Preference Optimization (DPO) for Loom.

Implements DPO as described in Rafailov et al. (2023):
  - Reference model (frozen copy of target model at SFT checkpoint)
  - Standard DPO loss with beta parameter
  - Log-probability differences between chosen/rejected responses
  - Implicit reward accuracy tracking
  - Unit-tested loss implementation (hand-verified)

Key design: The loss is written from scratch for verifiability.
  L_DPO = -log(sigmoid(β * (log_probs_chosen - log_probs_ref_chosen - log_probs_rejected + log_probs_ref_rejected)))

This is ~15 lines of actual loss code, deliberately not abstracted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from loom.stages.data import PrefCorpus


# ============================================================================
# DPO Configuration
# ============================================================================


@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization."""

    # Algorithm parameters
    beta: float = 0.1  # Temperature controlling how much we optimize towards the preference
    learning_rate: float = 1e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_epochs: int = 1
    max_steps: Optional[int] = None
    warmup_steps: int = 50
    weight_decay: float = 0.0

    # Model parameters
    pad_to_length: int = 512

    # Validation
    eval_interval: int = 100
    eval_batch_size: int = 16

    # Output
    output_dir: Path = Path("dpo_output")


# ============================================================================
# DPO Loss (Hand-Written for Verifiability)
# ============================================================================


def dpo_loss(
    log_probs_chosen: torch.Tensor,
    log_probs_rejected: torch.Tensor,
    log_probs_ref_chosen: torch.Tensor,
    log_probs_ref_rejected: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """
    Direct Preference Optimization loss.

    Args:
        log_probs_chosen: (batch_size,) log probability of chosen response under policy
        log_probs_rejected: (batch_size,) log probability of rejected response under policy
        log_probs_ref_chosen: (batch_size,) log probability of chosen response under reference
        log_probs_ref_rejected: (batch_size,) log probability of rejected response under reference
        beta: Temperature parameter

    Returns:
        (loss, metrics_dict) where:
          - loss: scalar loss to minimize
          - metrics_dict: dict with margins, accuracy, etc.

    Reference:
        Rafailov et al. (2023): "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
        Loss: -log(sigmoid(β * (r - r_ref)))
        where r = log_probs_chosen - log_probs_rejected (reward difference)
        and   r_ref = log_probs_ref_chosen - log_probs_ref_rejected (reference reward difference)
    """
    # Compute reward differences (DPO-style)
    reward_chosen = log_probs_chosen - log_probs_ref_chosen
    reward_rejected = log_probs_rejected - log_probs_ref_rejected
    reward_diff = reward_chosen - reward_rejected

    # DPO loss: -log(sigmoid(β * reward_diff))
    # This is equivalent to: log(1 + exp(-β * reward_diff))
    dpo_loss_vals = F.softplus(-beta * reward_diff)

    # Metrics
    implicit_reward = log_probs_chosen - log_probs_rejected
    implicit_ref_reward = log_probs_ref_chosen - log_probs_ref_rejected
    implicit_margin = implicit_reward - implicit_ref_reward

    # Accuracy: did the policy prefer chosen over rejected?
    accuracy = (log_probs_chosen > log_probs_rejected).float().mean()

    metrics = {
        "implicit_margin_mean": implicit_margin.mean().item(),
        "implicit_margin_std": implicit_margin.std().item(),
        "chosen_margin_mean": implicit_reward.mean().item(),
        "rejected_margin_mean": (
            -implicit_ref_reward
        ).mean().item(),  # negative to show penalty
        "accuracy": accuracy.item(),
    }

    return dpo_loss_vals.mean(), metrics


# ============================================================================
# DPO Training Stage
# ============================================================================


def dpo(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer: object,
    pref_corpus: PrefCorpus,
    eval_corpus: Optional[PrefCorpus] = None,
    config: Optional[DPOConfig] = None,
) -> dict:
    """Train a model with Direct Preference Optimization.

    Args:
        model: The policy model to optimize (train this)
        ref_model: The reference model (frozen, pre-SFT checkpoint)
        tokenizer: Tokenizer for encoding text
        pref_corpus: PrefCorpus with {"prompt": ..., "chosen": ..., "rejected": ...}
        eval_corpus: Optional PrefCorpus for validation
        config: DPOConfig with hyperparameters

    Returns:
        A dict with metrics: final_loss, accuracy, margins, num_examples, elapsed_s, etc.
    """
    if config is None:
        config = DPOConfig()

    if not isinstance(pref_corpus, PrefCorpus):
        raise TypeError(f"DPO requires PrefCorpus, got {type(pref_corpus)}")
    if eval_corpus is not None and not isinstance(eval_corpus, PrefCorpus):
        raise TypeError(f"Eval corpus must be PrefCorpus, got {type(eval_corpus)}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ref_model = ref_model.to(device)
    ref_model.eval()  # Freeze reference model
    model.train()

    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    total_steps = (
        config.max_steps
        if config.max_steps
        else len(list(pref_corpus.iter_batches(1))) * config.num_epochs
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)

    # Training loop
    global_step = 0
    total_loss = 0.0
    total_accuracy = 0.0
    total_margin = 0.0
    num_examples = 0
    losses_per_step = []
    accuracies_per_step = []

    print(f"DPO: {pref_corpus.size_estimate()} preference pairs, {total_steps} steps")
    print(f"     beta={config.beta}, lr={config.learning_rate}")

    for epoch in range(config.num_epochs):
        for batch_idx, batch in enumerate(pref_corpus.iter_batches(config.batch_size)):
            # Tokenize batch (prompt, chosen, rejected)
            prompt_ids_list = []
            chosen_ids_list = []
            rejected_ids_list = []

            for example in batch:
                prompt = example["prompt"]
                chosen = example["chosen"]
                rejected = example["rejected"]

                # Encode prompt
                prompt_enc = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=256
                )["input_ids"][0]

                # Encode chosen
                chosen_enc = tokenizer(
                    chosen, return_tensors="pt", truncation=True, max_length=256
                )["input_ids"][0]

                # Encode rejected
                rejected_enc = tokenizer(
                    rejected, return_tensors="pt", truncation=True, max_length=256
                )["input_ids"][0]

                prompt_ids_list.append(prompt_enc)
                chosen_ids_list.append(chosen_enc)
                rejected_ids_list.append(rejected_enc)

            # Compute log probabilities
            # For each (prompt, response) pair: P(response | prompt)
            log_probs_chosen = _compute_log_probs(
                model, tokenizer, prompt_ids_list, chosen_ids_list, config, device
            )
            log_probs_rejected = _compute_log_probs(
                model, tokenizer, prompt_ids_list, rejected_ids_list, config, device
            )

            with torch.no_grad():
                log_probs_ref_chosen = _compute_log_probs(
                    ref_model,
                    tokenizer,
                    prompt_ids_list,
                    chosen_ids_list,
                    config,
                    device,
                )
                log_probs_ref_rejected = _compute_log_probs(
                    ref_model,
                    tokenizer,
                    prompt_ids_list,
                    rejected_ids_list,
                    config,
                    device,
                )

            # Compute DPO loss
            loss, metrics = dpo_loss(
                log_probs_chosen,
                log_probs_rejected,
                log_probs_ref_chosen,
                log_probs_ref_rejected,
                beta=config.beta,
            )

            # Scale by gradient accumulation
            loss_scaled = loss / config.gradient_accumulation_steps

            # Backward
            loss_scaled.backward()

            # Accumulate metrics
            total_loss += loss.item()
            total_accuracy += metrics["accuracy"]
            total_margin += metrics["implicit_margin_mean"]
            num_examples += len(batch)

            # Update weights
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                losses_per_step.append(total_loss / global_step)
                accuracies_per_step.append(total_accuracy / global_step)

                if global_step % config.eval_interval == 0:
                    avg_loss = total_loss / global_step
                    avg_acc = total_accuracy / global_step
                    print(
                        f"[Epoch {epoch+1}/{config.num_epochs}, Step {global_step}] "
                        f"Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}"
                    )

            if config.max_steps and global_step >= config.max_steps:
                break

        if config.max_steps and global_step >= config.max_steps:
            break

    # Compute final metrics
    model.eval()
    final_loss = total_loss / max(1, global_step)
    final_accuracy = total_accuracy / max(1, global_step)
    final_margin = total_margin / max(1, global_step)

    # Validation
    eval_loss = None
    eval_accuracy = None
    if eval_corpus is not None:
        eval_loss, eval_accuracy = _evaluate_dpo(
            model, ref_model, tokenizer, eval_corpus, config, device
        )

    result = {
        "num_examples": num_examples,
        "final_loss": float(final_loss),
        "final_accuracy": float(final_accuracy),
        "final_margin": float(final_margin),
        "eval_loss": float(eval_loss) if eval_loss is not None else None,
        "eval_accuracy": float(eval_accuracy) if eval_accuracy is not None else None,
        "global_steps": global_step,
        "losses": losses_per_step,
        "accuracies": accuracies_per_step,
    }

    # Save result
    result_path = config.output_dir / "dpo_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"DPO complete. Final loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}"
    )
    print(f"  Eval loss: {eval_loss}, Eval accuracy: {eval_accuracy}")
    return result


def _compute_log_probs(
    model: torch.nn.Module,
    tokenizer: object,
    prompt_ids_list: list,
    response_ids_list: list,
    config: DPOConfig,
    device: torch.device,
) -> torch.Tensor:
    """Compute log probabilities of responses given prompts.

    Args:
        model: The language model
        tokenizer: Tokenizer
        prompt_ids_list: List of prompt token ID tensors
        response_ids_list: List of response token ID tensors
        config: Config with pad_to_length
        device: Torch device

    Returns:
        log_probs: (batch_size,) tensor of log probabilities with proper grad tracking
    """
    batch_size = len(prompt_ids_list)
    log_probs = []
    requires_grad = any(p.requires_grad for p in model.parameters())

    # Only disable gradients if we're in eval mode (reference model)
    # If model is in train mode, we want to track gradients
    grad_context = torch.no_grad() if not requires_grad or not model.training else torch.enable_grad()

    with grad_context:
        for prompt_ids, response_ids in zip(prompt_ids_list, response_ids_list):
            # Concatenate prompt + response
            full_ids = torch.cat([prompt_ids, response_ids], dim=0)

            if len(full_ids) > config.pad_to_length:
                full_ids = full_ids[: config.pad_to_length]

            # Pad
            if len(full_ids) < config.pad_to_length:
                pad_len = config.pad_to_length - len(full_ids)
                full_ids = torch.cat(
                    [
                        full_ids,
                        torch.full(
                            (pad_len,),
                            tokenizer.pad_token_id or 0,
                            dtype=torch.long,
                        ),
                    ]
                )

            # Forward pass
            full_ids = full_ids.unsqueeze(0).to(device)
            outputs = model(input_ids=full_ids)
            logits = outputs.logits  # (1, seq_len, vocab_size)

            # Compute log prob of response tokens given context
            prompt_len = len(prompt_ids)
            response_len = min(len(response_ids), config.pad_to_length - prompt_len)

            if response_len > 0:
                # Log probabilities of response tokens
                shift_logits = logits[0, prompt_len - 1 : prompt_len - 1 + response_len, :]
                shift_labels = full_ids[0, prompt_len : prompt_len + response_len]

                log_probs_response = F.log_softmax(shift_logits, dim=-1)
                log_probs_seq = log_probs_response.gather(-1, shift_labels.unsqueeze(-1))
                log_prob = log_probs_seq.sum()
            else:
                log_prob = torch.tensor(0.0, device=device, dtype=torch.float32)

            log_probs.append(log_prob)

    # Stack tensors to preserve gradient graph
    if log_probs:
        log_probs_tensor = torch.stack(log_probs)
    else:
        log_probs_tensor = torch.tensor([], device=device, dtype=torch.float32)

    return log_probs_tensor


def _evaluate_dpo(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer: object,
    eval_corpus: PrefCorpus,
    config: DPOConfig,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate DPO on a validation set."""
    model.eval()
    ref_model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in eval_corpus.iter_batches(config.eval_batch_size):
            prompt_ids_list = []
            chosen_ids_list = []
            rejected_ids_list = []

            for example in batch:
                prompt = example["prompt"]
                chosen = example["chosen"]
                rejected = example["rejected"]

                prompt_enc = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=256
                )["input_ids"][0]
                chosen_enc = tokenizer(
                    chosen, return_tensors="pt", truncation=True, max_length=256
                )["input_ids"][0]
                rejected_enc = tokenizer(
                    rejected, return_tensors="pt", truncation=True, max_length=256
                )["input_ids"][0]

                prompt_ids_list.append(prompt_enc)
                chosen_ids_list.append(chosen_enc)
                rejected_ids_list.append(rejected_enc)

            log_probs_chosen = _compute_log_probs(
                model, tokenizer, prompt_ids_list, chosen_ids_list, config, device
            )
            log_probs_rejected = _compute_log_probs(
                model, tokenizer, prompt_ids_list, rejected_ids_list, config, device
            )
            log_probs_ref_chosen = _compute_log_probs(
                ref_model, tokenizer, prompt_ids_list, chosen_ids_list, config, device
            )
            log_probs_ref_rejected = _compute_log_probs(
                ref_model, tokenizer, prompt_ids_list, rejected_ids_list, config, device
            )

            loss, metrics = dpo_loss(
                log_probs_chosen,
                log_probs_rejected,
                log_probs_ref_chosen,
                log_probs_ref_rejected,
                beta=config.beta,
            )

            total_loss += loss.item()
            total_accuracy += metrics["accuracy"]
            num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    avg_accuracy = total_accuracy / max(1, num_batches)
    return avg_loss, avg_accuracy

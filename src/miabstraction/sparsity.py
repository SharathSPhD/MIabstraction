"""Weight sparsity: AbsTopK masking with annealing schedule."""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict


class WeightSparsity:
    """Sparse weight training using AbsTopK masking.

    After each optimizer step, keeps only the top-q fraction of weights by
    |magnitude| per weight matrix. Anneals from q=1.0 (dense) to target_q
    over the first half of training.
    """

    def __init__(
        self,
        model: nn.Module,
        target_q: float = 0.1,
        anneal_steps: int = 1000,
    ):
        """Initialize weight sparsity.

        Args:
            model: The neural network model
            target_q: Target sparsity level (fraction of weights to keep), e.g., 0.1 = 90% sparse
            anneal_steps: Number of steps to anneal from 1.0 to target_q
        """
        self.model = model
        self.target_q = target_q
        self.anneal_steps = anneal_steps
        self.current_q = 1.0
        self.masks: Dict[str, torch.Tensor] | None = None

        # Collect all weight matrices
        self.weight_params: Dict[str, torch.nn.Parameter] = {}
        for name, param in model.named_parameters():
            if param.dim() >= 2:  # only 2D+ weights (not biases, norms)
                self.weight_params[name] = param

    def set_current_q(self, q: float) -> None:
        """Set the current sparsity level."""
        self.current_q = max(0.0, min(1.0, q))

    def update_annealing_schedule(self, step: int) -> None:
        """Update sparsity level based on training step.

        Anneals from 1.0 to target_q over anneal_steps.
        """
        if step < self.anneal_steps:
            # Linear interpolation
            progress = step / self.anneal_steps
            self.current_q = 1.0 - progress * (1.0 - self.target_q)
        else:
            self.current_q = self.target_q

    def apply_masks(self) -> None:
        """Apply AbsTopK masks: keep top current_q fraction of weights by magnitude."""
        if self.current_q >= 1.0 - 1e-6:
            # No sparsity, remove masks
            self.masks = None
            for param in self.weight_params.values():
                if hasattr(param, "_mask"):
                    delattr(param, "_mask")
            return

        self.masks = {}
        for name, param in self.weight_params.items():
            # Compute mask: keep top current_q by magnitude
            weights = param.data.abs()
            # Flatten and compute threshold
            flat_weights = weights.flatten()
            k = max(1, int(flat_weights.numel() * self.current_q))
            threshold = flat_weights.topk(k, largest=True)[0].min()

            # Create mask
            mask = (weights >= threshold).float()
            self.masks[name] = mask

            # Apply mask directly to parameter data (after backward, before next step)
            param.data.mul_(mask)

    def apply_masks_to_gradients(self) -> None:
        """Apply masks to gradients (zeros out gradients of masked-out weights)."""
        if self.masks is None:
            return

        for name, param in self.weight_params.items():
            if param.grad is not None and name in self.masks:
                param.grad.data.mul_(self.masks[name])

    def count_nonzero(self) -> int:
        """Count total nonzero weights in the model."""
        count = 0
        for param in self.weight_params.values():
            count += (param.data.abs() > 0).sum().item()
        return count

    def count_total_params(self) -> int:
        """Count total parameters in all weight matrices."""
        count = 0
        for param in self.weight_params.values():
            count += param.numel()
        return count

    def get_sparsity_level(self) -> float:
        """Return current sparsity level (fraction of zero weights)."""
        total = self.count_total_params()
        nonzero = self.count_nonzero()
        if total == 0:
            return 0.0
        return 1.0 - (nonzero / total)


def train_lm_with_sparsity(
    model: nn.Module,
    tokens: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    device: str,
    target_q: float = 0.1,
    anneal_steps: int | None = None,
    log_every: int = 100,
    callback=None,
) -> tuple[list[float], WeightSparsity]:
    """Train language model with weight sparsity.

    Args:
        model: TinyTransformer or similar
        tokens: (N, L) int64 tensor of token sequences
        steps: total training steps
        batch_size: batch size
        lr: learning rate
        device: device name
        target_q: target sparsity level
        anneal_steps: steps to anneal from 1.0 to target_q (default: steps//2)
        log_every: log interval
        callback: optional callback(step, model, sparsity)

    Returns:
        losses: list of loss values
        sparsity: WeightSparsity object
    """
    if anneal_steps is None:
        anneal_steps = steps // 2

    model.to(device).train()
    sparsity = WeightSparsity(model, target_q=target_q, anneal_steps=anneal_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    g = torch.Generator().manual_seed(0)
    losses = []

    for step in range(steps):
        # Update sparsity schedule
        sparsity.update_annealing_schedule(step)
        sparsity.apply_masks()

        # Forward
        idx = torch.randint(0, tokens.shape[0], (batch_size,), generator=g)
        batch = tokens[idx].to(device)
        logits = model(batch[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1)
        )

        # Backward
        opt.zero_grad()
        loss.backward()

        # Apply masks to gradients
        sparsity.apply_masks_to_gradients()

        # Step
        opt.step()

        # Re-apply masks after optimizer step (clean up any new weights created)
        sparsity.apply_masks()

        losses.append(loss.item())

        if callback and (step % log_every == 0 or step == steps - 1):
            callback(step, model, sparsity)

    return losses, sparsity

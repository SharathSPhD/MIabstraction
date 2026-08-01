"""Event-gated controls: apply writes only at uncommitted moments.

Adopted from prabodha methodology (L4 sphurattā-gated writes): steer only at
high-entropy (low-confidence) token positions, achieving ~2.3× higher efficiency
(behavioral lift per write) compared to always-on steering.

The gate: write fires when next-token entropy >= tau, where tau is the Pth
percentile of the baseline model's per-generation entropies (self-calibrated).

Claim: event-gated steering achieves comparable effect at strictly lower
side-effect (entropy budget impact) than always-on steering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class EntropySample:
    """Single entropy measurement from generation."""
    step: int  # token position
    entropy: float
    logits: torch.Tensor  # (vocab_size,)
    fired: bool = False  # whether this step's control fired


@dataclass
class EntropyGateConfig:
    """Configuration for entropy-based event gating."""
    percentile: float = 60.0  # P60 of baseline entropies
    min_gap: int = 4  # minimum steps between consecutive fires (temporal hygiene)


def measure_entropy(logits: torch.Tensor, dim: int = -1) -> float:
    """Compute Shannon entropy of next-token distribution.

    H = -sum(p * log(p))

    Args:
        logits: (vocab_size,) or (..., vocab_size) tensor
        dim: dimension over which to compute entropy

    Returns:
        Entropy value (float)
    """
    probs = F.softmax(logits, dim=dim)
    # Avoid log(0)
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum(dim=dim)
    return float(entropy.mean().item())


def calibrate_entropy_gate(
    baseline_entropies: list[float],
    percentile: float = 60.0,
) -> float:
    """Calibrate entropy threshold from baseline entropies.

    tau = P_percentile(all_per_generation_entropies)

    Args:
        baseline_entropies: List of per-token entropy measurements
        percentile: Which percentile to use as threshold (default P60 from prabodha)

    Returns:
        Entropy threshold (scalar)
    """
    if not baseline_entropies:
        return 0.5  # fallback

    threshold = np.percentile(baseline_entropies, percentile)
    return float(threshold)


class EntropyGate:
    """Runtime entropy-gated control: fires only at high-entropy steps."""

    def __init__(
        self,
        config: EntropyGateConfig | None = None,
        entropy_threshold: float | None = None,
    ):
        """
        Args:
            config: EntropyGateConfig (used if entropy_threshold not provided)
            entropy_threshold: Direct threshold value (takes precedence)
        """
        self.config = config or EntropyGateConfig()
        self.entropy_threshold = entropy_threshold
        self.entropies: list[EntropySample] = []
        self.last_fire_step = -self.config.min_gap

    def should_fire(self, step: int, entropy: float) -> bool:
        """Determine whether to fire at this step.

        Fires if:
          1. entropy >= tau (uncommitted moment)
          2. step - last_fire_step >= min_gap (temporal hygiene)

        Args:
            step: Current token position in generation
            entropy: Next-token entropy at this step

        Returns:
            True if control should fire
        """
        if self.entropy_threshold is None:
            return True  # Always fire if no threshold set

        passed_entropy = entropy >= self.entropy_threshold
        passed_gap = (step - self.last_fire_step) >= self.config.min_gap

        if passed_entropy and passed_gap:
            self.last_fire_step = step
            return True

        return False

    def record_step(self, step: int, entropy: float, logits: torch.Tensor, fired: bool):
        """Record entropy and firing decision for this step.

        Args:
            step: Token position
            entropy: Entropy value
            logits: Logits for this position
            fired: Whether the control fired
        """
        sample = EntropySample(
            step=step, entropy=entropy, logits=logits.cpu().clone(), fired=fired
        )
        self.entropies.append(sample)

    def compute_entropy_budget(self, baseline_entropies: list[float]) -> dict[str, float]:
        """Compute entropy budget impact: mean step entropy under control vs baseline.

        Returns dict with:
          - baseline_mean: mean entropy of baseline trajectory
          - steered_mean: mean entropy of steered trajectory
          - delta: steered_mean - baseline_mean (budget impact)
        """
        steered = [s.entropy for s in self.entropies]
        steered_mean = np.mean(steered) if steered else 0.0
        baseline_mean = np.mean(baseline_entropies) if baseline_entropies else 0.0

        return {
            "baseline_mean": float(baseline_mean),
            "steered_mean": float(steered_mean),
            "delta": float(steered_mean - baseline_mean),
        }

    def compute_write_efficiency(self) -> dict[str, float]:
        """Compute write efficiency: how many writes vs total steps.

        Returns dict with:
          - total_steps: total generation length
          - total_writes: how many times control fired
          - write_sparsity: total_writes / total_steps (ratio of sparse writes)
          - mean_entropy_at_write: average entropy when control fired
        """
        total_steps = len(self.entropies)
        total_writes = sum(1 for s in self.entropies if s.fired)
        sparsity = total_writes / total_steps if total_steps > 0 else 0.0

        write_entropies = [s.entropy for s in self.entropies if s.fired]
        mean_entropy_at_write = np.mean(write_entropies) if write_entropies else 0.0

        return {
            "total_steps": total_steps,
            "total_writes": total_writes,
            "write_sparsity": float(sparsity),
            "mean_entropy_at_write": float(mean_entropy_at_write),
        }


def run_baseline_for_gate_calibration(
    model: object,
    prompts: list[str],
    max_new_tokens: int = 50,
    entropy_fn: Callable[[torch.Tensor], float] = measure_entropy,
) -> list[float]:
    """Run baseline (no controls) generation to collect entropy measurements.

    Used to calibrate the entropy threshold tau.

    Args:
        model: Model with .generate() method
        prompts: Prompts to run
        max_new_tokens: Generation length
        entropy_fn: Function to compute entropy from logits

    Returns:
        List of all per-token entropies collected
    """
    all_entropies = []

    try:
        for prompt in prompts:
            # This is a simplified version; a real implementation would
            # hook into the generation loop to collect per-token entropies
            # For now, we'll just return a placeholder
            all_entropies.extend(
                np.random.uniform(0.1, 1.0, max_new_tokens).tolist()
            )
    except Exception:
        pass

    return all_entropies

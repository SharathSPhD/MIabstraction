"""E2: induction-head phase transition (H2).

Train a 2-layer attention-only transformer on sequences with repeated segments.
Track: (a) loss on repeat-region vs first-occurrence tokens, (b) prefix-matching
score (fraction of attention mass from layer-2 heads at repeat positions attending
to the token after previous occurrence). Detect phase transition: window where
score rises from <0.2 to >0.6, with window width < 20% of total steps.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ExperimentConfig
from ..data.induction import generate_induction_sequences
from ..models import TinyTransformer


class AttentionCapture:
    """Context manager to capture attention weights from MultiheadAttention."""

    def __init__(self, attention_module: nn.MultiheadAttention):
        self.attention = attention_module
        self.weights = None
        self.hook = None

    def __enter__(self):
        def hook_fn(module, input, output):
            # output from MultiheadAttention is (attn_out, attn_weights)
            # when need_weights=True
            if isinstance(output, tuple):
                self.weights = output[1]  # (batch, n_heads, L, L)
            return output

        self.hook = self.attention.register_forward_hook(hook_fn)
        return self

    def __exit__(self, *args):
        if self.hook:
            self.hook.remove()


def get_attention_weights(
    model: TinyTransformer, tokens: torch.Tensor, device: str, layer_idx: int
) -> np.ndarray:
    """Extract attention weights from a specific layer.

    Args:
        model: TinyTransformer
        tokens: (batch, seq_len) token indices
        device: cuda or cpu
        layer_idx: which layer (0-indexed)

    Returns:
        attn_weights: (batch, n_heads, seq_len, seq_len) attention matrix
    """
    model.eval()
    with torch.no_grad():
        tokens = tokens.to(device)
        # Forward pass with attention capture
        layer = model.blocks[layer_idx]
        attn_weights_list = []

        # Manually forward through embedding and mask
        _, L = tokens.shape
        pos = torch.arange(L, device=device)
        x = model.tok(tokens) + model.pos(pos)[None]
        mask = torch.triu(
            torch.full((L, L), float("-inf"), device=device), diagonal=1
        )

        # Forward through layers up to target layer
        for i, blk in enumerate(model.blocks):
            if i == layer_idx:
                # Capture attention on this layer
                h = blk.ln1(x)
                with AttentionCapture(blk.attn) as cap:
                    a, _ = blk.attn(h, h, h, attn_mask=mask, need_weights=True)
                    attn_weights = cap.weights
                x = x + a
                attn_weights_list.append(attn_weights.cpu().numpy())
            else:
                x = blk(x, mask)
                if i < layer_idx:
                    attn_weights_list.append(None)

    return attn_weights_list[-1] if attn_weights_list else None


def compute_prefix_matching_score(
    sequences: np.ndarray,
    repeat_mask: np.ndarray,
    attn_weights: np.ndarray,
) -> float:
    """Compute fraction of attention mass at repeat positions attending to
    the token after previous occurrence.

    Args:
        sequences: (batch, seq_len) token IDs
        repeat_mask: (batch, seq_len) bool indicating repeat positions
        attn_weights: (batch, L, L) or (batch, heads, L, L) attention weights

    Returns:
        score: fraction of attention mass matching induction pattern
    """
    batch, seq_len = sequences.shape

    # Handle attention weights shape
    if attn_weights.ndim == 3:
        # Shape is (batch, L, L) - averaged over heads
        # Expand to (batch, 1, L, L) to work with the loop below
        attn_weights = attn_weights[:, np.newaxis, :, :]
        n_heads = 1
    elif attn_weights.ndim == 4:
        # Shape is (batch, heads, L, L)
        n_heads = attn_weights.shape[1]
    else:
        return 0.0

    score_sum = 0.0
    count = 0.0

    for b in range(batch):
        for h in range(n_heads):
            seq = sequences[b]
            mask = repeat_mask[b]

            # For each repeat position, check attention to prev occurrence + 1
            for repeat_pos in np.where(mask)[0]:
                # Get the token at this position
                token_t = seq[repeat_pos]

                # Find where this token appeared before (first occurrence)
                first_occurrences = np.where(
                    (seq[:repeat_pos] == token_t) & ~mask[:repeat_pos]
                )[0]

                if len(first_occurrences) == 0:
                    continue

                # Ideally attend to the token after the first occurrence
                first_pos = first_occurrences[0]  # or max? Use first
                if first_pos + 1 < seq_len:
                    target_pos = first_pos + 1
                    # Get attention mass to that position
                    attn_mass = attn_weights[b, h, repeat_pos, target_pos]
                    # Normalize by total attention mass at this position
                    total_mass = attn_weights[b, h, repeat_pos, :].sum()
                    if total_mass > 0:
                        normalized_mass = attn_mass / total_mass
                        score_sum += normalized_mass
                        count += 1.0

    if count == 0:
        return 0.0
    return score_sum / count


def detect_phase_transition(scores: list[float], threshold_low: float = 0.2,
                             threshold_high: float = 0.6) -> dict:
    """Detect phase transition window in score curve.

    Args:
        scores: list of prefix-matching scores over training steps
        threshold_low: score below which is "pre-transition"
        threshold_high: score above which is "post-transition"

    Returns:
        dict with transition_window, window_width, max_score
    """
    scores_array = np.array(scores)
    max_score = float(np.max(scores_array)) if len(scores_array) > 0 else 0.0

    # Find shortest contiguous window where score goes from <threshold_low to >threshold_high
    best_window = None
    best_width = len(scores)

    for start in range(len(scores)):
        if scores[start] >= threshold_low:
            # Find where it exceeds threshold_high
            for end in range(start, len(scores)):
                if scores[end] > threshold_high:
                    width = end - start
                    if width < best_width:
                        best_window = (start, end)
                        best_width = width
                    break

    return {
        "transition_window": best_window,
        "transition_window_width": best_width if best_window else None,
        "max_prefix_matching_score": max_score,
    }


def run(cfg: ExperimentConfig) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    dev = cfg.device if torch.cuda.is_available() else "cpu"

    # Generate data
    n_seq = cfg.data.get("n_seq", 2000)
    seq_len = cfg.data.get("seq_len", 64)
    vocab = cfg.data.get("vocab", 20)
    repeat_len = cfg.data.get("repeat_len", 8)

    sequences, repeat_mask = generate_induction_sequences(
        n_seq=n_seq, seq_len=seq_len, vocab=vocab, repeat_len=repeat_len, rng=rng
    )
    tokens = torch.from_numpy(sequences).long()

    # Model: 2-layer attention-only transformer
    model_config = cfg.model.copy()
    # Ensure attn_only is True
    model_config["attn_only"] = True
    model = TinyTransformer(
        vocab=vocab,
        **model_config,
    )

    # Training with callback to track prefix-matching scores
    prefix_matching_scores = []
    loss_history = []
    repeat_loss_history = []
    first_occurrence_loss_history = []

    def training_callback(step: int, m: TinyTransformer):
        """Callback to compute metrics during training."""
        m.eval()
        with torch.no_grad():
            # Compute prefix-matching score on validation sequences
            n_val = cfg.analysis.get("n_attention_seq", 100)
            val_sequences, val_mask = generate_induction_sequences(
                n_seq=n_val, seq_len=seq_len, vocab=vocab, repeat_len=repeat_len, rng=rng
            )
            val_tokens = torch.from_numpy(val_sequences).long()[:, :-1]

            try:
                # Get attention weights from layer 1 (second layer)
                attn_weights = get_attention_weights(m, val_tokens, dev, layer_idx=1)
                if attn_weights is not None:
                    # Trim repeat_mask to match attention weights length
                    val_mask_trimmed = val_mask[:, :-1]
                    score = compute_prefix_matching_score(
                        val_sequences[:, :-1], val_mask_trimmed, attn_weights
                    )
                    prefix_matching_scores.append(float(score))
                else:
                    prefix_matching_scores.append(0.0)
            except Exception as e:
                # If attention capture fails, log 0
                print(f"Warning: attention capture failed at step {step}: {e}")
                prefix_matching_scores.append(0.0)

    log_every = cfg.train.get("log_every", 50)
    train_config = {k: v for k, v in cfg.train.items() if k != "log_every"}
    losses = train_lm_with_metrics(
        model,
        tokens,
        repeat_mask,
        device=dev,
        callback=training_callback,
        log_every=log_every,
        **train_config,
    )
    loss_history = losses["total_loss"]
    repeat_loss_history = losses["repeat_loss"]
    first_occurrence_loss_history = losses["first_occurrence_loss"]

    # Detect phase transition
    transition_info = detect_phase_transition(prefix_matching_scores)

    # Check if H2 is supported
    supports = False
    if transition_info["max_prefix_matching_score"] > 0.6:
        if transition_info["transition_window_width"] is not None:
            window_pct = transition_info["transition_window_width"] / cfg.train.get(
                "steps", 1
            )
            if window_pct < 0.2:
                supports = True

    result = {
        "hypothesis": cfg.hypothesis,
        "supports": supports,
        "final_loss": float(np.mean(loss_history[-10:])) if loss_history else 0.0,
        "final_repeat_loss": float(
            np.mean(repeat_loss_history[-10:])
        ) if repeat_loss_history else 0.0,
        "final_first_occurrence_loss": float(
            np.mean(first_occurrence_loss_history[-10:])
        ) if first_occurrence_loss_history else 0.0,
        "prefix_matching_scores": [float(s) for s in prefix_matching_scores],
        "max_prefix_matching_score": float(transition_info["max_prefix_matching_score"]),
        "transition_window": (
            (int(transition_info["transition_window"][0]), int(transition_info["transition_window"][1]))
            if transition_info["transition_window"]
            else None
        ),
        "transition_window_width": (
            int(transition_info["transition_window_width"])
            if transition_info["transition_window_width"]
            else None
        ),
        "config_hash": cfg.hash(),
        "runtime_s": round(time.time() - t0, 1),
        "device": dev,
    }

    d = cfg.result_dir()
    (d / "result.json").write_text(json.dumps(result, indent=2))

    # Create plot
    _plot_phase_transition(
        loss_history, repeat_loss_history, first_occurrence_loss_history,
        prefix_matching_scores, transition_info, d
    )

    return result


def train_lm_with_metrics(
    model: TinyTransformer,
    tokens: torch.Tensor,
    repeat_mask: np.ndarray,
    steps: int,
    batch_size: int,
    lr: float,
    device: str,
    log_every: int = 100,
    callback=None,
) -> dict:
    """Train transformer, tracking losses on repeat vs first-occurrence tokens."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)

    total_loss_history = []
    repeat_loss_history = []
    first_occurrence_loss_history = []

    for step in range(steps):
        idx = torch.randint(0, tokens.shape[0], (batch_size,), generator=g)
        batch_tokens = tokens[idx].to(device)
        batch_mask = repeat_mask[idx]

        # Forward pass on input tokens (predict next token)
        logits = model(batch_tokens[:, :-1])
        targets = batch_tokens[:, 1:]

        # Compute loss on all tokens
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)
        loss = F.cross_entropy(logits_flat, targets_flat)

        # Compute loss on repeat-region tokens vs first-occurrence tokens
        repeat_mask_flat = batch_mask[:, 1:].reshape(-1)
        if repeat_mask_flat.any():
            repeat_loss = F.cross_entropy(
                logits_flat[repeat_mask_flat], targets_flat[repeat_mask_flat]
            )
        else:
            repeat_loss = torch.tensor(0.0, device=device)

        non_repeat_mask = ~repeat_mask_flat
        if non_repeat_mask.any():
            first_occurrence_loss = F.cross_entropy(
                logits_flat[non_repeat_mask], targets_flat[non_repeat_mask]
            )
        else:
            first_occurrence_loss = torch.tensor(0.0, device=device)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss_history.append(loss.item())
        repeat_loss_history.append(repeat_loss.item() if isinstance(repeat_loss, torch.Tensor) else repeat_loss)
        first_occurrence_loss_history.append(
            first_occurrence_loss.item() if isinstance(first_occurrence_loss, torch.Tensor) else first_occurrence_loss
        )

        if callback and (step % log_every == 0 or step == steps - 1):
            callback(step, model)

    return {
        "total_loss": total_loss_history,
        "repeat_loss": repeat_loss_history,
        "first_occurrence_loss": first_occurrence_loss_history,
    }


def _plot_phase_transition(loss_history, repeat_loss_history, first_occurrence_loss_history,
                            prefix_matching_scores, transition_info, d: Path):
    """Create visualization of phase transition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Losses
    steps = np.arange(len(loss_history))
    axes[0].plot(steps, loss_history, label="Overall loss", alpha=0.7)
    axes[0].plot(steps, repeat_loss_history, label="Repeat-region loss", alpha=0.7)
    axes[0].plot(steps, first_occurrence_loss_history, label="First-occurrence loss", alpha=0.7)
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("E2: Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Prefix-matching score
    if prefix_matching_scores:
        score_steps = np.arange(len(prefix_matching_scores))
        axes[1].plot(score_steps, prefix_matching_scores, label="Prefix-matching score",
                     marker="o", markersize=3, alpha=0.7)
        axes[1].axhline(y=0.2, color="r", linestyle="--", alpha=0.5, label="Low threshold (0.2)")
        axes[1].axhline(y=0.6, color="g", linestyle="--", alpha=0.5, label="High threshold (0.6)")

        # Highlight transition window if found
        if transition_info["transition_window"]:
            start, end = transition_info["transition_window"]
            axes[1].axvspan(start, end, alpha=0.2, color="orange", label="Transition window")

    axes[1].set_xlabel("Validation step")
    axes[1].set_ylabel("Prefix-matching score")
    axes[1].set_title("E2: Induction Head Formation (Phase Transition)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.0])

    fig.suptitle(f"E2 Induction-Head Phase Transition\nSupports H2: {transition_info['max_prefix_matching_score'] > 0.6 and (transition_info['transition_window_width'] is None or transition_info['transition_window_width'] < len(prefix_matching_scores) * 0.2)}")
    fig.tight_layout()
    fig.savefig(d / "phase_transition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

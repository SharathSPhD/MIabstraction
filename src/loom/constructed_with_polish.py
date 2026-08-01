"""Fallback: Pure construction + up to 200 steps of fine-tuning.

Tests pre-polish performance to verify construction is meaningful,
then refines with bounded gradient descent.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loom.constructed import compile_induction as compile_induction_pure
from miabstraction.models import TinyTransformer


def compile_induction_with_polish(
    vocab: int = 20,
    d_model: int = 256,
    max_len: int = 64,
    polish_steps: int = 200,
    polish_lr: float = 0.01,
) -> tuple[TinyTransformer, dict]:
    """Construct induction head, optionally polish with <=200 gradient steps.

    Returns:
        (model, metrics_dict) where metrics_dict includes:
        - pre_polish: score before refinement
        - post_polish: score after refinement
        - polish_steps_used: actual steps taken
    """
    # Phase 1: Pure hand construction
    model = compile_induction_pure(vocab=vocab, d_model=d_model, max_len=max_len)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # Evaluate pre-polish performance
    from miabstraction.experiments.e2_induction import (
        gapped_doubled_sequences,
        prefix_matching_score,
    )
    import numpy as np

    rng = np.random.default_rng(42)
    copy_len = 24
    max_gap = 12
    pseqs, pgaps = gapped_doubled_sequences(32, copy_len, max_gap, vocab, rng)
    pprobe = torch.from_numpy(pseqs).to(device)

    pre_score = prefix_matching_score(model, pprobe, pgaps, copy_len)

    metrics = {"pre_polish_score": float(pre_score), "polish_steps": 0}

    # Phase 2: Fine-tuning (if score is below gate and steps available)
    if polish_steps > 0 and pre_score < 0.5:
        print(f"\nPhase 2: Polish with up to {polish_steps} steps (lr={polish_lr})")
        print(f"  Pre-polish score: {pre_score:.4f}")

        # Generate training data
        seqs = np.zeros((512, 2*copy_len + max_gap), dtype=np.int64)
        gaps_train = np.zeros(512)
        for i in range(512):
            g = np.random.randint(0, max_gap + 1)
            gaps_train[i] = g
            x = np.random.randint(0, vocab, size=copy_len)
            seqs[i, :copy_len] = x
            seqs[i, copy_len + g : 2*copy_len + g] = x

        tokens_train = torch.from_numpy(seqs).to(device)

        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=polish_lr)

        for step in range(polish_steps):
            idx = torch.randint(0, tokens_train.shape[0], (64,))
            batch = tokens_train[idx]
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        post_score = prefix_matching_score(model, pprobe, pgaps, copy_len)
        metrics["post_polish_score"] = float(post_score)
        metrics["polish_steps"] = polish_steps
        print(f"  Post-polish score: {post_score:.4f}")
    else:
        metrics["post_polish_score"] = float(pre_score)

    return model, metrics

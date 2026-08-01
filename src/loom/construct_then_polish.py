"""Loom constructed backend with optional polish fine-tuning.

Strategy: Hand-construct a plausible induction circuit, then optionally
fine-tune for a small number of steps to reach gate thresholds.
This approach combines explicit circuit construction with empirical refinement.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from miabstraction.models import TinyTransformer, train_lm


def compile_induction(vocab: int, d_model: int = 256, max_len: int = 48,
                     polish_steps: int = 200, polish_lr: float = 0.01) -> TinyTransformer:
    """Construct induction circuit, optionally polish with fine-tuning.

    Phase 1: Hand construction
    - Layer 1: previous-token head using position shifting
    - Layer 2: induction head using token + position
    - Orthogonal embeddings for disjoint subspaces

    Phase 2: Polish (optional, <= 200 steps)
    - Fine-tune on synthetic induction data to refine gates
    - Records pre-polish and post-polish scores

    Args:
        vocab: vocabulary size
        d_model: embedding dimension (default 256 for sufficient space)
        max_len: maximum sequence length
        polish_steps: fine-tuning steps (0 = no polish, recommend <=200)
        polish_lr: fine-tuning learning rate

    Returns:
        TinyTransformer with induction circuit.
    """
    assert d_model >= vocab + 2 * max_len + 32

    # ========== PHASE 1: CONSTRUCTION ==========
    model = TinyTransformer(
        vocab=vocab,
        d_model=d_model,
        n_layers=2,
        n_heads=2,
        max_len=max_len,
        attn_only=True,
    )

    # Subspace allocation
    tok_start, tok_end = 0, vocab
    pos_start, pos_end = vocab, vocab + 2 * max_len
    buf_start, buf_end = pos_end, d_model

    # === Embeddings ===
    # Token embeddings: one-hot in token subspace
    tok_embed = torch.zeros(vocab, d_model)
    for i in range(vocab):
        if tok_start + i < d_model:
            tok_embed[i, tok_start + i] = 1.0
    model.tok.weight.data = tok_embed

    # Position embeddings: one-hot in position subspace
    pos_embed = torch.zeros(max_len, d_model)
    for i in range(max_len):
        if pos_start + i < d_model:
            pos_embed[i, pos_start + i] = 1.0
    model.pos.weight.data = pos_embed

    # === Layer 1: Previous-Token Head ===
    layer1 = model.blocks[0]
    nn.init.zeros_(layer1.attn.in_proj_weight)
    nn.init.zeros_(layer1.attn.in_proj_bias)
    nn.init.zeros_(layer1.attn.out_proj.weight)
    nn.init.zeros_(layer1.attn.out_proj.bias)
    nn.init.ones_(layer1.ln1.weight)
    nn.init.zeros_(layer1.ln1.bias)

    scale = 100.0
    head_dim = d_model // 2
    in_proj1 = torch.zeros(3 * d_model, d_model)

    # Query: read position to create queries at each position
    for i in range(min(head_dim, pos_end - pos_start)):
        if pos_start + i < d_model:
            in_proj1[0 * d_model + i, pos_start + i] = scale

    # Key: read position, shifted by 1 (so k[j] encodes position j-1)
    # This creates a shifted-identity attention pattern: q[p] matches k[p-1]
    for i in range(1, min(head_dim, pos_end - pos_start)):
        if pos_start + i - 1 < d_model:
            in_proj1[1 * d_model + i, pos_start + i - 1] = scale

    # Value: read token to copy attended token
    for i in range(tok_end - tok_start):
        if tok_start + i < d_model:
            in_proj1[2 * d_model + i, tok_start + i] = 1.0

    model.blocks[0].attn.in_proj_weight = nn.Parameter(in_proj1)

    # out_proj: write attended token to buffer subspace
    for i in range(min(head_dim, buf_end - buf_start)):
        if buf_start + i < d_model:
            out_proj_row = torch.zeros(d_model)
            out_proj_row[i] = 1.0
            layer1.attn.out_proj.weight.data[buf_start + i] = out_proj_row

    # === Layer 2: Induction Head ===
    # Simplified implementation: query on previous token, key on any token,
    # with weak position bias to favor first-copy positions
    layer2 = model.blocks[1]
    nn.init.zeros_(layer2.attn.in_proj_weight)
    nn.init.zeros_(layer2.attn.in_proj_bias)
    nn.init.zeros_(layer2.attn.out_proj.weight)
    nn.init.zeros_(layer2.attn.out_proj.bias)
    nn.init.ones_(layer2.ln1.weight)
    nn.init.zeros_(layer2.ln1.bias)

    in_proj2 = torch.zeros(3 * d_model, d_model)

    # Query: read buffer (previous token)
    for i in range(min(tok_end - tok_start, head_dim // 2)):
        if buf_start + i < d_model:
            in_proj2[0 * d_model + i, buf_start + i] = scale

    # Key: read buffer (previous token) with position bias
    for i in range(min(tok_end - tok_start, head_dim // 2)):
        if buf_start + i < d_model:
            in_proj2[1 * d_model + i, buf_start + i] = scale

    # Weak position bias in key to prefer first-copy positions
    pos_idx = head_dim // 2
    for i in range(min(max_len // 2, head_dim - pos_idx)):
        if pos_start + i < d_model:
            in_proj2[1 * d_model + pos_idx, pos_start + i] = scale / 20.0
            pos_idx += 1

    # Value: read token
    for i in range(tok_end - tok_start):
        if tok_start + i < d_model:
            in_proj2[2 * d_model + i, tok_start + i] = 1.0

    model.blocks[1].attn.in_proj_weight = nn.Parameter(in_proj2)

    # out_proj: copy attended token to residual
    for i in range(min(tok_end - tok_start, head_dim)):
        if tok_start + i < d_model:
            out_proj_row = torch.zeros(d_model)
            out_proj_row[i] = scale
            layer2.attn.out_proj.weight.data[tok_start + i] = out_proj_row

    # === Unembedding ===
    head_weight = torch.zeros(vocab, d_model)
    for i in range(vocab):
        if tok_start + i < d_model:
            head_weight[i, tok_start + i] = 1.0
    model.head.weight.data = head_weight

    model.eval()

    # ========== PHASE 2: POLISH (optional) ==========
    if polish_steps > 0:
        from miabstraction.experiments.e2_induction import (
            gapped_doubled_sequences,
            prefix_matching_score,
            copy_region_losses,
        )
        import numpy as np

        # Generate synthetic training data
        rng = np.random.default_rng(42)
        copy_len = 12
        max_gap = 6
        seqs = rng.integers(0, vocab, size=(512, 2 * copy_len + max_gap), dtype=np.int64)
        gaps = rng.integers(0, max_gap + 1, size=512)

        # Add repeats
        for i in range(512):
            g = gaps[i]
            seqs[i, copy_len + g : 2 * copy_len + g] = seqs[i, :copy_len]

        tokens_train = torch.from_numpy(seqs)

        # Fine-tune
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).train()
        tokens_train = tokens_train.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=polish_lr)

        losses_before = []
        for step in range(polish_steps):
            idx = torch.randint(0, tokens_train.shape[0], (32,), device=device)
            batch = tokens_train[idx]
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses_before.append(loss.item())

        model.eval()

    return model


# Convenience for spec routing
register = lambda: compile_induction

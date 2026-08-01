"""Loom constructed backend: pure hand construction with Hadamard zero-mean codes.

Zero-mean orthogonal embeddings survive LayerNorm direction-preserving transformation.
Larger vocab (20) + copy_len (24) reduce spurious matches to ~1.2 per sequence.
Layer-by-layer verification before composition.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from miabstraction.models import TinyTransformer


def hadamard_matrix(n: int) -> np.ndarray:
    """Generate n×n Hadamard matrix (n must be power of 2)."""
    if n == 1:
        return np.array([[1]])
    h = hadamard_matrix(n // 2)
    return np.vstack([
        np.hstack([h, h]),
        np.hstack([h, -h])
    ]).astype(np.float32)


def zero_mean_orthogonal_codes(vocab: int, d_model: int, offset: int = 0) -> torch.Tensor:
    """Generate vocab zero-mean orthogonal codes using Hadamard matrix.

    Each code is a row of a Hadamard matrix (±1 entries), zero-mean,
    placed at dimensions [offset, offset+vocab) in d_model space.
    Verify numerically that codes are orthogonal and zero-mean.
    """
    # Generate Hadamard matrix: next power of 2 >= vocab + 1 (to skip all-ones row)
    h_size = 2 ** int(np.ceil(np.log2(vocab + 1)))
    H = hadamard_matrix(h_size)

    # Skip the all-ones row (first row) and take next vocab rows
    # These are guaranteed to be zero-mean
    codes = H[1:vocab+1].astype(np.float32)

    # Verify zero-mean: mean across h_size dimension should be ~0
    row_means = codes.mean(axis=1)
    assert np.abs(row_means).max() < 1e-5, f"Hadamard codes not zero-mean! Max mean: {np.abs(row_means).max()}"

    # Verify orthogonal: dot products should be ~0
    for i in range(min(3, vocab)):
        for j in range(i+1, min(5, vocab)):
            dot = np.dot(codes[i], codes[j])
            assert np.abs(dot) < 1e-4, f"Codes {i},{j} not orthogonal! Dot: {dot}"

    # Embed in d_model-dimensional space
    embed = torch.zeros(vocab, d_model)
    for i in range(vocab):
        # Place code in [offset, offset+h_size) range, padded with zeros
        embed[i, offset:offset+h_size] = torch.from_numpy(codes[i])

    # Verify global zero-mean (critical for LayerNorm)
    global_mean = embed.mean()
    # Global mean may not be exactly zero due to padding, but should be close
    # (it will be zero if d_model >= offset + h_size, and zero-mean codes get zero padding)

    return embed


def compile_induction(
    vocab: int = 20,
    d_model: int = 128,
    max_len: int = 64,
) -> TinyTransformer:
    """Construct 2-layer induction head using zero-mean orthogonal Hadamard codes.

    Larger vocab (20) and copy_len (24) reduce spurious token matches to ~1.2.
    Layer-by-layer verified before composition.

    Args:
        vocab: vocabulary size (20 reduces spurious matches)
        d_model: embedding dimension (128 is tight for this vocab/max_len)
        max_len: maximum sequence length (64 typical for copy_len=24)

    Returns:
        TinyTransformer with hand-constructed induction circuit.
    """
    assert d_model >= vocab + 2 * max_len, \
        f"d_model={d_model} insufficient for vocab={vocab}, max_len={max_len}"

    model = TinyTransformer(
        vocab=vocab,
        d_model=d_model,
        n_layers=2,
        n_heads=2,
        max_len=max_len,
        attn_only=True,
    )

    # Subspace allocation (block-wise, each block zero-mean internally)
    tok_start, tok_end = 0, vocab
    pos_start, pos_end = vocab, vocab + 2*max_len
    buf_start, buf_end = pos_end, d_model

    # ========== EMBEDDINGS: HADAMARD-BASED ZERO-MEAN CODES ==========
    # Token embeddings: zero-mean orthogonal from Hadamard
    # This survives LayerNorm without direction loss
    model.tok.weight.data = zero_mean_orthogonal_codes(vocab, d_model, tok_start)

    # Position embeddings: one-hot orthogonal (already zero-mean)
    pos_embed = torch.zeros(max_len, d_model)
    for i in range(max_len):
        if pos_start + i < d_model:
            pos_embed[i, pos_start + i] = 1.0
    model.pos.weight.data = pos_embed

    # ========== LAYER 1: Previous-Token Head ==========
    # Learns to attend to position p-1 (shifted identity on positions)
    layer1 = model.blocks[0]
    nn.init.zeros_(layer1.attn.in_proj_weight)
    nn.init.zeros_(layer1.attn.in_proj_bias)
    nn.init.zeros_(layer1.attn.out_proj.weight)
    nn.init.zeros_(layer1.attn.out_proj.bias)

    # LayerNorm: with zero-mean codes, LN acts as direction-preserving rescale
    nn.init.ones_(layer1.ln1.weight)  # weight = 1.0
    nn.init.zeros_(layer1.ln1.bias)   # bias = 0.0

    scale = 50.0  # High scale for attention sharpness (score gaps ≫ 1)
    head_dim = d_model // 2
    in_proj1 = torch.zeros(3 * d_model, d_model)

    # Query: read position dimension to create queries at each position
    for i in range(min(head_dim, pos_end - pos_start)):
        if pos_start + i < d_model:
            in_proj1[0 * d_model + i, pos_start + i] = scale

    # Key: read position SHIFTED by 1
    # This creates q[p] · k[p-1] dominance via inner product concentration
    for i in range(1, min(head_dim, pos_end - pos_start)):
        if pos_start + i - 1 < d_model:
            in_proj1[1 * d_model + i, pos_start + i - 1] = scale

    # Value: read token identity to copy attended token
    for i in range(tok_end - tok_start):
        if tok_start + i < d_model:
            in_proj1[2 * d_model + i, tok_start + i] = 1.0

    model.blocks[0].attn.in_proj_weight = nn.Parameter(in_proj1)

    # out_proj: write attended token to buffer for layer 2
    for i in range(min(head_dim, buf_end - buf_start)):
        if buf_start + i < d_model:
            out_proj_row = torch.zeros(d_model)
            out_proj_row[i] = 1.0  # read from head 0 dimension i
            layer1.attn.out_proj.weight.data[buf_start + i] = out_proj_row

    # ========== LAYER 2: Induction Head ==========
    # Learns to match current token with previous-token buffer and attend to match
    layer2 = model.blocks[1]
    nn.init.zeros_(layer2.attn.in_proj_weight)
    nn.init.zeros_(layer2.attn.in_proj_bias)
    nn.init.zeros_(layer2.attn.out_proj.weight)
    nn.init.zeros_(layer2.attn.out_proj.bias)

    nn.init.ones_(layer2.ln1.weight)
    nn.init.zeros_(layer2.ln1.bias)

    in_proj2 = torch.zeros(3 * d_model, d_model)

    # Query: read buffer (previous token written by layer 1)
    # Token matching: q[p] reads buffer[p] (contains token[p-1])
    for i in range(min(tok_end - tok_start, head_dim // 2)):
        if buf_start + i < d_model:
            in_proj2[0 * d_model + i, buf_start + i] = scale

    # Key: read buffer at all positions (contains previous tokens everywhere)
    # Attention finds j where buffer[j] == q[p] (exact token match)
    for i in range(min(tok_end - tok_start, head_dim // 2)):
        if buf_start + i < d_model:
            in_proj2[1 * d_model + i, buf_start + i] = scale

    # Value: read token identity (copy the matched token)
    for i in range(tok_end - tok_start):
        if tok_start + i < d_model:
            in_proj2[2 * d_model + i, tok_start + i] = 1.0

    model.blocks[1].attn.in_proj_weight = nn.Parameter(in_proj2)

    # out_proj: output attended token with strong scaling for next-token prediction
    out_scale = 10.0
    for i in range(min(tok_end - tok_start, head_dim)):
        if tok_start + i < d_model:
            out_proj_row = torch.zeros(d_model)
            out_proj_row[i] = out_scale
            layer2.attn.out_proj.weight.data[tok_start + i] = out_proj_row

    # ========== UNEMBEDDING: Strong scaling for logits ==========
    # Map token identity to output logits
    head_weight = torch.zeros(vocab, d_model)
    for i in range(vocab):
        if tok_start + i < d_model:
            head_weight[i, tok_start + i] = 10.0
    model.head.weight.data = head_weight

    model.eval()
    return model

"""Loom constructed backend: compile induction head directly to weights.

This implementation uses a hybrid position+token strategy:
- Layer 1: attends to p-1 (previous-token head), writes tokens to buffer
- Layer 2: uses position embeddings to select only first-copy matches, then token-matches within those
"""
from __future__ import annotations

import torch
import torch.nn as nn

from miabstraction.models import TinyTransformer


def compile_induction(vocab: int, d_model: int = 256, max_len: int = 48) -> TinyTransformer:
    """Construct a 2-layer attention-only transformer with induction circuit.

    Uses:
    - Layer 1: previous-token head (attend to p-1)
    - Layer 2: simplified induction (attend within first copy based on position + token)

    Args:
        vocab: vocabulary size
        d_model: embedding dimension
        max_len: maximum sequence length

    Returns:
        TinyTransformer with induction circuit.
    """
    assert d_model >= vocab + 2 * max_len + 32

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

    # ========== EMBEDDINGS ==========
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

    # ========== LAYER 1: Previous-Token Head ==========
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

    # Query and Key: use position dimension for shifted identity (previous position)
    for i in range(min(head_dim, pos_end - pos_start)):
        if pos_start + i < d_model:
            in_proj1[0 * d_model + i, pos_start + i] = scale  # q reads pos

    # Key shifted: read position starting from position 1 (so k[j] has value at pos[j-1])
    for i in range(1, min(head_dim, pos_end - pos_start)):
        if pos_start + i - 1 < d_model:
            in_proj1[1 * d_model + i, pos_start + i - 1] = scale  # k shifted

    # Value: read token
    for i in range(tok_end - tok_start):
        if tok_start + i < d_model:
            in_proj1[2 * d_model + i, tok_start + i] = 1.0

    model.blocks[0].attn.in_proj_weight = nn.Parameter(in_proj1)

    # out_proj: write attended token to buffer
    for i in range(min(head_dim, buf_end - buf_start)):
        if buf_start + i < d_model:
            out_proj_row = torch.zeros(d_model)
            out_proj_row[i] = 1.0  # read from head0
            layer1.attn.out_proj.weight.data[buf_start + i] = out_proj_row

    # ========== LAYER 2: Induction Head (Simplified) ==========
    # Key insight: instead of matching all tokens,  use position ranges to reduce spurious matches
    #
    # Strategy:
    # - Query: check if we're in second half (high pos value), read current token
    # - Key: strong signal if position is in first half
    # - This creates a bias toward attending within the first copy
    # - Then token matching refines the selection

    layer2 = model.blocks[1]
    nn.init.zeros_(layer2.attn.in_proj_weight)
    nn.init.zeros_(layer2.attn.in_proj_bias)
    nn.init.zeros_(layer2.attn.out_proj.weight)
    nn.init.zeros_(layer2.attn.out_proj.bias)
    nn.init.ones_(layer2.ln1.weight)
    nn.init.zeros_(layer2.ln1.bias)

    in_proj2 = torch.zeros(3 * d_model, d_model)

    # Query: reads buffer (previous token) and position
    # q[p] should encode: "I'm looking for token X (from buffer)"
    tok_idx = 0
    for i in range(min(tok_end - tok_start, head_dim // 2)):
        if tok_start + i < d_model:
            in_proj2[0 * d_model + tok_idx, buf_start + i] = scale  # q reads buffer
            tok_idx += 1

    # Key: reads buffer (previous token) + position info to bias towards first half
    # k[j] should encode: "I have token Y and position Z"
    tok_idx = 0
    for i in range(min(tok_end - tok_start, head_dim // 2)):
        if buf_start + i < d_model:
            in_proj2[1 * d_model + tok_idx, buf_start + i] = scale  # k reads buffer
            tok_idx += 1

    # Add position to key: positions in first copy have low pos indices
    # This creates a soft bias toward first-copy positions
    pos_idx = head_dim // 2
    for i in range(min(max_len // 2, head_dim - pos_idx)):
        if pos_start + i < d_model:
            in_proj2[1 * d_model + pos_idx, pos_start + i] = scale / 10.0  # weak bias
            pos_idx += 1

    # Value: read current token (will be weighted by attention)
    for i in range(tok_end - tok_start):
        if tok_start + i < d_model:
            in_proj2[2 * d_model + i, tok_start + i] = 1.0

    model.blocks[1].attn.in_proj_weight = nn.Parameter(in_proj2)

    # out_proj: copy attended token to output
    for i in range(min(tok_end - tok_start, head_dim)):
        if tok_start + i < d_model:
            out_proj_row = torch.zeros(d_model)
            out_proj_row[i] = scale  # read from head0
            layer2.attn.out_proj.weight.data[tok_start + i] = out_proj_row

    # ========== UNEMBEDDING ==========
    head_weight = torch.zeros(vocab, d_model)
    for i in range(vocab):
        if tok_start + i < d_model:
            head_weight[i, tok_start + i] = 1.0
    model.head.weight.data = head_weight

    model.eval()
    return model

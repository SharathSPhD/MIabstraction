"""Nemotron-H style hybrid: mostly state-space mixers, attention every N blocks.

The recipe follows the configuration used in the sibling prabhasa-samskrutam project
(`configs/train/nemotron_h_*.yaml`): a stack where most blocks are Mamba-2 mixers and
roughly one in eight is attention, plus FFNs.

Honest scope: the mixer here is a *selective state-space block written for this repo* —
gated depthwise convolution, input-dependent decay, and an output gate. It has Mamba-2's
shape and its linear-in-sequence-length behaviour, but it is not the official Mamba-2
CUDA kernel and makes no claim to its throughput. It exists so that a Loom program can
target a genuinely different architecture family from a plain decoder, which is what
makes the multi-substrate claim testable rather than decorative.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSMMixer(nn.Module):
    """Selective state-space mixer: sequence mixing without attention.

    h_t = a_t * h_{t-1} + b_t * x_t, with a_t input-dependent (the "selective" part),
    followed by an output gate. Linear in sequence length, no quadratic term.
    """

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner = d_inner
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner,
                              padding=d_conv - 1, bias=True)
        self.d_conv = d_conv
        # input-dependent decay and input gain
        self.decay_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.gain_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        nn.init.constant_(self.decay_proj.bias, 2.0)   # start near "remember"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xz = self.in_proj(x)
        u, gate = xz.chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[:, :, :L].transpose(1, 2)
        u = F.silu(u)

        a = torch.sigmoid(self.decay_proj(u))          # (B, L, d_inner) in (0, 1)
        b = self.gain_proj(u)

        # Causal scan. Written as a loop for clarity and correctness; the point here is
        # architectural difference, not a fused kernel.
        h = torch.zeros(B, self.d_inner, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(L):
            h = a[:, t] * h + b[:, t]
            outs.append(h)
        y = torch.stack(outs, dim=1)
        return self.out_proj(y * F.silu(gate))


class HybridBlock(nn.Module):
    """One block: either an SSM mixer or attention, then an FFN. Same residual
    interface either way, so the rest of the system cannot tell which it got."""

    def __init__(self, d_model: int, n_heads: int, d_ffn: int,
                 use_attention: bool, d_state: int = 64):
        super().__init__()
        self.use_attention = use_attention
        self.ln1 = nn.LayerNorm(d_model)
        if use_attention:
            self.mixer = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        else:
            self.mixer = SSMMixer(d_model, d_state=d_state)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ffn), nn.GELU(),
                                 nn.Linear(d_ffn, d_model))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln1(x)
        if self.use_attention:
            a, _ = self.mixer(h, h, h, attn_mask=mask, need_weights=False)
        else:
            a = self.mixer(h)
        x = x + a
        return x + self.ffn(self.ln2(x))


class NemotronH(nn.Module):
    """Hybrid stack: attention every `attention_every` blocks, SSM elsewhere."""

    def __init__(self, vocab: int, d_model: int = 256, n_layers: int = 8,
                 n_heads: int = 4, d_ffn: int | None = None,
                 attention_every: int = 4, max_len: int = 512, d_state: int = 64):
        super().__init__()
        d_ffn = d_ffn or 4 * d_model
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            HybridBlock(d_model, n_heads, d_ffn,
                        use_attention=((i + 1) % attention_every == 0),
                        d_state=d_state)
            for i in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.attention_every = attention_every

    def pattern(self) -> str:
        """The override pattern, in the notation the recipe configs use: M = mixer,
        A = attention."""
        return "".join("A" if b.use_attention else "M" for b in self.blocks)

    def forward(self, tokens: torch.Tensor,
                collect: bool = False):
        _, L = tokens.shape
        x = self.tok(tokens) + self.pos(torch.arange(L, device=tokens.device))[None]
        mask = torch.triu(torch.full((L, L), float("-inf"), device=tokens.device),
                          diagonal=1)
        resid = []
        for blk in self.blocks:
            x = blk(x, mask if blk.use_attention else None)
            if collect:
                resid.append(x)
        logits = self.head(self.ln_f(x))
        return (logits, resid) if collect else logits

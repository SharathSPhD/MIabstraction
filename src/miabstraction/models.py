"""Minimal hook-instrumented decoder-only transformer.

Written from scratch (no HF dependency) so the residual stream is directly
addressable — the point of the project is to inspect it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d: int, heads: int, attn_only: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.attn_only = attn_only
        if not attn_only:
            self.ln2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(
                nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        if not self.attn_only:
            x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    """Decoder-only LM. forward(tokens, collect=True) also returns the residual
    stream after every block: list of (B, L, d) tensors."""

    def __init__(
        self,
        vocab: int,
        d_model: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        max_len: int = 512,
        attn_only: bool = False,
    ):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, attn_only) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(
        self, tokens: torch.Tensor, collect: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        _, L = tokens.shape
        pos = torch.arange(L, device=tokens.device)
        x = self.tok(tokens) + self.pos(pos)[None]
        mask = torch.triu(
            torch.full((L, L), float("-inf"), device=tokens.device), diagonal=1
        )
        resid = []
        for blk in self.blocks:
            x = blk(x, mask)
            if collect:
                resid.append(x)
        logits = self.head(self.ln_f(x))
        return (logits, resid) if collect else logits


def train_lm(
    model: TinyTransformer,
    tokens: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    device: str,
    log_every: int = 100,
    callback=None,
) -> list[float]:
    """Next-token training. tokens: (N, L) int64. Returns loss history."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    losses = []
    for step in range(steps):
        idx = torch.randint(0, tokens.shape[0], (batch_size,), generator=g)
        batch = tokens[idx].to(device)
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if callback and (step % log_every == 0 or step == steps - 1):
            callback(step, model)
    return losses

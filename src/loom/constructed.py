"""Constructed backend: compile an induction skill DIRECTLY into transformer weights.

No training. The compiler writes machine code:

  layer 0 (shift-1 head):  buf1[p] <- token[p-1]
  layer 1 (shift-2 head):  buf2[p] <- token[p-2]   (reads pos only; independent of L0)
  layer 2 (induction):     attend j where (buf1[j], buf2[j]) == (token[p], buf1[p]),
                           i.e. the position right after an earlier occurrence of the
                           current TWO-token context; copy token[j] to the logits.

Trigram matching is the load-bearing choice: 1-gram induction (match prev-token only)
collides ~copy_len/vocab times per sequence, capping argmax accuracy near 0.5. Matching
two context tokens drops expected collisions to ~copy_len/vocab², so attention mass
concentrates on the true induction target.

LayerNorm survival: every code is a zero-mean ±1 Hadamard row confined to its own
coordinate block, so mean(x)=0 exactly and pre-LN reduces to a pure rescale — direction
is preserved, and large attention-score scales make the rescale irrelevant.

Residual-stream memory map (d_model = 192):
  [  0: 32)  TOK   token identity        H(32) rows 1..vocab
  [ 32: 96)  POS   position              H(64) rows 1..max_len
  [ 96:128)  BUF1  previous token        H(32) rows (same codebook as TOK)
  [128:160)  BUF2  token two back        H(32) rows
  [160:192)  unused (zeros)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from miabstraction.models import TinyTransformer

D = 192
TOK = slice(0, 32)
POS = slice(32, 96)
BUF1 = slice(96, 128)
BUF2 = slice(128, 160)
CYCLE_BLOCK = slice(160, 192)   # the succession skill's private frame; see compile_composed
MAX_VOCAB = 31   # H(32) rows 1..31 are zero-mean
MAX_LEN = 63     # H(64) rows 1..63


def _hadamard(n: int) -> np.ndarray:
    if n == 1:
        return np.array([[1.0]])
    h = _hadamard(n // 2)
    return np.block([[h, h], [h, -h]])


def _codes(n: int, dim: int) -> np.ndarray:
    """n zero-mean orthogonal ±1 codes of length dim (Hadamard rows, row 0 skipped)."""
    H = _hadamard(dim)
    assert n <= dim - 1
    return H[1 : n + 1]


def _shift_matrix(codes: np.ndarray, shift: int) -> np.ndarray:
    """M with M @ c_p = c_{p-shift} (zero for p < shift). Codes are rows."""
    dim = codes.shape[1]
    M = np.zeros((dim, dim))
    for p in range(shift, codes.shape[0]):
        M += np.outer(codes[p - shift], codes[p]) / dim
    return M


def _block(mat_small: np.ndarray, rows: slice, cols: slice) -> np.ndarray:
    """Embed a small matrix into a (D, D) zero matrix at [rows, cols]."""
    M = np.zeros((D, D))
    M[rows, cols] = mat_small
    return M


def _set(t: torch.Tensor, arr: np.ndarray) -> None:
    with torch.no_grad():
        t.copy_(torch.from_numpy(arr).to(t.dtype))


def compile_induction(
    vocab: int = 20,
    max_len: int = 56,
    score_scale: float = 12.0,
    logit_scale: float = 8.0,
) -> TinyTransformer:
    """Emit a 3-layer, 1-head, attention-only TinyTransformer implementing induction."""
    if vocab > MAX_VOCAB or max_len > MAX_LEN:
        raise ValueError(f"constructed induction supports vocab<={MAX_VOCAB}, "
                         f"max_len<={MAX_LEN}")
    model = TinyTransformer(vocab=vocab, d_model=D, n_layers=3, n_heads=1,
                            max_len=max_len, attn_only=True)
    tok = _codes(vocab, 32)
    pos = _codes(max_len, 64)

    # ---- embeddings: block-confined zero-mean codes
    E_tok = np.zeros((vocab, D)); E_tok[:, TOK] = tok
    E_pos = np.zeros((max_len, D)); E_pos[:, POS] = pos
    _set(model.tok.weight, E_tok)
    _set(model.pos.weight, E_pos)

    # ---- neutralize LayerNorms into pure rescale (weight=1, bias=0)
    for blk in model.blocks:
        nn.init.ones_(blk.ln1.weight); nn.init.zeros_(blk.ln1.bias)
    nn.init.ones_(model.ln_f.weight); nn.init.zeros_(model.ln_f.bias)

    S1 = _shift_matrix(pos, 1)
    S2 = _shift_matrix(pos, 2)
    I32 = np.eye(32)

    def wire_shift_layer(blk, shift_mat: np.ndarray, buf: slice) -> None:
        """Q reads shifted own position; K reads position; V reads token -> buf."""
        WQ = _block(shift_mat, POS, POS) * score_scale
        WK = _block(np.eye(64), POS, POS)
        WV = _block(I32, TOK, TOK)          # value carries the token code (TOK rows)
        in_proj = np.concatenate([WQ, WK, WV], axis=0)
        _set(blk.attn.in_proj_weight, in_proj)
        nn.init.zeros_(blk.attn.in_proj_bias)
        # out_proj routes the TOK rows of the value into the buffer block
        WO = np.zeros((D, D)); WO[buf, TOK] = I32
        _set(blk.attn.out_proj.weight, WO)
        nn.init.zeros_(blk.attn.out_proj.bias)

    wire_shift_layer(model.blocks[0], S1, BUF1)
    wire_shift_layer(model.blocks[1], S2, BUF2)

    # ---- layer 2: trigram-matching induction head
    # q_p carries (token[p] -> BUF1-rows, buf1[p] -> BUF2-rows); compared against
    # k_j = (buf1[j] in BUF1-rows, buf2[j] in BUF2-rows).
    # Score ~ tok[p]·buf1[j] + buf1[p]·buf2[j]: maximal iff trigram context matches.
    WQ = (_block(I32, BUF1, TOK) + _block(I32, BUF2, BUF1)) * score_scale
    WK = _block(I32, BUF1, BUF1) + _block(I32, BUF2, BUF2)
    WV = _block(I32, TOK, TOK)
    _set(model.blocks[2].attn.in_proj_weight, np.concatenate([WQ, WK, WV], axis=0))
    nn.init.zeros_(model.blocks[2].attn.in_proj_bias)
    # write the copied token code strongly back into TOK (dominates own-token code)
    WO = np.zeros((D, D)); WO[TOK, TOK] = I32 * logit_scale
    _set(model.blocks[2].attn.out_proj.weight, WO)
    nn.init.zeros_(model.blocks[2].attn.out_proj.bias)

    # ---- unembedding: read TOK block against the token codebook
    W_U = np.zeros((vocab, D)); W_U[:, TOK] = tok
    _set(model.head.weight, W_U)
    model.eval()
    return model


def compile_composed(
    vocab: int = 20,
    n_cycle: int = 6,
    max_len: int = 56,
    score_scale: float = 12.0,
    logit_scale: float = 8.0,
    cycle_scale: float | None = None,
) -> TinyTransformer:
    """Two skills, one weight set: trigram induction plus a succession rule.

    The last `n_cycle` tokens of the vocabulary form a cycle alphabet; on those tokens
    the model predicts the next symbol in the cycle, whatever the context says. All
    other tokens behave exactly as under `compile_induction`: the succession skill
    occupies CYCLE_BLOCK, coordinates the induction circuit never reads or writes,
    which are zero on traffic containing no cycle token — equal logits in exact
    arithmetic, ~1e-14 of float summation residue in practice (LayerNorm's scalar
    shift read against a zero-mean code), and identical argmax everywhere.

    The mechanism is deliberately the shallowest one that is still a mechanism: the
    embedding of a cycle token carries its successor's code in the private block, the
    unembedding reads that block against the codebook for cycle rows only, and nothing
    in between touches it. Arbitration with induction is by scale — the succession
    read must outvote the copied-token write on the one kind of traffic where the two
    skills disagree, which is what `cycle_scale` buys and the adversarial test proves.
    """
    if not 2 <= n_cycle <= min(vocab, MAX_VOCAB):
        raise ValueError(f"n_cycle must be in [2, {min(vocab, MAX_VOCAB)}]")
    model = compile_induction(vocab=vocab, max_len=max_len,
                              score_scale=score_scale, logit_scale=logit_scale)
    tok = _codes(vocab, 32)
    cycle = list(range(vocab - n_cycle, vocab))
    scale = cycle_scale if cycle_scale is not None else 2.5 * logit_scale

    # Embedding: a cycle token announces its successor in the private block. The code
    # is a zero-mean Hadamard row like every other, so LayerNorm stays a pure rescale.
    E = model.tok.weight.detach().cpu().numpy().copy()
    for i, v in enumerate(cycle):
        E[v, CYCLE_BLOCK] = tok[cycle[(i + 1) % n_cycle]]
    _set(model.tok.weight, E)

    # Unembedding: only cycle rows read the private block, scaled to outvote the
    # induction circuit's copied-token write when the two disagree.
    W = model.head.weight.detach().cpu().numpy().copy()
    for v in cycle:
        W[v, CYCLE_BLOCK] = tok[v] * scale
    _set(model.head.weight, W)
    model.eval()
    return model

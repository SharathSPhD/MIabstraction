"""E2: induction-head phase transition (H2).

Train a 2-layer attention-only transformer on doubled sequences [x; x] (x random).
The only generalizing strategy for the second half is induction: at position p >= T,
attend to position p - T + 1 (the token after the previous occurrence) and copy it.

We track (a) loss on the second half (in-context) vs first half, and (b) the
prefix-matching score: attention mass on the induction target, max over layer-2 heads.
H2 supported iff final score > 0.6, the <0.2 -> >0.6 rise happens within a window
< 20% of training, and the second-half loss drop overlaps that window.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..config import ExperimentConfig
from ..models import TinyTransformer, train_lm


def doubled_sequences(n_seq: int, half_len: int, vocab: int,
                      rng: np.random.Generator) -> np.ndarray:
    x = rng.integers(0, vocab, size=(n_seq, half_len), dtype=np.int64)
    return np.concatenate([x, x], axis=1)


def gapped_doubled_sequences(
    n_seq: int, copy_len: int, max_gap: int, vocab: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """[x, junk(g), x, junk(max_gap-g)] with per-sequence random gap g.

    The variable gap defeats the fixed-relative-offset positional shortcut, so
    only content-based induction solves the second copy. Returns (seqs, gaps);
    total length = 2*copy_len + max_gap.
    """
    L = 2 * copy_len + max_gap
    seqs = rng.integers(0, vocab, size=(n_seq, L), dtype=np.int64)
    gaps = rng.integers(0, max_gap + 1, size=n_seq)
    for i in range(n_seq):
        g = gaps[i]
        seqs[i, copy_len + g : 2 * copy_len + g] = seqs[i, :copy_len]
    return seqs, gaps


def attention_patterns(model: TinyTransformer, tokens: torch.Tensor) -> list[torch.Tensor]:
    """Per-layer attention weights (B, heads, L, L), mirroring model.forward."""
    _, L = tokens.shape
    pos = torch.arange(L, device=tokens.device)
    x = model.tok(tokens) + model.pos(pos)[None]
    mask = torch.triu(torch.full((L, L), float("-inf"), device=tokens.device), diagonal=1)
    pats = []
    with torch.no_grad():
        for blk in model.blocks:
            h = blk.ln1(x)
            a, w = blk.attn(h, h, h, attn_mask=mask, need_weights=True,
                            average_attn_weights=False)
            pats.append(w)  # (B, heads, L, L)
            x = x + a
            if not blk.attn_only:
                x = x + blk.mlp(blk.ln2(x))
    return pats


def prefix_matching_score(
    model: TinyTransformer,
    tokens: torch.Tensor,
    gaps: np.ndarray,
    copy_len: int,
    per_layer: bool = False,
) -> float | list[float]:
    """Mean attention mass on the induction target (prev occurrence + 1),
    max over heads; per layer if requested, else for the best layer."""
    pats = attention_patterns(model, tokens)
    scores = []
    for w in pats:  # (B, H, L, L)
        masses = []
        for i in range(w.shape[0]):
            g = int(gaps[i])
            # queries: second-copy positions predicting x[j+1], j = 1..copy_len-1
            q = torch.arange(copy_len + g + 1, 2 * copy_len + g, device=w.device)
            k = q - (copy_len + g) + 1  # = j + 1 in first copy
            masses.append(w[i, :, q, k].mean(dim=0))  # (H,) mean over queries
        m = torch.stack(masses).mean(0)  # (H,)
        scores.append(float(m.max().item()))
    return scores if per_layer else max(scores)


def copy_region_losses(
    model: TinyTransformer, tokens: torch.Tensor, gaps: np.ndarray, copy_len: int
) -> tuple[float, float]:
    """(loss on first copy, loss on second copy) — second is the ICL region."""
    with torch.no_grad():
        logits = model(tokens[:, :-1])
        ce = F.cross_entropy(
            logits.transpose(1, 2), tokens[:, 1:], reduction="none"
        )  # (B, L-1)
    first, second = [], []
    for i in range(tokens.shape[0]):
        g = int(gaps[i])
        first.append(ce[i, : copy_len - 1].mean())
        # targets x[1..] at positions copy_len+g .. 2*copy_len+g-2 in ce index space
        second.append(ce[i, copy_len + g : 2 * copy_len + g - 1].mean())
    return float(torch.stack(first).mean()), float(torch.stack(second).mean())


def find_transition(steps: list[int], scores: list[float],
                    lo: float = 0.2, hi: float = 0.6) -> tuple[int, int] | None:
    """Shortest [s_i, s_j] with score[i] < lo, score[j] > hi."""
    best = None
    for j, sj in enumerate(scores):
        if sj <= hi:
            continue
        cands = [i for i in range(j) if scores[i] < lo]
        if not cands:
            continue
        i = max(cands)
        if best is None or steps[j] - steps[i] < best[1] - best[0]:
            best = (steps[i], steps[j])
    return best


def run(cfg: ExperimentConfig) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    dev = cfg.device if torch.cuda.is_available() else "cpu"

    vocab = cfg.data["vocab"]
    copy_len = cfg.data["copy_len"]
    max_gap = cfg.data["max_gap"]
    seqs, _ = gapped_doubled_sequences(cfg.data["n_seq"], copy_len, max_gap, vocab, rng)
    tokens = torch.from_numpy(seqs)
    pseqs, pgaps = gapped_doubled_sequences(
        cfg.analysis.get("n_probe_seq", 256), copy_len, max_gap, vocab, rng
    )
    probe = torch.from_numpy(pseqs).to(dev)

    model = TinyTransformer(vocab=vocab, **cfg.model)
    log: dict[str, list] = {"step": [], "score": [], "loss_first": [], "loss_second": []}

    def callback(step: int, m: TinyTransformer) -> None:
        m.eval()
        s = prefix_matching_score(m, probe, pgaps, copy_len)
        lf, ls = copy_region_losses(m, probe, pgaps, copy_len)
        log["step"].append(step)
        log["score"].append(s)
        log["loss_first"].append(lf)
        log["loss_second"].append(ls)
        m.train()

    steps = cfg.train["steps"]
    train_lm(model, tokens, device=dev, callback=callback, **cfg.train)

    final_score = log["score"][-1]
    window = find_transition(log["step"], log["score"])
    window_frac = (window[1] - window[0]) / steps if window else None

    # co-timing: >50% of the total second-half loss drop happens inside the window
    co_timed = False
    if window:
        ls = np.array(log["loss_second"])
        st = np.array(log["step"])
        in_w = (st >= window[0]) & (st <= window[1])
        total_drop = ls.max() - ls.min()
        drop_in_window = (ls[in_w].max() - ls[in_w].min()) if in_w.any() else 0.0
        co_timed = bool(total_drop > 0 and drop_in_window / total_drop > 0.5)

    per_layer = prefix_matching_score(model, probe, pgaps, copy_len, per_layer=True)
    supports = bool(final_score > 0.6 and window and window_frac < 0.2 and co_timed)
    result = {
        "prefix_score_per_layer": per_layer,
        "hypothesis": cfg.hypothesis,
        "supports": supports,
        "final_prefix_score": final_score,
        "max_prefix_score": float(max(log["score"])),
        "transition_window": list(window) if window else None,
        "window_frac": window_frac,
        "co_timed": co_timed,
        "final_loss_first_copy": log["loss_first"][-1],
        "final_loss_second_copy": log["loss_second"][-1],
        "uniform_loss_nats": float(np.log(vocab)),
        "leak_budget": float(1 - final_score),
        "config_hash": cfg.hash(),
        "runtime_s": round(time.time() - t0, 1),
        "device": dev,
    }
    d = cfg.result_dir()
    (d / "result.json").write_text(json.dumps(result, indent=2))
    (d / "curves.json").write_text(json.dumps(log))
    _plot(log, window, d)
    return result


def _plot(log, window, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(log["step"], log["score"], "b-", label="prefix-matching score")
    ax1.set_xlabel("step")
    ax1.set_ylabel("score", color="b")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(log["step"], log["loss_first"], "g--", label="first-half loss")
    ax2.plot(log["step"], log["loss_second"], "r-", label="second-half (ICL) loss")
    ax2.set_ylabel("loss (nats)")
    if window:
        ax1.axvspan(window[0], window[1], alpha=0.15, color="orange")
    fig.legend(loc="center right")
    fig.suptitle("E2: induction-head phase transition")
    fig.savefig(d / "phase_transition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

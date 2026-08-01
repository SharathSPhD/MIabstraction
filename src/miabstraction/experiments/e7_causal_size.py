"""E7 — algorithm size measured by ablation, not by counting masked weights.

H5 asked whether sparse training finds smaller circuits and every size metric answered
with the sparsity knob itself: weight ratio 0.2086, edge ratio 0.2093, imposed q
0.200 — the measurement was the setting, and zero variance across seeds was the tell.

The replacement counts *causally necessary heads*: greedily ablate whichever head the
task misses least, stop when any further ablation costs more than epsilon, and count
what remains. Nothing in that procedure mentions the mask, so the number is free to
disagree with q, free to vary across seeds — and free to say the sparse model's
algorithm is no smaller, if that is what is true.

Calibration before use: on the hand-compiled induction circuit, whose true size is
known by construction, the measure must report exactly the three wired heads and must
not count a deliberately idle one (tests/test_e7_causal_size.py).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..config import ExperimentConfig


class _HeadAblation:
    """Zero one head's output contribution, reversibly.

    nn.MultiheadAttention concatenates the heads and mixes them through out_proj, so
    column block [h*d_head:(h+1)*d_head] of out_proj.weight is exactly head h's route
    into the residual stream. Zeroing it removes the head's effect and nothing else.
    """

    def __init__(self, model, layer: int, head: int):
        attn = model.blocks[layer].attn
        d_head = attn.embed_dim // attn.num_heads
        self.weight = attn.out_proj.weight
        self.cols = slice(head * d_head, (head + 1) * d_head)
        self.saved: torch.Tensor | None = None

    def apply(self) -> None:
        with torch.no_grad():
            self.saved = self.weight[:, self.cols].clone()
            self.weight[:, self.cols] = 0.0

    def restore(self) -> None:
        assert self.saved is not None
        with torch.no_grad():
            self.weight[:, self.cols] = self.saved
        self.saved = None

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, *a):
        self.restore()


def head_handles(model) -> list[tuple[tuple[int, int], _HeadAblation]]:
    out = []
    for layer, blk in enumerate(model.blocks):
        for head in range(blk.attn.num_heads):
            out.append(((layer, head), _HeadAblation(model, layer, head)))
    return out


def necessary_heads(model, accuracy_fn, epsilon: float = 0.05
                    ) -> tuple[list[tuple[int, int]], list[dict]]:
    """Greedy backward elimination at head granularity.

    At every step, tentatively ablate each remaining head on top of everything already
    ablated, and permanently remove the one the task misses least — as long as the
    model stays within epsilon of its full accuracy. What survives is the necessary
    set. The trace records every decision so a surprising count can be audited.
    """
    handles = dict(head_handles(model))
    full = accuracy_fn(model)
    ablated: list[tuple[int, int]] = []
    trace: list[dict] = [{"step": 0, "full_accuracy": full}]

    while True:
        remaining = [k for k in handles if k not in ablated]
        if not remaining:
            break
        best_key, best_acc = None, -1.0
        for key in remaining:
            for k in ablated:
                handles[k].apply()
            handles[key].apply()
            acc = accuracy_fn(model)
            handles[key].restore()
            for k in reversed(ablated):
                handles[k].restore()
            if acc > best_acc:
                best_key, best_acc = key, acc
        if best_acc < full - epsilon:
            trace.append({"stopped": True, "would_remove": best_key,
                          "accuracy_would_be": best_acc,
                          "floor": full - epsilon})
            break
        ablated.append(best_key)
        trace.append({"removed": best_key, "accuracy": best_acc,
                      "n_ablated": len(ablated)})

    kept = sorted(k for k in handles if k not in ablated)
    return kept, trace


def run(cfg: ExperimentConfig) -> dict:
    """Dense vs sparse E5 models, sized by ablation, multi-seed.

    Retrains the E5 models (checkpoints are not committed), then counts necessary
    heads for each. The H5 question becomes: does sparse training reduce the number
    of heads the algorithm actually uses — answered by a measure that cannot return
    the knob, with the variance reported so a constant cannot masquerade as a result.
    """
    from ..data.algo import BracketMatchingDataset
    from ..models import TinyTransformer
    from .e5_sparsity import _train_model, _train_model_sparse

    t0 = time.time()
    dev = cfg.device if torch.cuda.is_available() else "cpu"
    seeds = list(cfg.analysis.get("seeds", [0, 1]))
    epsilon = float(cfg.analysis.get("epsilon", 0.05))
    target_q = float(cfg.analysis.get("target_q", 0.2))

    # The same task E5 trained on, so the size comparison is about the models rather
    # than about a change of task.
    vocab_size = cfg.data.get("vocab_size", 10)
    seq_len = cfg.data.get("seq_len", 16)
    ds_train = BracketMatchingDataset(vocab_size=vocab_size, seq_len=seq_len,
                                      n_samples=cfg.data.get("n_seq_train", 500),
                                      seed=cfg.seed)
    ds_val = BracketMatchingDataset(vocab_size=vocab_size, seq_len=seq_len,
                                    n_samples=cfg.data.get("n_seq_val", 200),
                                    seed=cfg.seed + 1)
    train_seqs = torch.stack([ds_train[i] for i in range(len(ds_train))])
    val_seqs = torch.stack([ds_val[i] for i in range(len(ds_val))])
    val_masks, val_labels = [], []
    for i in range(len(ds_val)):
        _, mask, label = ds_val.get_with_mask(i)
        val_masks.append(mask)
        val_labels.append(label)
    val_masks = torch.stack(val_masks)
    val_labels = torch.tensor(val_labels)

    @torch.no_grad()
    def accuracy(model) -> float:
        model.eval()
        logits = model(val_seqs[:, :-1].to(dev))
        correct = 0
        for b in range(len(val_seqs)):
            ans_pos = val_masks[b].nonzero(as_tuple=True)[0].item()
            if 0 < ans_pos <= logits.shape[1]:
                pred = logits[b, ans_pos - 1, :].argmax().cpu().item()
                correct += int(pred == val_labels[b].item())
        return correct / len(val_seqs)

    rows = []
    for seed in seeds:
        for kind in ("dense", "sparse"):
            torch.manual_seed(seed)
            model = TinyTransformer(vocab=vocab_size, **cfg.model)
            if kind == "dense":
                _train_model(model, train_seqs, cfg, dev, dataset=ds_train)
            else:
                _train_model_sparse(model, train_seqs, cfg, dev,
                                    target_q=target_q, dataset=ds_train)
            model.eval()
            kept, trace = necessary_heads(model, accuracy, epsilon)
            rows.append({"seed": seed, "kind": kind,
                         "accuracy_full": accuracy(model),
                         "necessary_heads": len(kept),
                         "which": [list(k) for k in kept],
                         "trace_tail": trace[-1]})
            del model
            torch.cuda.empty_cache()

    dense = [r["necessary_heads"] for r in rows if r["kind"] == "dense"]
    sparse = [r["necessary_heads"] for r in rows if r["kind"] == "sparse"]
    return {
        "experiment": cfg.name, "hypothesis": cfg.hypothesis, "seed": cfg.seed,
        "epsilon": epsilon, "rows": rows,
        "dense_necessary_heads": dense, "sparse_necessary_heads": sparse,
        "dense_mean": float(np.mean(dense)), "sparse_mean": float(np.mean(sparse)),
        "dense_var": float(np.var(dense)), "sparse_var": float(np.var(sparse)),
        "measure_can_vary": bool(np.var(dense + sparse) > 0
                                 or len(set(dense + sparse)) > 1),
        "note": ("size by causal ablation at head granularity; the imposed q appears "
                 "nowhere in the measure. Zero variance across every cell would be "
                 "the old tautology tell and must be treated as such."),
        "wall_clock_s": round(time.time() - t0, 1),
    }

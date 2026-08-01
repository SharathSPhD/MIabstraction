"""E7 — a size measure with no weight-counting in it, calibrated on ground truth.

H5 was undecidable because every size metric returned the imposed sparsity: weight
ratio 0.2086, edge ratio 0.2093, q 0.200. The replacement counts what ablation proves
necessary, so before trusting it anywhere it must pass the one model whose true size
is known by construction: the hand-compiled induction circuit uses exactly its three
heads, and a metric that reports anything else is measuring something else.
"""
import numpy as np
import pytest
import torch

from loom.constructed import compile_induction
from miabstraction.experiments.e2_induction import gapped_doubled_sequences
from miabstraction.experiments.e7_causal_size import (head_handles,
                                                      necessary_heads)

VOCAB, COPY, GAP = 20, 22, 16


@pytest.fixture(scope="module")
def circuit():
    return compile_induction(vocab=VOCAB, max_len=60)


@pytest.fixture(scope="module")
def accuracy():
    seqs, gaps = gapped_doubled_sequences(24, COPY, GAP, VOCAB,
                                          np.random.default_rng(0))
    tokens = torch.from_numpy(seqs)

    @torch.no_grad()
    def fn(model) -> float:
        logits = model(tokens[:, :-1])
        hits = total = 0
        for b in range(seqs.shape[0]):
            g = int(gaps[b])
            for p in range(COPY + g + 2, 2 * COPY + g):
                hits += int(logits[b, p - 1].argmax().item() == seqs[b, p])
                total += 1
        return hits / total

    return fn


def test_every_head_of_the_constructed_circuit_is_necessary(circuit, accuracy):
    """Ground truth: three layers, one head each, all load-bearing. The measure must
    find exactly {(0,0),(1,0),(2,0)} — reporting fewer means the ablation leaks,
    reporting more is impossible here and would mean the bookkeeping is wrong."""
    kept, trace = necessary_heads(circuit, accuracy, epsilon=0.05)
    assert set(kept) == {(0, 0), (1, 0), (2, 0)}, (kept, trace)


def test_ablating_any_circuit_head_destroys_the_skill(circuit, accuracy):
    full = accuracy(circuit)
    assert full > 0.9
    for (layer, head), handle in head_handles(circuit):
        with handle:
            assert accuracy(circuit) < full - 0.05, (
                f"head ({layer},{head}) ablated with no accuracy cost — the "
                "ablation is not reaching the computation")
    assert accuracy(circuit) == full, "ablation leaked outside its context manager"


def test_a_gifted_extra_head_is_not_counted(accuracy):
    """A circuit compiled into a model with idle capacity must measure the same size:
    the idle head does nothing, ablation proves it, and the count excludes it."""
    from miabstraction.models import TinyTransformer
    model = compile_induction(vocab=VOCAB, max_len=60)
    # a 4th layer of pure zeros: present in the architecture, absent from the algorithm
    import torch.nn as nn
    extra = TinyTransformer(vocab=VOCAB, d_model=192, n_layers=4, n_heads=1,
                            max_len=60, attn_only=True)
    with torch.no_grad():
        for src, dst in zip(model.blocks, extra.blocks[:3]):
            dst.load_state_dict(src.state_dict())
        for p in extra.blocks[3].attn.parameters():
            p.zero_()
        nn.init.ones_(extra.blocks[3].ln1.weight)
        nn.init.zeros_(extra.blocks[3].ln1.bias)
        extra.tok.load_state_dict(model.tok.state_dict())
        extra.pos.load_state_dict(model.pos.state_dict())
        extra.head.load_state_dict(model.head.state_dict())
        nn.init.ones_(extra.ln_f.weight)
        nn.init.zeros_(extra.ln_f.bias)
    kept, _ = necessary_heads(extra, accuracy, epsilon=0.05)
    assert (3, 0) not in kept
    assert set(kept) == {(0, 0), (1, 0), (2, 0)}

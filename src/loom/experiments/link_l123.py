"""L1/L2/L3 — does separate compilation and linking actually work for transformers?

Host: a small LM trained on a mixed corpus (structured Mess3-like text + noise) that
does NOT contain the copy structure the linked unit implements — so any induction
ability after linking demonstrably came from the link, not from the host.

Unit A: the hand-compiled induction circuit (constructed backend, zero training).
Unit B: a second independently compiled unit (a bias unit over a token subset), used
        only to test composition.

L1 no-clobber   link A into the host: host loss within budget AND A's gate still passes.
L2 composition  link A and B together: both gates hold, host still within budget.
L3 capacity     out-of-band cost vs the in-band alternative (reserving host dims).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from miabstraction.data.mess3 import mess3_matrices, sample_sequences
from miabstraction.experiments.e2_induction import (
    gapped_doubled_sequences,
    prefix_matching_score,
)
from miabstraction.models import TinyTransformer, train_lm
from miabstraction.seeding import set_determinism

from ..abi import Envelope, Mode, ReadKind, Unit, WriteKind
from ..constructed import compile_induction
from ..linker import LinkedModel, host_loss, in_band_cost, link

VOCAB, COPY, GAP, MAXLEN = 20, 22, 16, 60
SEQ = 2 * COPY + GAP  # 60


def bigram_process() -> np.ndarray:
    """The host's language, fixed once. Train and held-out must come from the SAME
    process — drawing a fresh matrix per call made held-out loss worse than uniform
    and silently inverted the capacity comparison."""
    return np.random.default_rng(1234).dirichlet(np.ones(VOCAB) * 0.4, size=VOCAB)


def host_corpus(n: int, rng: np.random.Generator) -> torch.Tensor:
    """Text-like data with local structure but NO long-range copy structure.

    A first-order Markov chain over the full vocabulary: the host can learn real
    (bigram) statistics, but nothing here rewards induction, so any copying ability
    after linking demonstrably came from the linked unit.

    The earlier version embedded a 3-symbol Mess3 process, which gave each sequence
    only three distinct tokens — the induction unit then found a "repeat" almost
    everywhere and fired constantly. That was an envelope violation on the unit's
    side, and it is why the linker refused; a token-diverse host is the fair test.
    """
    P = bigram_process()  # ONE process shared by train and held-out
    out = np.empty((n, SEQ), dtype=np.int64)
    out[:, 0] = rng.integers(0, VOCAB, size=n)
    for t in range(1, SEQ):
        probs = P[out[:, t - 1]]
        cdf = np.cumsum(probs, axis=1)
        out[:, t] = (rng.random(n)[:, None] < cdf).argmax(axis=1)
    return torch.from_numpy(out)


def induction_gate(model, tokens, gaps, threshold=0.55):
    """Did the model just predict the second copy? Measured on the linked model."""
    with torch.no_grad():
        logits = model(tokens[:, :-1])
    correct = total = 0
    for b in range(tokens.shape[0]):
        g = int(gaps[b])
        for p in range(COPY + g + 2, 2 * COPY + g):
            correct += int(logits[b, p - 1].argmax().item() == tokens[b, p].item())
            total += 1
    acc = correct / max(total, 1)
    return {"passed": acc > threshold, "icl_acc": acc}


class BiasUnit(nn.Module):
    """A trivially, independently compiled second unit: favours a token subset.
    Its only job here is to test composition (L2)."""

    def __init__(self, vocab: int, favored: list[int]):
        super().__init__()
        w = torch.full((vocab,), -4.0)
        w[favored] = 4.0
        self.register_buffer("w", w)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L = tokens.shape
        return self.w.view(1, 1, -1).expand(B, L, -1)


def run(device: str = "cuda", out: str = "results/loom_link_demo.json") -> dict:
    t0 = time.time()
    set_determinism(0)
    dev = device if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    # ---- host: trained, and deliberately induction-free
    host = TinyTransformer(vocab=VOCAB, d_model=128, n_layers=4, n_heads=4,
                           max_len=MAXLEN + 4)
    corpus = host_corpus(20000, rng)
    train_lm(host, corpus, steps=3000, batch_size=256, lr=1e-3, device=dev)
    host.to(dev).eval()

    held = host_corpus(512, rng).to(dev)
    ind_tokens_np, gaps = gapped_doubled_sequences(128, COPY, GAP, VOCAB,
                                                   np.random.default_rng(7))
    ind_tokens = torch.from_numpy(ind_tokens_np).to(dev)

    base_host_loss = host_loss(host, held, dev)
    host_alone = induction_gate(host, ind_tokens, gaps)

    # ---- units, compiled separately and never trained together with the host
    def context_seen_before(host_logits, unit_logprobs, tokens):
        """The induction unit's VALIDITY signal: fire only where the current two-token
        context actually occurred earlier in the sequence, so there is a real
        continuation to copy.

        Output confidence cannot serve this role. The constructed circuit uses near-hard
        attention, so it is confidently wrong off-distribution — it fired on 97.8% of
        ordinary text while being right on almost none of it. Applicability is a
        property of the input, and the ABI asks the unit to declare it.
        """
        B, L = tokens.shape
        mask = torch.zeros(B, L, device=tokens.device)
        prev, cur = tokens[:, :-1], tokens[:, 1:]           # bigram at each position
        for p in range(2, L):
            hit = ((prev[:, : p - 1] == prev[:, p - 1: p])
                   & (cur[:, : p - 1] == cur[:, p - 1: p])).any(dim=1)
            mask[:, p] = hit.float()
        return mask

    unit_a = Unit(
        name="induction",
        core=compile_induction(vocab=VOCAB, max_len=MAXLEN),
        read=ReadKind.TOKENS, write=WriteKind.LOGITS, mode=Mode.OUT_OF_BAND,
        envelope=Envelope(vocab=VOCAB, max_len=MAXLEN,
                          notes="verified on variable-gap doubled sequences"),
        when=context_seen_before,
    )
    unit_b = Unit(
        name="token_bias",
        core=BiasUnit(VOCAB, favored=[1, 2, 3]),
        read=ReadKind.TOKENS, write=WriteKind.LOGITS, mode=Mode.OUT_OF_BAND,
        envelope=Envelope(vocab=VOCAB, max_len=MAXLEN),
    )

    budget = 0.05
    gate_a = lambda m: induction_gate(m, ind_tokens, gaps)

    def gate_b(m):
        with torch.no_grad():
            lg = m(held[:, :-1])
        share = torch.softmax(lg.float(), -1)[..., [1, 2, 3]].sum(-1).mean().item()
        return {"passed": share > 0.5, "favored_mass": share}

    # ---- L1: link the induction unit alone, at a strict budget.
    lm1, rep1 = link(host, [unit_a], held, {"induction": gate_a}, device=dev,
                     budget=budget)

    # The price curve: what does the host actually pay for each level of linked skill?
    # A refusal is only informative if it comes with the trade-off it refused.
    pareto = []
    for g in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4):
        unit_a.gain = g
        lm = LinkedModel(host, [unit_a], device=dev)
        gt = induction_gate(lm, ind_tokens, gaps)
        pareto.append({"gain": g, "icl_acc": round(gt["icl_acc"], 4),
                       "host_delta": round(host_loss(lm, held, dev) - base_host_loss, 4),
                       "firing_rate_on_host": round(lm.firing_rate.get("induction", 1.0), 4)})
    unit_a.gain = 1.0

    # A second link at a budget the operator is willing to pay, to show the linker
    # succeeds when the price is acceptable — the dial is policy, not mechanism.
    lax = 0.75
    lm1b, rep1b = link(host, [unit_a], held, {"induction": gate_a}, device=dev,
                       budget=lax)

    # ---- L2: composition. The question separate compilation actually turns on is
    # INTERFERENCE: does adding a second, independently compiled unit change what the
    # first one does? Run at the budget where a single link is feasible, then compare
    # each unit's score alone against its score alongside the other.
    unit_a2 = Unit(**{**unit_a.__dict__})
    unit_b.gain = 1.0
    lm_b_only = LinkedModel(host, [unit_b], device=dev)
    b_alone = gate_b(lm_b_only)

    # Linking costs ADD: each unit's write perturbs the host independently, so a
    # composed link needs a budget covering the sum of its parts. Refusing a composition
    # under a single-unit budget is arithmetic, not interference — the interesting
    # question only becomes visible once the budget admits both.
    compose_budget = 1.25
    lm2, rep2 = link(host, [unit_a2, unit_b], held,
                     {"induction": gate_a, "token_bias": gate_b},
                     device=dev, budget=compose_budget)

    # Measure interference whether or not the link is accepted — a refusal is only
    # informative if it says by how much the units disturbed each other.
    from ..linker import LinkedModel as _LM
    lm_pair = _LM(host, [unit_a2, unit_b], device=dev)
    interference = None
    if True:
        a_with_b = induction_gate(lm_pair, ind_tokens, gaps)["icl_acc"]
        b_with_a = gate_b(lm_pair)["favored_mass"]
        a_alone = rep1b.unit_gates["induction"].get("icl_acc")
        interference = {
            "a_alone": a_alone, "a_with_b": a_with_b,
            "a_shift": (a_with_b - a_alone) if a_alone is not None else None,
            "b_alone": b_alone["favored_mass"], "b_with_a": b_with_a,
            "b_shift": b_with_a - b_alone["favored_mass"],
        }

    # ---- L3: capacity — what the in-band alternative would have cost
    width = host.tok.embedding_dim
    in_band = {}
    for k in (8, 16, 32):
        in_band[str(k)] = in_band_cost(host, held, dev, n_dims=k) - base_host_loss
    oob_cost = rep1.host_delta

    result = {
        "experiment": "L1/L2/L3 separate compilation and linking",
        "host": {"d_model": width, "n_layers": 4, "trained_steps": 3000,
                 "val_loss": base_host_loss,
                 "induction_acc_alone": host_alone["icl_acc"]},
        "L1_no_clobber": {
            "linked": rep1.linked,
            "gain": rep1.gains.get("induction"),
            "unit_icl_acc": rep1.unit_gates["induction"].get("icl_acc"),
            "host_loss_before": rep1.host_loss_before,
            "host_loss_after": rep1.host_loss_after,
            "host_delta": rep1.host_delta,
            "budget": budget,
            "capacity_cost_dims": rep1.capacity_cost.get("induction"),
            "diagnosis": rep1.diagnosis,
            "price_curve": pareto,
            "at_relaxed_budget": {
                "budget": lax, "linked": rep1b.linked,
                "gain": rep1b.gains.get("induction"),
                "unit_icl_acc": rep1b.unit_gates["induction"].get("icl_acc"),
                "host_delta": rep1b.host_delta,
                "diagnosis": rep1b.diagnosis,
            },
        },
        "L2_composition": {
            "budget": compose_budget,
            "linked": rep2.linked,
            "gains": rep2.gains,
            "gates": rep2.unit_gates,
            "host_delta": rep2.host_delta,
            "interference": interference,
            "diagnosis": rep2.diagnosis,
        },
        "L3_capacity": {
            "out_of_band_host_delta": oob_cost,
            "in_band_host_delta_by_reserved_dims": in_band,
            "out_of_band_dims_taken": 0,
            "verdict": ("out-of-band cheaper" if all(
                oob_cost < v for v in in_band.values()) else "in-band competitive"),
        },
        "wall_clock_s": round(time.time() - t0, 1),
        "device": dev,
    }
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

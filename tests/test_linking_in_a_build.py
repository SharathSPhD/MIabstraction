"""The ABI used by a build, not studied beside one.

An abstraction layer that no build path calls is a description of an abstraction
layer. These tests pin the three things linking gives a build that nothing else in
the compiler can:

  - a skill realized with **no training at all**, because the circuit was compiled
    once and verified once;
  - a **refusal** when the host is outside the circuit's verified envelope, carrying
    the numbers that caused it — the honest outcome a fine-tuning fallback hides;
  - **separation on condition**: the unit writes only where its predicate fires, so a
    host is not taxed on traffic the unit knows nothing about.
"""
from __future__ import annotations

import numpy as np
import torch

from loom.app.linking import induction_unit, link_skill
from miabstraction.experiments.e2_induction import gapped_doubled_sequences
from miabstraction.models import TinyTransformer

VOCAB, COPY, GAP, CTX = 20, 22, 16, 60


def _host(seed: int = 0):
    torch.manual_seed(seed)
    return TinyTransformer(vocab=VOCAB, d_model=64, n_layers=4, n_heads=4,
                           max_len=CTX)


def _tokens(n: int = 16):
    seqs, _ = gapped_doubled_sequences(n, COPY, GAP, VOCAB,
                                       np.random.default_rng(0))
    return torch.from_numpy(seqs)


def test_linking_realizes_a_skill_without_training():
    """The claim that distinguishes this from fine-tuning: no gradient is taken."""
    host, toks = _host(), _tokens()
    before = [p.detach().clone() for p in host.parameters()]
    r = link_skill(host, toks, vocab=VOCAB, max_len=CTX, device="cpu")

    assert r["linked"], r
    assert r["skill_linked"] > r["skill_alone"] * 2, (
        f"linking bought little: {r['skill_alone']} -> {r['skill_linked']}")
    # The host's weights are untouched. A LoRA build cannot say this.
    for p, q in zip(host.parameters(), before):
        assert torch.equal(p.detach(), q), "linking modified the host's weights"


def test_a_host_outside_the_envelope_is_refused_with_its_numbers():
    """A promise measured at one scale is void at another, and the refusal has to
    say which number broke it — otherwise it reads as a failure of the tooling."""
    r = link_skill(_host(), _tokens(), vocab=50_000, max_len=CTX, device="cpu")
    assert not r["linked"]
    assert "envelope" in r and r["envelope"]["vocab_max"] < 50_000
    assert "50,000" in r["reason"] and str(r["envelope"]["vocab_max"]) in r["reason"]


def test_the_unit_does_not_fire_everywhere():
    """Separation on condition. A unit that speaks on every token taxes the host for
    the positions it knows nothing about; this one speaks where the context repeats."""
    r = link_skill(_host(), _tokens(), vocab=VOCAB, max_len=CTX, device="cpu")
    rate = r["firing_rate_on_host_traffic"]
    assert 0.0 < rate < 1.0, f"unit fired at {rate}: that is not a condition"


def test_the_condition_is_about_the_input_not_the_units_confidence():
    """A hand-built circuit with near-hard attention is confidently wrong off its
    distribution, so its own confidence cannot be its validity signal. The predicate
    must therefore be a function of the tokens."""
    unit = induction_unit(VOCAB, CTX)
    assert unit is not None and unit.when is not None
    toks = torch.tensor([[1, 2, 3, 1, 2, 3]])
    mask = unit.when(torch.zeros(1, 6, VOCAB), torch.zeros(1, 6, VOCAB), toks)
    # Positions 0-2 are first occurrences; 3-5 repeat and are where induction applies.
    assert mask[0, :3].sum() == 0 and mask[0, 3:].sum() == 3


def test_the_host_is_not_damaged():
    """Linking is only worth having if the host survives it. The budget is declared
    and the report states what was actually paid."""
    r = link_skill(_host(), _tokens(), vocab=VOCAB, max_len=CTX, device="cpu")
    assert r["host_paid_nats"] <= r["budget_nats"], (
        f"host paid {r['host_paid_nats']} against a budget of {r['budget_nats']}")

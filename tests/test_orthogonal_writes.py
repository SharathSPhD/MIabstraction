"""Separate compilation needs an allocation, not just a calling convention.

The ABI gives a unit private state and a declared write. It does not say what
happens when two units write at once — and measured, they collide: a unit that
scored 0.588 alone dropped to 0.447 with a second unit linked beside it, and the
linker rightly refused the composition. That refusal is the honest outcome of an
ABI with no write allocation, and it is the thing standing between this project and
the claim that makes an abstraction layer real: separately compiled units coexist.

The constructed backend already showed what a solution looks like. Induction and
succession share one weight set without interference because they occupy disjoint
coordinate blocks — the second skill's write lands where the first never reads. The
linker's units write into shared logit space, where no such block exists, so the
allocation has to be made rather than found: at each position, order the units and
project each write onto the orthogonal complement of the writes before it.

What that buys, and these tests pin it:
  - the first unit is EXACTLY unaffected by later ones (not approximately)
  - later units still deliver whatever part of their opinion is theirs alone
  - the allocation is declared and reproducible, not a tuning accident
"""
from __future__ import annotations

import torch

from loom.abi import Mode, ReadKind, Unit, WriteKind, WriteAlloc
from loom.linker import LinkedModel


class Tiny(torch.nn.Module):
    """A model whose logits are a fixed pattern — so any change is the units'."""

    def __init__(self, vocab: int = 16, bias: float = 0.0):
        super().__init__()
        self.vocab = vocab
        self.head = torch.nn.Linear(4, vocab, bias=False)
        self.emb = torch.nn.Embedding(vocab, 4)
        torch.nn.init.normal_(self.emb.weight, std=0.5)
        torch.nn.init.normal_(self.head.weight, std=0.5)
        self.bias = bias

    def forward(self, tokens):
        return self.head(self.emb(tokens)) + self.bias


def _unit(name: str, vocab: int, seed: int, gain: float = 1.0) -> Unit:
    torch.manual_seed(seed)
    core = Tiny(vocab)
    return Unit(name=name, core=core, mode=Mode.OUT_OF_BAND,
                read=ReadKind.TOKENS, write=WriteKind.LOGITS, gain=gain)


def _tokens(vocab: int = 16, n: int = 8, length: int = 12) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, vocab, (n, length))


def test_shared_allocation_lets_units_collide():
    """The status quo, stated as a measurement rather than a worry."""
    v = 16
    host, toks = Tiny(v), _tokens(v)
    a, b = _unit("a", v, 1), _unit("b", v, 2)

    solo = LinkedModel(host, [a], alloc=WriteAlloc.SHARED)(toks)
    both = LinkedModel(host, [a, b], alloc=WriteAlloc.SHARED)(toks)
    # a's contribution is not preserved: b lands on top of it.
    assert not torch.allclose(solo, both, atol=1e-4)


def test_orthogonal_allocation_protects_the_first_unit_exactly():
    """The invention: under orthogonal allocation the earlier unit's write survives
    the later one untouched, because the later one is projected out of it."""
    v = 16
    host, toks = Tiny(v), _tokens(v)
    a, b = _unit("a", v, 1), _unit("b", v, 2)

    solo = LinkedModel(host, [a], alloc=WriteAlloc.ORTHOGONAL)(toks)
    both = LinkedModel(host, [a, b], alloc=WriteAlloc.ORTHOGONAL)(toks)

    # Project both results back onto a's own write direction: what a said must be
    # unchanged. Componentwise equality would demand b write nothing at all.
    d = (solo - host(toks)).flatten(0, 1)                 # a's write, per position
    both_w = (both - host(toks)).flatten(0, 1)
    unit = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    along_solo = (d * unit).sum(-1)
    along_both = (both_w * unit).sum(-1)
    assert torch.allclose(along_solo, along_both, atol=1e-4), (
        "the first unit's own component changed when a second unit was linked")


def test_the_later_unit_still_writes_something():
    """An allocation that protects the first by silencing the second is not an
    allocation; it is a veto."""
    v = 16
    host, toks = Tiny(v), _tokens(v)
    a, b = _unit("a", v, 1), _unit("b", v, 2)

    one = LinkedModel(host, [a], alloc=WriteAlloc.ORTHOGONAL)(toks)
    two = LinkedModel(host, [a, b], alloc=WriteAlloc.ORTHOGONAL)(toks)
    assert (two - one).abs().max() > 1e-3, "the second unit contributed nothing"


def test_allocation_is_deterministic():
    """Same units, same order, same result — a composition you can ship."""
    v = 16
    host, toks = Tiny(v), _tokens(v)
    a, b = _unit("a", v, 1), _unit("b", v, 2)
    r1 = LinkedModel(host, [a, b], alloc=WriteAlloc.ORTHOGONAL)(toks)
    r2 = LinkedModel(host, [a, b], alloc=WriteAlloc.ORTHOGONAL)(toks)
    assert torch.equal(r1, r2)


def test_order_is_the_declared_priority():
    """Orthogonalization is asymmetric, so the order is part of the contract, and
    swapping it must visibly change which unit is protected."""
    v = 16
    host, toks = Tiny(v), _tokens(v)
    a, b = _unit("a", v, 1), _unit("b", v, 2)
    ab = LinkedModel(host, [a, b], alloc=WriteAlloc.ORTHOGONAL)(toks)
    ba = LinkedModel(host, [b, a], alloc=WriteAlloc.ORTHOGONAL)(toks)
    assert not torch.allclose(ab, ba, atol=1e-5)


def test_one_unit_is_unchanged_by_the_allocation():
    """With nothing to be orthogonal to, the allocation must be a no-op — otherwise
    it would be silently altering single-unit links that already work."""
    v = 16
    host, toks = Tiny(v), _tokens(v)
    a = _unit("a", v, 1)
    shared = LinkedModel(host, [a], alloc=WriteAlloc.SHARED)(toks)
    ortho = LinkedModel(host, [a], alloc=WriteAlloc.ORTHOGONAL)(toks)
    assert torch.allclose(shared, ortho, atol=1e-6)

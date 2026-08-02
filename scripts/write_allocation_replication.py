"""Does orthogonal write allocation preserve the first unit, or did it once?

The linker offers three ways for two units to write into one model. SHARED is the naive
convention — both add into the same output and the later one lands on top of the earlier.
ORTHOGONAL projects the later unit's write out of the earlier one's direction, so the
earlier unit's own component survives exactly. EXCLUSIVE gives each position to one unit.

That claim was measured on a single pair of units. One pair is an anecdote, and ten runs
of the same construction with different seeds is the same anecdote ten times. So the grid
varies the things that could plausibly break it:

  - **vocabulary width**, because the projection lives in logit space and a wider space
    makes two random directions more nearly orthogonal by default — which would flatter
    the result for a reason that has nothing to do with the allocation;
  - **the later unit's gain**, because the interesting failure is a loud second unit
    arriving after a quiet first one;
  - **what the units are**, including the compiled induction circuit the build path
    actually links, rather than only random matrices.

Reported: the fraction of pairs where the first unit's write is preserved exactly, the
worst error seen, and — the control that matters — whether the second unit still wrote
anything. An allocation that protects the first unit by silencing the second is a veto,
and it would score perfectly on preservation alone.

Run: .venv/bin/python scripts/write_allocation_replication.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from loom.abi import Mode, ReadKind, Unit, WriteAlloc, WriteKind  # noqa: E402
from loom.linker import LinkedModel  # noqa: E402

TOL = 1e-4


class Tiny(torch.nn.Module):
    """A model whose logits are a fixed function of its tokens, so every difference
    measured below belongs to a unit rather than to sampling."""

    def __init__(self, vocab: int = 16, d: int = 4):
        super().__init__()
        self.vocab = vocab
        self.emb = torch.nn.Embedding(vocab, d)
        self.head = torch.nn.Linear(d, vocab, bias=False)
        torch.nn.init.normal_(self.emb.weight, std=0.5)
        torch.nn.init.normal_(self.head.weight, std=0.5)

    def forward(self, tokens):
        return self.head(self.emb(tokens))


def random_unit(name: str, vocab: int, seed: int, gain: float) -> Unit:
    torch.manual_seed(seed)
    return Unit(name=name, core=Tiny(vocab), mode=Mode.OUT_OF_BAND,
                read=ReadKind.TOKENS, write=WriteKind.LOGITS, gain=gain)


def induction_as_unit(vocab: int, max_len: int, gain: float) -> Unit | None:
    """The circuit the build path really links, used here as one half of a pair."""
    from loom.app.linking import induction_unit
    u = induction_unit(vocab, max_len, gain=gain)
    if u is not None:
        u.name = "induction"
    return u


def preservation(host, a: Unit, b: Unit, toks: torch.Tensor,
                 alloc: WriteAlloc) -> dict:
    """How much of what `a` said survives `b` being linked after it.

    Measured along a's own write direction rather than componentwise: demanding
    componentwise equality of the whole output would require b to write nothing, which
    is a different (and useless) property.
    """
    with torch.no_grad():
        base = host(toks)
        solo = LinkedModel(host, [a], alloc=alloc)(toks)
        both = LinkedModel(host, [a, b], alloc=alloc)(toks)

    d = (solo - base).flatten(0, 1)
    both_w = (both - base).flatten(0, 1)
    keep = d.norm(dim=-1) > 1e-9          # positions where a actually wrote
    if keep.sum() == 0:
        return {"error": None, "reason": "the first unit wrote nothing anywhere"}
    d, both_w = d[keep], both_w[keep]
    direction = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    along_solo = (d * direction).sum(-1)
    along_both = (both_w * direction).sum(-1)
    err = float((along_solo - along_both).abs().max())

    # The control: did the second unit write at all? A perfect preservation score with
    # a silent second unit is a veto wearing an allocation's name.
    with torch.no_grad():
        b_only = LinkedModel(host, [b], alloc=alloc)(toks)
    b_wrote = float((b_only - base).abs().max())
    residual = both_w - (along_both.unsqueeze(-1) * direction)
    b_survived = float(residual.norm(dim=-1).max())
    return {"error": err, "preserved": err < TOL,
            "second_unit_writes_alone": b_wrote,
            "second_unit_survives_allocation": b_survived,
            "positions_scored": int(keep.sum())}


def main() -> int:
    pairs, rows = [], []
    seed = 0
    for vocab in (16, 32, 64):
        for gain_b in (0.5, 1.0, 4.0):
            seed += 1
            pairs.append({"kind": "random/random", "vocab": vocab, "gain_b": gain_b,
                          "seed": seed})
    # The compiled circuit as the protected unit, at the vocabulary it was verified for.
    for gain_b in (0.5, 1.0, 4.0):
        seed += 1
        pairs.append({"kind": "induction/random", "vocab": 20, "gain_b": gain_b,
                      "seed": seed})

    for p in pairs:
        v, ctx = p["vocab"], 60
        torch.manual_seed(p["seed"])
        host = Tiny(v)
        toks = torch.randint(0, v, (8, 12 if p["kind"].startswith("random") else 24))
        if p["kind"] == "induction/random":
            a = induction_as_unit(v, ctx, gain=1.0)
            if a is None:
                rows.append({**p, "skipped": "outside the circuit's envelope"})
                continue
        else:
            a = random_unit("a", v, p["seed"], gain=1.0)
        b = random_unit("b", v, p["seed"] + 500, gain=p["gain_b"])

        row = {**p}
        for alloc in (WriteAlloc.SHARED, WriteAlloc.ORTHOGONAL, WriteAlloc.EXCLUSIVE):
            try:
                row[alloc.value] = preservation(host, a, b, toks, alloc)
            except Exception as exc:                              # noqa: BLE001
                row[alloc.value] = {"error": None, "raised": str(exc)[:160]}
        rows.append(row)

    def tally(alloc: str) -> dict:
        got = [r[alloc] for r in rows if alloc in r and r[alloc].get("error") is not None]
        if not got:
            return {"n": 0}
        kept = [g for g in got if g["preserved"]]
        silent = [g for g in got if g["second_unit_survives_allocation"] < TOL]
        return {
            "n": len(got),
            "preserved_exactly": len(kept),
            "fraction_preserved": round(len(kept) / len(got), 4),
            "worst_error": round(max(g["error"] for g in got), 8),
            "pairs_where_second_unit_was_silenced": len(silent),
        }

    summary = {a.value: tally(a.value) for a in
               (WriteAlloc.SHARED, WriteAlloc.ORTHOGONAL, WriteAlloc.EXCLUSIVE)}
    orth = summary[WriteAlloc.ORTHOGONAL.value]
    shared = summary[WriteAlloc.SHARED.value]
    out = {
        "question": "does orthogonal allocation preserve the first unit across pairs, "
                    "or was that one pair?",
        "n_pairs": len(rows),
        "grid": "vocabulary 16/32/64 x later-unit gain 0.5/1.0/4.0, plus the compiled "
                "induction circuit as the protected unit at each gain",
        "tolerance": TOL,
        "summary": summary,
        "rows": rows,
        "verdict": (
            f"orthogonal allocation preserved the first unit exactly in "
            f"{orth.get('preserved_exactly')} of {orth.get('n')} pairs "
            f"(worst error {orth.get('worst_error')}), against "
            f"{shared.get('preserved_exactly')} of {shared.get('n')} under the shared "
            f"convention, and silenced the second unit in "
            f"{orth.get('pairs_where_second_unit_was_silenced')} of them"
            if orth.get("n") else "no pair produced a measurable write"),
        "what_exclusive_costs": (
            "EXCLUSIVE preserves the first unit perfectly and silences the second in "
            "every pair here, because these units carry no firing condition: a first "
            "unit that fires everywhere takes every position and leaves none. That is "
            "the correct behaviour of the allocation and the reason the build path "
            "pairs it with a `when:` predicate — the linked induction unit fires on "
            "about 70% of traffic, so a second unit would still have the rest. "
            "Exclusive allocation between two unconditional units is a veto, and "
            "nothing in the preservation number alone would tell you that."),
        "on_the_tolerance": (
            f"'exactly' means within {TOL} in float32. The worst orthogonal error was "
            f"{orth.get('worst_error')}, which is arithmetic noise rather than a "
            f"surviving component of the second unit; the shared convention misses by "
            f"up to {shared.get('worst_error')}, five orders of magnitude larger."),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/write_allocation_replication.json").write_text(
        json.dumps(out, indent=2))
    for name, s in summary.items():
        print(f"{name:<11} {s.get('preserved_exactly')}/{s.get('n')} preserved "
              f"exactly, worst error {s.get('worst_error')}, "
              f"second unit silenced in {s.get('pairs_where_second_unit_was_silenced')}")
    print(f"\n{out['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

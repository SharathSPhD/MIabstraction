"""Linking a separately compiled unit into a build — the ABI, used rather than studied.

The project has an ABI (`loom.abi`), a linker that solves the write gain at link time
(`loom.linker`), a hand-compiled induction circuit that needs no training at all
(`loom.constructed`), and a write allocation whose exclusive mode preserves an
earlier unit's behaviour exactly. Until this module, none of them were called from a
build: they lived in `loom/experiments/`, which meant the compiler could describe an
abstraction layer it never used.

What a build gets here that it cannot get any other way:

  - a skill realized with **zero training**, because the circuit was compiled once
    and verified once, and linking it is arithmetic;
  - a **refusal to link** when the host is outside the circuit's verified envelope,
    which is the honest outcome and the one a fine-tuning fallback hides;
  - **separation on condition** — the unit writes only where its `when` predicate
    fires, so a host that never meets the unit's distribution never pays for it.

The scratch substrate is where this pays off, because there the compiler chooses the
vocabulary and context length, and can therefore build a host that fits the circuit's
envelope instead of discovering it does not.
"""
from __future__ import annotations

import time
from typing import Any

import torch

from ..abi import Envelope, Mode, ReadKind, Unit, WriteAlloc, WriteKind
from ..linker import LinkedModel, host_loss, solve_gain


def induction_unit(vocab: int, max_len: int, gain: float = 1.0) -> Unit | None:
    """The hand-compiled induction circuit as a linkable unit, or None if this host
    is outside the envelope it was verified on."""
    from ..constructed import MAX_LEN, MAX_VOCAB, compile_induction
    if vocab > MAX_VOCAB or max_len > MAX_LEN:
        return None
    core = compile_induction(vocab=vocab, max_len=max_len)

    def fires_where_the_context_repeats(host_logits, unit_logprobs, tokens):
        """The circuit is an induction head: it has something to say exactly where the
        current token has been seen before in this sequence. Asking that of the INPUT
        rather than of the unit's own confidence matters — a hand-built circuit with
        near-hard attention is confidently wrong off-distribution, so confidence
        cannot serve as its own validity signal."""
        b, L = tokens.shape
        mask = torch.zeros(b, L, device=tokens.device)
        for i in range(b):
            seen: dict[int, int] = {}
            for p in range(L):
                t = int(tokens[i, p])
                if t in seen:
                    mask[i, p] = 1.0
                seen[t] = p
        return mask

    return Unit(name="induction", core=core, mode=Mode.OUT_OF_BAND,
                read=ReadKind.TOKENS, write=WriteKind.LOGITS, gain=gain,
                when=fires_where_the_context_repeats,
                envelope=Envelope(vocab=MAX_VOCAB, max_len=MAX_LEN))


def link_skill(host: torch.nn.Module, tokens: torch.Tensor, vocab: int,
               max_len: int, device: str = "cpu",
               budget: float = 0.25) -> dict:
    """Link the induction circuit into this host, or say precisely why not.

    Returns a record for the build report: whether it linked, the gain the linker
    solved for, the firing rate on the host's own traffic, and what the host paid in
    nats. A refusal carries the envelope numbers that caused it — an unlinkable host
    is information about the host, not a failure to report.
    """
    t0 = time.time()
    unit = induction_unit(vocab, max_len)
    if unit is None:
        from ..constructed import MAX_LEN, MAX_VOCAB
        return {
            "linked": False, "unit": "induction",
            "reason": (f"this host is outside the circuit's verified envelope: "
                       f"vocabulary {vocab:,} against {MAX_VOCAB}, context "
                       f"{max_len} against {MAX_LEN}. The circuit is sound; it was "
                       "never verified at this scale, so it is not grafted."),
            "envelope": {"vocab_max": MAX_VOCAB, "len_max": MAX_LEN},
            "host": {"vocab": vocab, "max_len": max_len},
            "wall_clock_s": round(time.time() - t0, 2),
        }

    def gate(m) -> dict:
        """What the linked model must do: continue a repeated context correctly.
        Measured on the host's own tokens, not on the circuit's training data."""
        with torch.no_grad():
            logits = m(tokens[:, :-1].to(device))
        pred = logits.argmax(-1).cpu()
        tgt = tokens[:, 1:]
        # Score only positions whose token has appeared before — the ones induction
        # is for. Scoring everywhere would measure the host and call it the unit.
        hits = total = 0
        for i in range(tokens.shape[0]):
            seen = set()
            for p in range(tokens.shape[1] - 1):
                t = int(tokens[i, p])
                if t in seen:
                    hits += int(pred[i, p] == tgt[i, p])
                    total += 1
                seen.add(t)
        acc = hits / max(total, 1)
        return {"passed": acc > 0.30, "icl_acc": acc, "scored_positions": total}

    base = host_loss(host, tokens, device)
    alone = gate(host)
    gain, gate_report = solve_gain(host, unit, tokens, gate, device, budget)
    unit.gain = gain

    lm = LinkedModel(host, [unit], device=device, alloc=WriteAlloc.EXCLUSIVE)
    after_gate = gate(lm)
    after_loss = host_loss(lm, tokens, device)

    return {
        "linked": bool(gate_report.get("passed")),
        "unit": "induction",
        "how": "compiled once, verified once, grafted with no training at all",
        "gain_solved_at_link_time": round(float(gain), 5),
        "firing_rate_on_host_traffic": round(
            float(lm.firing_rate.get("induction", 1.0)), 4),
        "write_allocation": WriteAlloc.EXCLUSIVE.value,
        "host_before": round(base, 4),
        "host_after": round(after_loss, 4),
        "host_paid_nats": round(after_loss - base, 4),
        "budget_nats": budget,
        "skill_alone": round(alone["icl_acc"], 4),
        "skill_linked": round(after_gate["icl_acc"], 4),
        "scored_positions": after_gate["scored_positions"],
        "gate": {k: v for k, v in gate_report.items() if k != "trace"},
        "wall_clock_s": round(time.time() - t0, 2),
    }

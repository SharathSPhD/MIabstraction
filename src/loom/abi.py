"""The Loom ABI — a calling convention for transformer weights.

C's power was never its syntax; it was separate compilation plus a stable ABI plus a
linker. A library is compiled once, verified once, and linked into any program. Doing
that for transformers runs into a contradiction: to link and verify a circuit you must
give it private, non-clobbered space in the residual stream, but the residual stream's
capacity comes precisely from superposition — packing more features than dimensions.
Reserving dimensions taxes the host.

TRIZ resolves this as a *physical* contradiction by separation in space and condition
(Principle 24, Intermediary; Principle 3, Local Quality): **a linked unit does not live
inside the host's residual stream at all**. It brings its own address space and touches
the host only through a declared interface:

    read  (arguments)     tokens            | residual @ layer L, projected by R
    core  (private frame) the unit's own weights, its own dimensions
    write (return value)  logits += g * U   | residual @ layer L' += g * W u

So the host's capacity cost is not `k` reserved dimensions; it is the *rank of the
write*, and (with `when:` conditions) only at the steps where the unit fires. That is
the out-of-band ABI.

Two modes exist so the claim is testable rather than asserted:

  OUT_OF_BAND  the unit computes privately; nothing in the host is reserved.
  IN_BAND      the classical alternative: `reserved_dims` of the host residual are
               given to the unit and denied to the host. Kept precisely so the
               capacity cost of reservation can be *measured* against out-of-band
               (experiment L3), not assumed.

A unit is verification-carrying: it ships with the gates it passed and the envelope
(distribution, sequence lengths, vocab) those gates were measured on. Linking outside
the envelope is outside the promise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import torch
import torch.nn as nn


class Mode(str, Enum):
    OUT_OF_BAND = "out_of_band"
    IN_BAND = "in_band"


class ReadKind(str, Enum):
    TOKENS = "tokens"          # the unit is a function of the input ids
    RESIDUAL = "residual"      # the unit reads the host residual at read_layer


class WriteAlloc(str, Enum):
    """How the writes of several linked units share one output.

    SHARED is the naive convention: every unit adds its contribution and they
    collide — measured, a unit scoring 0.588 alone fell to 0.447 beside another.
    ORTHOGONAL is the allocation this project needed to make separate compilation
    real: at each position the units are taken in their declared order and each
    write is projected onto the orthogonal complement of the writes before it, so an
    earlier unit's own component is preserved exactly. It is the linker's version of
    what the constructed backend gets for free by putting two skills in disjoint
    coordinate blocks.

    EXCLUSIVE is what the measurement forced. Orthogonalizing the writes barely
    moved the composition (induction 0.447 -> 0.454; the other unit slightly worse),
    because behaviour is decided by argmax and an orthogonal addition can still
    change which coordinate is largest. Preserving a projection is not preserving a
    prediction. What the constructed backend actually had was not orthogonal writes
    but DISJOINT SUPPORT: the second skill's coordinates were identically zero on
    the first skill's traffic, so there was nothing to interfere with. The linker's
    analogue is mutual exclusivity in firing: a later unit writes only at positions
    where every earlier unit stayed silent, which needs the `when` conditions the
    ABI already carries.
    """
    SHARED = "shared"
    ORTHOGONAL = "orthogonal"
    EXCLUSIVE = "exclusive"


class WriteKind(str, Enum):
    LOGITS = "logits"          # contributes to the output distribution
    RESIDUAL = "residual"      # writes back into the host residual at write_layer


class ABIError(RuntimeError):
    """A link that would violate the calling convention."""


@dataclass
class Envelope:
    """What the unit's gates were measured on. Outside this, the promise is void."""
    vocab: int
    max_len: int
    notes: str = ""

    def check(self, vocab: int, seq_len: int) -> list[str]:
        out = []
        if vocab > self.vocab:
            out.append(f"host vocab {vocab} exceeds unit envelope {self.vocab}")
        if seq_len > self.max_len:
            out.append(f"sequence length {seq_len} exceeds unit envelope {self.max_len}")
        return out


@dataclass
class Unit:
    """A separately compiled, verification-carrying unit of transformer program."""
    name: str
    core: nn.Module
    read: ReadKind
    write: WriteKind
    mode: Mode = Mode.OUT_OF_BAND
    read_layer: int = -1
    write_layer: int = -1
    gain: float = 1.0
    # Out-of-band write map: (host_width <- unit_width). None => identity/logit-aligned.
    write_proj: torch.Tensor | None = None
    read_proj: torch.Tensor | None = None
    # In-band only: the host dimensions this unit claims.
    reserved_dims: tuple[int, int] | None = None
    gates: dict[str, Any] = field(default_factory=dict)
    envelope: Envelope | None = None
    # Optional firing condition — the ABI's "separation on condition". Receives
    # (host_logits, unit_contribution) and returns a per-position mask. A unit that
    # fires unconditionally taxes the host on every token, including the ones it knows
    # nothing about; gating on the unit's own confidence is the cheap, honest default.
    when: Callable[..., torch.Tensor] | None = None   # (host_logits, unit_logprobs, tokens)

    def write_rank(self, host_width: int) -> int:
        """Host capacity actually consumed by this unit's write."""
        if self.mode is Mode.IN_BAND and self.reserved_dims:
            lo, hi = self.reserved_dims
            return hi - lo
        if self.write is WriteKind.LOGITS:
            return 0  # writes to the output, not to the stream
        if self.write_proj is not None:
            return int(torch.linalg.matrix_rank(self.write_proj.float()).item())
        return host_width


def check_no_clobber(units: list[Unit]) -> None:
    """The ABI's core guarantee: no two units may claim the same host dimensions.

    Out-of-band units claim none, so they never collide — which is the point of the
    convention. In-band units are checked for overlap and refused if they collide.
    """
    claims: list[tuple[str, int, int]] = []
    for u in units:
        if u.mode is Mode.IN_BAND:
            if u.reserved_dims is None:
                raise ABIError(
                    f"Unit '{u.name}' is in-band but declares no reserved_dims; "
                    "an in-band unit must state which host dimensions it claims."
                )
            claims.append((u.name, *u.reserved_dims))
    for i, (na, lo_a, hi_a) in enumerate(claims):
        for nb, lo_b, hi_b in claims[i + 1:]:
            if lo_a < hi_b and lo_b < hi_a:
                raise ABIError(
                    f"Units '{na}' [{lo_a}:{hi_a}) and '{nb}' [{lo_b}:{hi_b}) claim "
                    "overlapping host dimensions. Separate compilation requires "
                    "disjoint frames; re-allocate or link them out-of-band."
                )


def allocate_frames(units: list[Unit], host_width: int, widths: dict[str, int]) -> None:
    """Assign disjoint in-band frames (a stack-frame allocator for the residual stream)."""
    cursor = 0
    for u in units:
        if u.mode is not Mode.IN_BAND:
            continue
        w = widths[u.name]
        if cursor + w > host_width:
            raise ABIError(
                f"Cannot allocate {w} dimensions for '{u.name}': only "
                f"{host_width - cursor} of {host_width} remain. The host is out of "
                "in-band capacity — link out-of-band instead."
            )
        u.reserved_dims = (cursor, cursor + w)
        cursor += w
    check_no_clobber(units)

"""What a linear write has been measured to deliver, consulted before lowering to one.

The steering-capacity ledger (`results/steering_capacity.json`, derived from a build
report by `scripts/steering_capacity.py`) is the compiler's only evidence about the
ceiling of activation steering: demand spanned 23x across capabilities and what one
write delivered stayed in a band of 0.004-0.018 nats. This module turns that ledger
into two decisions:

  - whether to search steering at all: a capability whose target exceeds the most a
    write has EVER delivered here is not lowered to a control, and the refusal cites
    the measurement rather than a constant;
  - what the escalation search looks like when training replaces steering: a space
    ordered so its cost grows with the gap instead of a fixed 30 steps.

No number in this file is chosen by hand. When the ledger is absent the compiler has
no prior and must measure, which is why `should_skip_steering` refuses to skip without
one — a guess dressed as a measurement is worse than the search it would save.
"""
from __future__ import annotations

import json
from pathlib import Path

from .search import Lever

LEDGER = Path("results/steering_capacity.json")


def delivery_ceiling(path: Path | str = LEDGER,
                     base_model: str | None = None) -> tuple[float | None, str]:
    """The most a searched steering control has ever delivered, in nats, with where
    that number came from. (None, reason) when there is no measured ledger.

    The ceiling is substrate-specific and this check is load-bearing: measured on
    Llama-3.2-1B the ceiling was 0.0179 nats, while the same search on Qwen2.5-1.5B
    delivered 0.0266 and met its target — a ceiling carried across families would have
    skipped a search that wins. So a ledger measured on a different base model is not
    evidence here, and the compiler searches instead.
    """
    p = Path(path)
    if not p.exists():
        return None, f"no measured steering-capacity ledger at {p}"
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return None, f"no measured steering-capacity ledger: {p} unreadable ({e})"
    measured_on = data.get("base_model")
    if base_model is not None and measured_on != base_model:
        return None, (f"the ledger at {p} was measured on {measured_on}, which is "
                      f"not evidence about {base_model}")
    rows = data.get("capabilities", [])
    delivered = [r["delivered_nats"] for r in rows
                 if isinstance(r.get("delivered_nats"), (int, float))]
    if not delivered:
        return None, f"no measured steering-capacity ledger: {p} has no delivery rows"
    top = max(delivered)
    return top, (f"{p}: max delivered {top:g} nats across "
                 f"{len(delivered)} measured capabilities")


def should_skip_steering(gap: float, recover: float, ceiling: float | None,
                         provenance: str) -> tuple[bool, str]:
    """Is this capability's target beyond what a linear write has ever delivered?

    The target is `gap * recover` — the nats the program demands the control produce.
    Skipping is only permitted on evidence: no ledger, or no measured demand, means the
    search runs and produces the measurement itself.
    """
    target = gap * recover
    if ceiling is None or gap <= 0 or target <= 0:
        return False, ""
    if target > ceiling:
        return True, (f"not lowered to steering: the target of {round(target, 4):g} nats "
                      f"({recover:g} of a {gap:g}-nat gap) exceeds the most a linear "
                      f"write has ever delivered here ({provenance})")
    return False, ""


# `escalation_levers` lived here: the adaptation space a build walked when steering
# could not reach a declared target. The only capability that ever escalated was
# refusal, and refusal is no longer compiled into weights, so the space had no
# caller. It is deleted rather than kept warm — see git history if a capability ever
# needs a training fallback again.

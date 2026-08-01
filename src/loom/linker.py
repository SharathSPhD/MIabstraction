"""The Loom linker — graft separately compiled units into a host model.

A linker's classical jobs are relocation (fix addresses so a library works in this
program), symbol resolution (connect the library's needs to what the program provides),
and verification (refuse a link that cannot work). This linker does all three
numerically:

  relocation        solve the write gain `g` at link time — the smallest write that
                    still passes the unit's gates, subject to the host's loss budget
                    (TRIZ Principle 23, Feedback: measure the output, adjust the input)
  symbol resolution match the unit's declared read (tokens / residual@L) to the host
  verification      refuse the link unless BOTH hold: the unit's gates still pass
                    post-link, and the host degrades by less than its budget

The falsifiable claims this module exists to test:

  L1  no-clobber   linking a verified unit into a trained host preserves the host's
                   loss within budget AND the unit's gates still pass.
  L2  composition  two independently compiled units linked into one host both keep
                   their gates — the actual "separate compilation" claim.
  L3  capacity     out-of-band linking costs the host strictly less than in-band
                   dimension reservation at matched unit performance. This is the
                   TRIZ resolution, measured rather than asserted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .abi import Mode, ReadKind, Unit, WriteKind, check_no_clobber


@dataclass
class LinkReport:
    linked: bool
    units: list[str]
    host_loss_before: float
    host_loss_after: float
    host_delta: float
    budget: float
    unit_gates: dict[str, dict] = field(default_factory=dict)
    gains: dict[str, float] = field(default_factory=dict)
    capacity_cost: dict[str, int] = field(default_factory=dict)
    envelope_warnings: list[str] = field(default_factory=list)
    diagnosis: str = ""

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


class LinkedModel(torch.nn.Module):
    """Host model with units linked in through the ABI. The host is never retrained."""

    def __init__(self, host: torch.nn.Module, units: list[Unit], device: str = "cpu"):
        super().__init__()
        check_no_clobber(units)
        self.host = host
        self.units = units
        self.device_ = device
        self.enabled = {u.name: True for u in units}
        # Fraction of positions where each unit actually fires — the envelope signal.
        # A unit verified on one distribution that fires everywhere on the host's is
        # out of its envelope, whatever the vocab/length check says.
        self.firing_rate: dict[str, float] = {}
        for u in units:
            u.core.to(device).eval()

    def _unit_logits(self, u: Unit, tokens: torch.Tensor) -> torch.Tensor:
        """Run a token-reading, logit-writing unit and align its vocab to the host."""
        out = u.core(tokens)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        host_out = self.host(tokens)
        logits = host_out[0] if isinstance(host_out, tuple) else host_out

        for u in self.units:
            if not self.enabled[u.name]:
                continue
            if u.read is ReadKind.TOKENS and u.write is WriteKind.LOGITS:
                ulog = self._unit_logits(u, tokens)
                v = min(ulog.shape[-1], logits.shape[-1])
                contrib = torch.zeros_like(logits)
                # The return-value convention: a unit contributes CENTERED log-odds.
                # Raw log-softmax carries a large negative mean (a confident unit sits
                # near -20 on every token it dislikes), which swamps the host's logits
                # and made early links cost +10 nats at gain 0.5. Adding a constant to
                # all logits is a no-op, so only the shape of the unit's opinion is
                # meaningful; centering transmits exactly that and nothing else.
                ls = F.log_softmax(ulog[..., :v].float(), dim=-1)
                contrib[..., :v] = ls - ls.mean(dim=-1, keepdim=True)
                if u.when is not None:
                    # Separation on condition: a unit that speaks when it has nothing
                    # to say costs the host on every token. `when` receives the host's
                    # logits and the unit's UNCENTERED log-probs — centering shifts
                    # every value upward, so a confidence test on the centered write
                    # is always true and the gate silently never fires. It also
                    # receives the tokens, because the honest applicability test is
                    # usually about the INPUT ("has this context occurred before?"),
                    # not the output: a hand-built circuit with near-hard attention is
                    # confidently wrong off-distribution, so confidence cannot serve
                    # as its own validity signal.
                    mask = u.when(logits, ls, tokens).to(contrib.dtype)
                    self.firing_rate[u.name] = float(mask.mean())
                    contrib = contrib * mask.unsqueeze(-1)
                else:
                    self.firing_rate[u.name] = 1.0
                logits = logits + u.gain * contrib
            else:
                raise NotImplementedError(
                    f"Unit '{u.name}': read={u.read} write={u.write} is not yet "
                    "supported by the linker (v1 supports tokens -> logits)."
                )
        return logits

    def set_enabled(self, name: str, on: bool) -> None:
        self.enabled[name] = on


# ---------------------------------------------------------------- measurement

@torch.no_grad()
def host_loss(model: torch.nn.Module, tokens: torch.Tensor, device: str,
              batch: int = 64) -> float:
    """Mean next-token CE on the host's own distribution (the thing we must not break)."""
    model.eval()
    total, n = 0.0, 0
    for i in range(0, tokens.shape[0], batch):
        b = tokens[i : i + batch].to(device)
        out = model(b[:, :-1])
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), b[:, 1:].reshape(-1)
        )
        total += float(loss) * b.shape[0]
        n += b.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def in_band_cost(host: torch.nn.Module, tokens: torch.Tensor, device: str,
                 n_dims: int, layer: int = -1) -> float:
    """Host loss when `n_dims` residual dimensions are RESERVED (denied to the host).

    This is the honest price of the classical in-band convention: a reserved frame is
    capacity the host no longer has. Emulated by zeroing those dimensions after the
    given block, which is exactly what "these dims belong to the unit" means for the
    host's own computation.
    """
    blocks = host.blocks
    idx = layer if layer >= 0 else len(blocks) + layer

    def hook(module, args, output):
        out = output.clone()
        out[..., :n_dims] = 0.0
        return out

    h = blocks[idx].register_forward_hook(hook)
    try:
        return host_loss(host, tokens, device)
    finally:
        h.remove()


# ---------------------------------------------------------------- relocation

def solve_gain(
    host: torch.nn.Module,
    unit: Unit,
    host_tokens: torch.Tensor,
    unit_gate_fn,
    device: str,
    budget: float,
    lo: float = 1e-3,
    hi: float = 8.0,
    steps: int = 16,
) -> tuple[float, dict]:
    """Link-time relocation: smallest gain whose unit-gate passes inside the host budget.

    Feedback loop (TRIZ 23): sweep the write gain, measure both sides, and keep the
    smallest gain that satisfies the unit — a larger write is never free, since every
    increment costs the host.
    """
    base = host_loss(host, host_tokens, device)
    best: tuple[float, dict] | None = None
    trace = []
    # Log-spaced: the feasible window is often two orders of magnitude below 1, and a
    # linear grid from 0 to 8 never looks there (it reported "no gain works" while a
    # working gain sat at 0.02).
    for i in range(steps):
        g = lo * (hi / lo) ** (i / (steps - 1))
        unit.gain = g
        lm = LinkedModel(host, [unit], device=device)
        gate = unit_gate_fn(lm)
        hl = host_loss(lm, host_tokens, device)
        delta = hl - base
        trace.append({"gain": round(g, 3), "host_delta": round(delta, 4),
                      **{k: round(float(v), 4) for k, v in gate.items()
                         if isinstance(v, (int, float))}})
        if gate.get("passed") and delta <= budget and best is None:
            best = (g, gate)
            break
    if best is None:
        unit.gain = 0.0
        return 0.0, {"passed": False, "trace": trace}
    unit.gain = best[0]
    return best[0], {"passed": True, "trace": trace, **best[1]}


def link(
    host: torch.nn.Module,
    units: list[Unit],
    host_tokens: torch.Tensor,
    unit_gate_fns: dict,
    device: str = "cpu",
    budget: float = 0.05,
) -> tuple[LinkedModel | None, LinkReport]:
    """Link units into a host, verifying both sides. Refuses rather than degrade."""
    check_no_clobber(units)
    base = host_loss(host, host_tokens, device)
    warnings: list[str] = []
    vocab = getattr(getattr(host, "head", None), "out_features", None) or 0
    for u in units:
        if u.envelope is not None:
            warnings += [f"{u.name}: {w}" for w in
                         u.envelope.check(vocab, host_tokens.shape[1])]

    gains, gates = {}, {}
    for u in units:
        g, gate = solve_gain(host, u, host_tokens, unit_gate_fns[u.name], device, budget)
        gains[u.name] = g
        gates[u.name] = gate

    lm = LinkedModel(host, units, device=device)
    after = host_loss(lm, host_tokens, device)
    delta = after - base
    all_gates = all(gates[u.name].get("passed") for u in units)
    ok = all_gates and delta <= budget and math.isfinite(delta)

    if ok:
        diagnosis = ""
    elif not all_gates:
        failed = [n for n, g in gates.items() if not g.get("passed")]
        diagnosis = (
            f"Link refused: no write gain lets {failed} meet their gates without "
            f"costing the host more than {budget:.3g} nats. The unit works alone but "
            "cannot be heard over this host — check the envelope (vocab/length) matches "
            "what the unit was verified on."
        )
    else:
        diagnosis = (
            f"Link refused: the host's loss rose by {delta:.4g} nats, over its "
            f"{budget:.3g} budget. The units are too loud for this host."
        )

    return (lm if ok else None), LinkReport(
        linked=ok,
        units=[u.name for u in units],
        host_loss_before=base,
        host_loss_after=after,
        host_delta=delta,
        budget=budget,
        unit_gates={k: {kk: vv for kk, vv in v.items() if kk != "trace"}
                    for k, v in gates.items()},
        gains=gains,
        capacity_cost={u.name: u.write_rank(getattr(host, "tok").embedding_dim)
                       for u in units},
        envelope_warnings=warnings,
        diagnosis=diagnosis,
    )

"""The Loom ISA — the instruction set the compiler lowers a program into.

A programming language is not its syntax; it is the set of operations its compiler
can actually emit, the operands those operations take, and the machine model that
says which of them a given target permits. C has that. Until this module, Loom's
version of it existed only implicitly, scattered across the lowering catalogue, the
design space and the ABI — which meant it could be described but not *read*.

Nothing here is written by hand. The instruction table is derived from the same
`CATALOGUE` the compiler consults when it lowers a capability, the operand table
from the same `DESIGN_SPACE` the autotuner searches, and the target descriptions
from the same `SubstrateProfile`s the selector checks. If the compiler learns a new
instruction, this spec gains it in the same commit — the alternative is a manual
that lies.

Three levels, exactly as the compiler walks them:

  L2  capability graph   what the program asked for, in kinds
  L1  mech-interp IR     THE INSTRUCTION SET — reads and writes on internal
                         objects this project measured to be real (features,
                         circuits, directions), plus the escape hatch to training
  L0  substrate ops      what the hardware/library layer is asked to do

Emit with:  python -m loom.isa            (human)
            python -m loom.isa --json     (the app renders this)
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from .app.capability import Kind
from .app.design_space import DESIGN_SPACE, KNOB_DOMAINS
from .app.lowering import CATALOGUE
from .app.substrate import PROFILES


# --------------------------------------------------------------------- opcodes

@dataclass
class Instruction:
    """One L1 operation the compiler can emit, with what it acts on."""
    op: str                      # read | amplify | suppress | install | monitor
    target: str                  # feature(style), circuit(induction), ...
    emitted_by: list[str]        # strategies that lower to it
    for_kinds: list[str]         # capability kinds that can reach it
    operands: list[str]          # searchable levers this instruction takes
    verified_by: str             # which experiment licenses using it at all
    note: str = ""


@dataclass
class SubstrateOp:
    """One L0 operation — what the instruction becomes on a real machine."""
    op: str
    emitted_by: list[str]
    requires_lever: list[str]


@dataclass
class Operand:
    """A searchable operand: the compiler chooses its value by measurement."""
    name: str
    stage: str
    range: tuple[float, float]
    spacing: str
    bounded_by: str | None
    why: str


# Which measured result licenses each instruction. An instruction whose object was
# never shown to exist is a wish, not an opcode — this column is the difference
# between an abstraction layer and a naming scheme.
LICENCE = {
    "read": ("E1/E4 — a concept is linearly readable from the residual stream, and "
             "the honest measure is what a probe adds beyond a trivial baseline "
             "(incremental R² 0.031 trained vs 0.0003 untrained)"),
    "amplify": ("steering-capacity ledger — a linear write delivers a measured, "
                "substrate-specific band of nats; the compiler consults it before "
                "emitting this instruction at all"),
    "suppress": ("same ledger, negative direction; bounded by the output-variety "
                 "guard, which rejects any dose that damages the model"),
    "install": ("E2 + the constructed backend — induction exists as a circuit and "
                "can be written into weights directly, verified layer by layer "
                "against ground truth (prefix score 0.979, zero training)"),
    "monitor": ("E1/E4 — a probe that reads a property, with the untrained-model "
                "control reported beside it"),
    "train": ("no mech-interp claim: the escape hatch. Reached when the measured "
              "gap exceeds what a write has been shown to deliver"),
}

_OP_RE = re.compile(r"^(\w+)\((.*)\)$")


def instructions() -> list[Instruction]:
    """The instruction set, read out of the lowering catalogue."""
    by_key: dict[tuple[str, str], Instruction] = {}
    steer_levers = [lv.name for lv in DESIGN_SPACE if lv.stage == "steering"]
    train_levers = [lv.name for lv in DESIGN_SPACE if lv.stage in
                    ("adaptation", "pretraining")]

    for kind, strategies in CATALOGUE.items():
        for st in strategies:
            for raw in st.mech_ops:
                m = _OP_RE.match(raw)
                op, target = (m.group(1), m.group(2)) if m else (raw, "")
                key = (op, target.split(",")[0].strip())
                inst = by_key.get(key)
                if inst is None:
                    inst = Instruction(
                        op=op, target=key[1] or target,
                        emitted_by=[], for_kinds=[],
                        operands=(steer_levers if op in ("amplify", "suppress")
                                  else [] if op in ("read", "monitor")
                                  else ["circuit envelope (vocab, length)"]),
                        verified_by=LICENCE.get(op, "—"))
                    by_key[key] = inst
                if st.name not in inst.emitted_by:
                    inst.emitted_by.append(st.name)
                if kind.value not in inst.for_kinds:
                    inst.for_kinds.append(kind.value)

            if not st.mech_ops and st.substrate_ops:
                key = ("train", st.substrate_ops[0])
                inst = by_key.setdefault(key, Instruction(
                    op="train", target=st.substrate_ops[0], emitted_by=[],
                    for_kinds=[], operands=train_levers,
                    verified_by=LICENCE["train"],
                    note="the compiler reaches this only after measuring that the "
                         "mech-interp instruction cannot meet the declared target"))
                if st.name not in inst.emitted_by:
                    inst.emitted_by.append(st.name)
                if kind.value not in inst.for_kinds:
                    inst.for_kinds.append(kind.value)
    return sorted(by_key.values(), key=lambda i: (i.op, i.target))


def substrate_ops() -> list[SubstrateOp]:
    out: dict[str, SubstrateOp] = {}
    for strategies in CATALOGUE.values():
        for st in strategies:
            for op in st.substrate_ops:
                s = out.setdefault(op, SubstrateOp(op, [], list(st.requires)))
                if st.name not in s.emitted_by:
                    s.emitted_by.append(st.name)
    return sorted(out.values(), key=lambda s: s.op)


def operands() -> list[Operand]:
    return [Operand(lv.name, lv.stage, (lv.low, lv.high), lv.scale, lv.knob, lv.why)
            for lv in DESIGN_SPACE]


def targets() -> dict:
    """The machine model: which instructions each substrate permits, and why not."""
    out = {}
    for name, prof in PROFILES.items():
        out[name] = {
            "family": prof.family, "notes": prof.notes,
            "levers": {k: {"available": lv.available, "cost": lv.cost,
                           "note": lv.note}
                       for k, lv in prof.levers.items()},
        }
    return out


def lowering_table() -> dict:
    """For each capability kind, the strategies in preference order — the compiler's
    own ranking, which is what makes a build explicable rather than magical."""
    return {kind.value: [{"strategy": st.name, "requires": st.requires,
                          "mech_ops": st.mech_ops,
                          "substrate_ops": st.substrate_ops,
                          "rationale": st.rationale}
                         for st in strategies]
            for kind, strategies in CATALOGUE.items()}


def spec() -> dict:
    return {
        "levels": [
            {"id": "L3", "name": "Program",
             "what": "clauses in consequences: knows / speaks / always / never / "
                     "refuses / expect, plus tune and effort"},
            {"id": "L2", "name": "Capability graph",
             "what": "kinds the compiler reasons about",
             "kinds": [k.value for k in Kind]},
            {"id": "L1", "name": "Mech-interp IR (the instruction set)",
             "what": "reads and writes on internal objects measured to be real"},
            {"id": "L0", "name": "Substrate operations",
             "what": "what a machine is actually asked to do"},
        ],
        "instructions": [asdict(i) for i in instructions()],
        "substrate_ops": [asdict(s) for s in substrate_ops()],
        "operands": [asdict(o) for o in operands()],
        "knobs": {k: {"domain": v,
                      "meaning": "a position on a dial the program may narrow"}
                  for k, v in KNOB_DOMAINS.items()},
        "targets": targets(),
        "lowering": lowering_table(),
        "note": ("Generated from the compiler's own tables (lowering.CATALOGUE, "
                 "design_space.DESIGN_SPACE, substrate.PROFILES). A manual written "
                 "by hand drifts from the compiler; this one cannot."),
    }


def render() -> str:
    s = spec()
    out = ["LOOM ISA", "=" * 60, ""]
    for lvl in s["levels"]:
        out.append(f"{lvl['id']}  {lvl['name']}: {lvl['what']}")
    out += ["", "INSTRUCTIONS (L1)", "-" * 60]
    for i in s["instructions"]:
        out.append(f"  {i['op']}({i['target']})")
        out.append(f"      kinds     {', '.join(i['for_kinds'])}")
        out.append(f"      operands  {', '.join(i['operands']) or '—'}")
        out.append(f"      licensed  {i['verified_by'][:100]}")
    out += ["", "OPERANDS (searched, never declared)", "-" * 60]
    for o in s["operands"]:
        bound = f"bounded by `tune {o['bounded_by']}`" if o["bounded_by"] else "compiler's own"
        out.append(f"  {o['name']:<11} {o['stage']:<12} "
                   f"[{o['range'][0]:g}, {o['range'][1]:g}] {o['spacing']:<14} {bound}")
    out += ["", "TARGETS", "-" * 60]
    for name, t in s["targets"].items():
        avail = [k for k, v in t["levers"].items() if v["available"]]
        out.append(f"  {name} ({t['family']}): {', '.join(avail)}")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    print(json.dumps(spec(), indent=2) if "--json" in sys.argv else render())

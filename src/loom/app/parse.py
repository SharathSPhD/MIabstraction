"""L3 — parse an application program.

The syntax is deliberately close to how someone describes what they want out loud, because
the person writing it is not an AI engineer. Every clause maps to exactly one capability,
so there is no hidden magic between what they wrote and what the compiler builds.

    app Tutor {
        knows   facts from "corpus/*.txt";
        knows   how to explain step by step;
        speaks  patient, concise;
        always  cites the source it used;
        never   discusses pricing;
        refuses topics outside its corpus;
        expect  answers("what is X?") mentions "definition";
        expect  refuses("what is your pricing?");
    }

    build Tutor on scratch(size = small);
    build Tutor on "Qwen/Qwen3-0.6B";
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .capability import App, Capability, Expectation, Kind


class AppSyntaxError(SyntaxError):
    """A program that cannot be read, reported with the line and what was expected."""


@dataclass
class BuildTarget:
    app: str
    spec: dict
    line: int


@dataclass
class Program:
    apps: dict[str, App] = field(default_factory=dict)
    builds: list[BuildTarget] = field(default_factory=list)


_APP = re.compile(r"^\s*app\s+(\w+)\s*\{\s*$")
_END = re.compile(r"^\s*\}\s*$")
_BUILD = re.compile(r"^\s*build\s+(\w+)\s+on\s+(.+?)\s*;\s*$")
_KNOWS_FROM = re.compile(r"^knows\s+(?:facts\s+)?from\s+\"([^\"]+)\"$")
_KNOWS_HOW = re.compile(r"^knows\s+how\s+to\s+(.+)$")
_SPEAKS = re.compile(r"^speaks\s+(.+)$")
_ALWAYS = re.compile(r"^always\s+(.+)$")
_NEVER = re.compile(r"^never\s+(.+)$")
_REFUSES = re.compile(r"^refuses\s+(.+)$")
_EXP_ANS = re.compile(r"^expect\s+answers\(\"([^\"]*)\"\)\s+mentions\s+\"([^\"]*)\"$")
_EXP_REF = re.compile(r"^expect\s+refuses\(\"([^\"]*)\"\)$")
_EXP_KNOWS = re.compile(
    r"^expect\s+knows\s+(?:the\s+)?material\s+better\s+than\s+the\s+base\s+model$")
_SCRATCH = re.compile(r"^scratch\s*\(([^)]*)\)$")
_TUNE = re.compile(r"^tune\s+(\w+)\s+(?:from|between)\s+(.+)$")
_EFFORT = re.compile(r"^effort\s+(quick|balanced|thorough)$")


def _target_spec(text: str, line: int) -> dict:
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        return {"kind": "load", "name": text[1:-1]}
    m = _SCRATCH.match(text)
    if m:
        spec: dict = {"kind": "scratch"}
        for part in filter(None, (p.strip() for p in m.group(1).split(","))):
            if "=" not in part:
                raise AppSyntaxError(
                    f"line {line}: '{part}' in scratch(...) should look like key = value")
            k, v = (s.strip() for s in part.split("=", 1))
            spec[k] = v
        return spec
    raise AppSyntaxError(
        f"line {line}: don't understand the build target {text!r}. Write either "
        'scratch(size = small) to build a model, or "org/model" to start from an '
        "existing one.")


def parse_program(path: str | Path) -> Program:
    lines = Path(path).read_text().split("\n")
    prog = Program()
    current: App | None = None

    for i, raw in enumerate(lines, start=1):
        line = raw.split("//")[0].rstrip()
        if not line.strip():
            continue

        if current is None:
            m = _APP.match(line)
            if m:
                current = App(name=m.group(1))
                prog.apps[current.name] = current
                continue
            m = _BUILD.match(line)
            if m:
                prog.builds.append(
                    BuildTarget(app=m.group(1), spec=_target_spec(m.group(2), i), line=i))
                continue
            raise AppSyntaxError(
                f"line {i}: expected 'app <Name> {{' or 'build <Name> on <target>;', "
                f"found {line.strip()!r}")

        if _END.match(line):
            current = None
            continue

        body = line.strip().rstrip(";").strip()
        cap = _clause(body, i)
        if isinstance(cap, Capability):
            current.capabilities.append(cap)
        elif isinstance(cap, Expectation):
            current.expectations.append(cap)

    if current is not None:
        raise AppSyntaxError(f"app '{current.name}' is missing its closing '}}'")
    for b in prog.builds:
        if b.app not in prog.apps:
            raise AppSyntaxError(
                f"line {b.line}: build refers to '{b.app}', which is not defined. "
                f"Defined apps: {sorted(prog.apps) or 'none'}")
    return prog


def _tune_values(text: str):
    """Parse a declared search range: `1 to 8`, or `light, medium, heavy`."""
    text = text.strip()
    if " to " in text:
        lo, hi = (t.strip() for t in text.split(" to ", 1))
        return {"range": [float(lo), float(hi)]}
    return {"choices": [t.strip() for t in text.split(",") if t.strip()]}


def _clause(body: str, line: int) -> Capability | Expectation:
    if (m := _EFFORT.match(body)):
        # How hard the compiler should search. The programmer sets the budget, not the
        # hyperparameters: "thorough" means try more configurations, not "use lr=3e-5".
        return Capability(Kind.TUNING, "effort", {"effort": m.group(1)}, line)
    if (m := _TUNE.match(body)):
        # A bound on ONE named lever, in the programmer's terms. `tune adaptation from
        # 1 to 8` bounds how much the model may be changed; it never names a rank or a
        # learning rate, because those are the compiler's business.
        return Capability(Kind.TUNING, m.group(1), _tune_values(m.group(2)), line)
    if _EXP_KNOWS.match(body):
        # The one acceptance test a knowledge capability can actually be held to. Asking
        # a small model to recite a fact and grepping the answer for a keyword measures
        # luck; asking whether it predicts unseen material from the same source better
        # than it did before measures learning. This is what the person doing the job by
        # hand would look at, so it is what the language lets you ask for.
        return Expectation("knows_better", "held-out material from the app's corpus",
                           None, line)
    if (m := _EXP_ANS.match(body)):
        return Expectation("answers", m.group(1), m.group(2), line)
    if (m := _EXP_REF.match(body)):
        return Expectation("refuses", m.group(1), None, line)
    if (m := _KNOWS_FROM.match(body)):
        return Capability(Kind.KNOWLEDGE, "corpus", {"corpus": m.group(1)}, line)
    if (m := _KNOWS_HOW.match(body)):
        return Capability(Kind.SKILL, m.group(1), {}, line)
    if (m := _SPEAKS.match(body)):
        traits = [t.strip() for t in m.group(1).split(",") if t.strip()]
        return Capability(Kind.STYLE, "-".join(traits), {"traits": traits}, line)
    if (m := _ALWAYS.match(body)):
        return Capability(Kind.INVARIANT, m.group(1), {}, line)
    if (m := _NEVER.match(body)):
        return Capability(Kind.PROHIBITION, m.group(1), {}, line)
    if (m := _REFUSES.match(body)):
        return Capability(Kind.GUARDRAIL, m.group(1), {}, line)
    raise AppSyntaxError(
        f"line {line}: don't understand {body!r}. A clause starts with one of: "
        "knows / speaks / always / never / refuses / expect.")

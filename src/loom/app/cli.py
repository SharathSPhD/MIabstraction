"""The loom command.

    loom explain app.loom          what would be built, and how, on each target
    loom build   app.loom          compile, execute, verify, emit an artifact
    loom run     build/Tutor-...   talk to the model that came out

`explain` is the one to reach for first: it prints the plan without spending a GPU
second, so you can see what the compiler decided before it acts. That is the same reason
a C programmer reads the assembly before blaming the optimizer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .lowering import plan
from .parse import AppSyntaxError, parse_program
from .substrate import profile_for


def _target_name(spec: dict) -> str:
    return spec.get("name", f"scratch({spec.get('size', 'small')})")


def cmd_explain(path: str) -> int:
    try:
        prog = parse_program(path)
    except AppSyntaxError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    for build in prog.builds:
        app = prog.apps[build.app]
        sub = profile_for(build.spec)
        print(f"\n{app.name} on {_target_name(build.spec)}   [{sub.id} — {sub.family}]")
        print(f"  {sub.notes}\n")
        for ch in plan(app.to_realize(), sub):
            head = f"  {ch.capability.describe()}"
            print(f"{head:<46} -> {ch.strategy.name if ch.ok else 'NO STRATEGY'}")
            print(f"{'':<46}    {ch.reason}")
            for name, why in ch.rejected:
                print(f"{'':<46}    (not {name}: {why})")
        if app.expectations:
            print("\n  will be checked against:")
            for e in app.expectations:
                print(f"    - {e.describe()}")
    print()
    return 0


def cmd_build(path: str, out: str | None, dry: bool) -> int:
    try:
        prog = parse_program(path)
    except AppSyntaxError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    root = Path(out or "build")
    root.mkdir(parents=True, exist_ok=True)
    rc = 0
    for build in prog.builds:
        app = prog.apps[build.app]
        sub = profile_for(build.spec)
        choices = plan(app.to_realize(), sub)
        art = root / f"{app.name}-{sub.id}"
        art.mkdir(parents=True, exist_ok=True)
        (art / "plan.json").write_text(json.dumps({
            "app": app.name,
            "target": _target_name(build.spec),
            "substrate": sub.id,
            "capabilities": [c.to_dict() for c in choices],
            "expectations": [e.describe() for e in app.expectations],
        }, indent=2))
        print(f"{app.name} -> {art}  ({len(choices)} capabilities planned)")

        if dry:
            continue
        try:
            if sub.id == "open_weight":
                from .exec_open import execute_open
                report = execute_open(choices, build.spec, app, str(art))
            else:
                from .exec_scratch import execute_scratch
                report = execute_scratch(choices, build.spec, app, str(art))
        except ImportError as e:
            print(f"  executor unavailable ({e}); wrote the plan only. "
                  f"Use --dry-run to plan without building.", file=sys.stderr)
            rc = max(rc, 2)
            continue
        (art / "report.json").write_text(json.dumps(report, indent=2, default=str))
        ok = report.get("passed", False)
        print(f"  {'built' if ok else 'built with unmet expectations'}: {art}")
        rc = max(rc, 0 if ok else 1)
    return rc


def cmd_run(artifact: str) -> int:
    from .runtime import load_artifact
    model = load_artifact(artifact)
    print(f"loaded {artifact}. Ctrl-D to exit.\n")
    while True:
        try:
            q = input("> ")
        except EOFError:
            print()
            return 0
        if q.strip():
            print(model.respond(q))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="loom", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("explain", help="show what would be built, and how")
    p.add_argument("program")

    p = sub.add_parser("build", help="compile and build the model")
    p.add_argument("program")
    p.add_argument("--out", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="plan and write plan.json without spending GPU time")

    p = sub.add_parser("run", help="talk to a built model")
    p.add_argument("artifact")

    a = ap.parse_args(argv)
    if a.cmd == "explain":
        return cmd_explain(a.program)
    if a.cmd == "build":
        return cmd_build(a.program, a.out, a.dry_run)
    return cmd_run(a.artifact)


if __name__ == "__main__":
    raise SystemExit(main())

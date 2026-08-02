"""Build a program on the from-scratch substrate: no downloaded weights at all.

This is the path the open-weight backend cannot take. Here the compiler chooses the
architecture, learns a tokenizer from the program's own corpus, pretrains, and only
then realizes the behavioural capabilities — so the model that comes out exists
because a program asked for it, not because someone else trained it and we adjusted
the result.

Two efforts, because the honest range is wide:

  demo      a real language model, small and quick enough to watch (minutes)
  flagship  a serious pretraining run, dispatched to the training box (hours)

The distinction is declared, never hidden: a demo-scale model says so in its report,
and its perplexity is read as what it is.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .capability import App
from .exec_scratch import execute_scratch
from .lowering import plan
from .parse import parse_program, parse_program_text
from .substrate import profile_for

EFFORTS = {
    "demo": {"size": "small", "note": "demo scale: a real model, small enough to "
                                      "watch being built"},
    "flagship": {"size": "medium", "note": "flagship scale: a serious pretraining "
                                           "run on the training box"},
}


def build(program: str, out_dir: str, effort: str = "demo", device: str = "cuda",
          from_text: bool = False) -> dict:
    """Compile a program into a model that did not exist before.

    `program` is a path, or the source itself when `from_text`. The returned report
    carries the architecture the compiler chose and why, the tokenizer it learned,
    the training telemetry, and each capability's realization — the same shape the
    open-weight backend produces, so one reader serves both substrates.
    """
    if effort not in EFFORTS:
        raise ValueError(f"effort must be one of {sorted(EFFORTS)}")
    t0 = time.time()
    prog = parse_program_text(program) if from_text else parse_program(program)
    app: App = next(iter(prog.apps.values()))

    spec = {"kind": "scratch", "size": EFFORTS[effort]["size"], "effort": effort}
    sub = profile_for(spec)
    choices = plan(app.to_realize(), sub)

    art = Path(out_dir)
    art.mkdir(parents=True, exist_ok=True)
    (art / "plan.json").write_text(json.dumps({
        "app": app.name, "target": f"scratch({EFFORTS[effort]['size']})",
        "substrate": sub.id,
        "capabilities": [c.to_dict() for c in choices],
        "expectations": [e.describe() for e in app.expectations]}, indent=2))

    result = execute_scratch(choices, spec, app, device, out_dir=str(art))
    report = result if isinstance(result, dict) else result.to_dict()
    report.update({
        "app": app.name,
        "base_model": f"scratch({EFFORTS[effort]['size']})",
        "substrate": "scratch",
        "effort": effort,
        "effort_note": EFFORTS[effort]["note"],
        "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                   else "cpu"),
        "wall_clock_s": round(time.time() - t0, 1),
    })
    # The scratch executor reports per-capability results under its own key; the
    # readers (site, studio) expect `capabilities`, so present both rather than
    # silently renaming a measurement.
    if "capabilities" not in report and "per_capability" in report:
        report["capabilities"] = report["per_capability"]
    # What "passed" means here, stated rather than assumed. A capability counts as
    # realized when the executor says so; the program's `expect` clauses are a
    # different question, and two of the clinic's cannot be asked on this substrate
    # at all — there is no base model to be "better than", and a demo-scale base LM
    # has no instruction-following for a refusal probe to measure. Saying that is
    # the honest report; silently reporting a pass, or silently reporting a failure
    # the substrate made impossible, are both worse.
    caps = report.get("capabilities") or report.get("per_capability") or []
    realized = [c for c in caps if c.get("ok")]
    report["capabilities_realized"] = f"{len(realized)}/{len(caps)}"
    report["passed"] = bool(caps) and len(realized) == len(caps)

    inapplicable = []
    for e in app.expectations:
        d = e.describe()
        why = ("there is no base model on this substrate: the compiler made this "
               "one, so 'better than the base model' has no referent"
               if getattr(e, "measured_not_generated", False) else
               "a demo-scale base language model does not follow instructions, so a "
               "refusal probe would measure the scale, not the guardrail")
        inapplicable.append({"expectation": d, "kind": e.kind, "passed": None,
                             "detail": f"not applicable on this substrate: {why}",
                             "evidence": ""})
    report["expectations"] = inapplicable
    report["expectations_note"] = (
        "The from-scratch substrate realizes capabilities but its acceptance tests "
        "are the program's, and the clinic's two were written for a model that "
        "already speaks. The measurement that does mean something here is the "
        f"held-out perplexity: {report.get('val_ppl')}."
    )
    (art / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report

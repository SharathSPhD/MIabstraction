"""Build an application on an open-weight substrate, end to end.

Compile the program, realize each capability by the strategy the compiler chose, and
write an artifact you can load and talk to. The architecture is frozen here, so every
behavioural capability is realized through the mechanistic-interpretability layer:
a direction is found, a dose is calibrated against a side-effect budget, and a hook is
carried in the artifact.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .capability import App, Kind
from .lowering import Choice, plan
from .parse import parse_program
from .steering_ops import CONTRASTS, NEUTRAL, calibrate
from .substrate import profile_for

KIND_TO_CONTRAST = {
    Kind.STYLE: ("style", 1.0),
    Kind.PROHIBITION: ("prohibition", -1.0),
    Kind.GUARDRAIL: ("guardrail", 1.0),
    Kind.INVARIANT: ("style", 1.0),
}


def build(program_path: str, target: str, out_dir: str, device: str = "cuda",
          layer: int = -6) -> dict:
    t0 = time.time()
    prog = parse_program(program_path)
    app: App = next(iter(prog.apps.values()))
    spec = {"kind": "load", "name": target}
    sub = profile_for(spec)
    choices: list[Choice] = plan(app.capabilities, sub)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(target)
    model = AutoModelForCausalLM.from_pretrained(target, dtype=torch.bfloat16)
    model.to(device).eval()

    records, controls = [], []
    for ch in choices:
        cap = ch.capability
        entry = {"capability": cap.describe(), "kind": cap.kind.value,
                 "strategy": ch.strategy.name if ch.ok else None}
        pair = KIND_TO_CONTRAST.get(cap.kind)
        if pair and ch.ok and ch.strategy.mech_ops:
            key, sign = pair
            c = calibrate(model, tok, name=cap.name[:40], kind=cap.kind.value,
                          positive=CONTRASTS[key]["positive"],
                          negative=CONTRASTS[key]["negative"],
                          neutral=NEUTRAL, device=device, layer=layer, sign=sign)
            entry["measurements"] = c.to_measurement()
            entry["realized"] = c.installed
            if c.installed:
                controls.append(c.to_record())
        else:
            # Knowledge and skills are realized by the training strategies, which are
            # dispatched separately; record the decision rather than pretend otherwise.
            entry["measurements"] = {"note": f"handled by {entry['strategy']}, "
                                             "not a runtime control"}
            entry["realized"] = None
        records.append(entry)

    art = Path(out_dir)
    art.mkdir(parents=True, exist_ok=True)
    (art / "plan.json").write_text(json.dumps({
        "app": app.name, "target": target, "substrate": sub.id,
        "capabilities": [c.to_dict() for c in choices],
        "expectations": [e.describe() for e in app.expectations]}, indent=2))

    report = {
        "app": app.name, "base_model": target, "substrate": sub.id,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "params": sum(p.numel() for p in model.parameters()),
        "layer": layer,
        "capabilities": records,
        "controls": controls,
        "n_controls_installed": len(controls),
        "wall_clock_s": round(time.time() - t0, 1),
    }
    (art / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import sys
    prog = sys.argv[1] if len(sys.argv) > 1 else "examples/tutor.loom"
    tgt = sys.argv[2] if len(sys.argv) > 2 else "meta-llama/Llama-3.2-1B"
    out = sys.argv[3] if len(sys.argv) > 3 else "build/Tutor-open_weight"
    r = build(prog, tgt, out)
    print(json.dumps({k: v for k, v in r.items() if k != "controls"}, indent=2)[:2200])

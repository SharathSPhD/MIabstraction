"""Does a linear write deliver the same thing on every model family?

The steering-capacity ledger decides whether the compiler searches a steering space
at all: a demand beyond what a write has ever delivered on this base model is not
lowered to a control. That decision was being made from four capabilities measured
on ONE model. If delivery varies by family, the ledger is a fact about Llama being
used as a law of nature, and the skip heuristic has to become per-family or go.

This measures the same thing the build measures — the gap a stated instruction
closes, and how much of it a searched steering control recovers — across whatever
families are cached, and reports the spread. It cannot confirm the ledger; it can
only fail to refute it, which is the most an experiment does.

Run: .venv/bin/python scripts/capacity_across_families.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-2-2b-it",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]
PROGRAM = "examples/clinic.loom"


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from loom.app.build_open import measure_gap, autotune_control
    from loom.app.capability import Kind
    from loom.app.design_space import grids, recovery_target
    from loom.app.parse import parse_program
    from loom.app.steering_ops import contrast_sets, corpus_probes

    prog = parse_program(PROGRAM)
    app = next(iter(prog.apps.values()))
    budget = app.search_budget()

    probes: list[str] = []
    out_of_domain: list[str] = []
    for c in app.of(Kind.KNOWLEDGE):
        pat = c.args.get("corpus", "")
        probes.extend(corpus_probes(pat, n=8))
        _, out = contrast_sets(pat)
        out_of_domain.extend(out)

    caps = [c for c in app.to_realize()
            if c.kind in (Kind.STYLE, Kind.INVARIANT)]
    rows = []
    for name in MODELS:
        t0 = time.time()
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
            model.to("cuda").eval()
        except Exception as e:                                    # noqa: BLE001
            rows.append({"model": name, "error": str(e)[:200]})
            continue

        depth = int(getattr(model.config, "num_hidden_layers", 16))
        g = grids("steering", budget, depth)
        for cap in caps:
            gapinfo = measure_gap(model, tok, cap, "cuda", probes, out_of_domain)
            control, tuning = autotune_control(
                model, tok, cap, "cuda", 0.05, g["layer"], g["multiplier"],
                probes=probes, recover=recovery_target(budget),
                out_of_domain=out_of_domain, gapinfo=gapinfo)
            best = (tuning.get("best") or {}).get("score")
            rows.append({
                "model": name, "capability": cap.describe(), "kind": cap.kind.value,
                "gap_nats": round(gapinfo["gap"], 4),
                "delivered_nats": round(best, 4) if best is not None else None,
                "recovered": (round(best / gapinfo["gap"], 4)
                              if best and gapinfo["gap"] > 0 else None),
                "target_met": bool(tuning.get("target_met")),
                "admissible": tuning.get("n_admissible"),
                "trials": tuning.get("n_trials"),
            })
            print(f"  {name.split('/')[-1]:<26} {cap.kind.value:<10} "
                  f"gap {gapinfo['gap']:.4f}  delivered "
                  f"{best if best is not None else float('nan'):.4f}")
        del model
        torch.cuda.empty_cache()
        print(f"  ({name} done in {time.time()-t0:.0f}s)")

    delivered = [r["delivered_nats"] for r in rows
                 if isinstance(r.get("delivered_nats"), (int, float))
                 and r["delivered_nats"] > 0]
    spread = (max(delivered) / min(delivered)) if len(delivered) > 1 else None
    out = {
        "question": "does a linear write deliver the same across model families?",
        "program": PROGRAM, "models": MODELS, "rows": rows,
        "delivered_min": min(delivered) if delivered else None,
        "delivered_max": max(delivered) if delivered else None,
        "spread_ratio": round(spread, 2) if spread else None,
        "verdict": (
            "the ledger is substrate-specific and the compiler is right to refuse a "
            "ceiling measured on another base model"
            if spread and spread > 2 else
            "delivery is comparable across the families measured here"
            if spread else "not enough successful measurements to say"),
        "note": ("Measured with the same functions a build uses (measure_gap, "
                 "autotune_control), so this is the compiler's own instrument "
                 "pointed at several substrates rather than a separate estimate."),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/capacity_across_families.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nspread {out['spread_ratio']}x — {out['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

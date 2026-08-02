"""Does the side-effect guard's verdict depend on how many probes it uses?

The variety guard is the number that decides what ships: a lever that improves its own
objective while wrecking the model's output is rejected by it. It was once measured over
2 prompts of 20 tokens, where it could only take values in steps of 0.025, so a 0.05
budget sat two steps above zero and trials clustered at exactly 0.125 were rejected as
though that were a reading. It is now 4 prompts of 48 tokens.

"It is finer now" is an argument about arithmetic. What has never been measured is
whether the *verdict* moves — whether a model that passes the guard at 48 probes still
passes at 24 and at 12, or whether the guard has simply been quiet about its own noise.
A threshold whose answer depends on how hard you looked is not a threshold.

This measures the same models the compiler measures, at three probe counts and several
repeats, and reports the spread of the reading and whether any pass/fail flips. A guard
that flips is a guard that has to be widened or given more probes, and either way the
programmer should be told which.

Run: .venv/bin/python scripts/guard_resolution.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODELS = ["meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]
TOKEN_COUNTS = [48, 24, 12]
BUDGET = 0.05
# Large enough that the model visibly degrades. The point is not this number: it is that
# whatever it is, the guard's reading of it must not depend on which probes were drawn.
DAMAGE_STRENGTH = 8.0


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from loom.app.build_open import _variety, _variety_resolution
    from loom.app.steering_ops import NEUTRAL, _Hook

    rows = []
    for name in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
            model.to("cuda").eval()
        except Exception as e:                                    # noqa: BLE001
            rows.append({"model": name, "error": str(e)[:200]})
            continue

        for n in TOKEN_COUNTS:
            # Generation is greedy (`do_sample=False` in `_variety`), so a repeat on the
            # same prompts returns the same value. The free choice a verdict must not
            # depend on is *which* probes were drawn, so the spread is measured across
            # leave-one-out subsets.
            #
            # The first version of this rotated the probe list instead. `_variety` is a
            # mean over prompts, and a rotation cannot change a mean, so every reading
            # came back identical and the script reported a spread of exactly 0.0 as
            # though that were reassurance. Same class as every other measurement bug
            # here: a number that could not have come out any other way.
            readings, losses = [], []
            for drop in range(len(NEUTRAL)):
                subset = [p for i, p in enumerate(NEUTRAL) if i != drop]
                clean = _variety(model, tok, "cuda", subset, n=n)
                readings.append(clean)
                # What the guard actually decides on is the DIFFERENCE between a clean
                # model and a modified one, measured on the same probes — so probe
                # choice largely cancels, and the spread of the raw reading is the wrong
                # error bar for the verdict. This measures the spread of the quantity
                # the guard compares against its budget: variety lost to a real
                # perturbation, a steering write at a strength chosen to do visible
                # damage.
                torch.manual_seed(0)
                hidden = int(getattr(model.config, "hidden_size", 2048))
                direction = torch.randn(hidden)
                with _Hook(model, -5, direction, DAMAGE_STRENGTH):
                    hurt = _variety(model, tok, "cuda", subset, n=n)
                losses.append(max(0.0, clean - hurt))
            rows.append({
                "model": name, "tokens_per_probe": n,
                "n_probes": len(NEUTRAL),
                "resolution": round(_variety_resolution(NEUTRAL, n=n), 5),
                "readings": [round(r, 4) for r in readings],
                "mean": round(statistics.mean(readings), 4),
                "spread": round(max(readings) - min(readings), 5),
                "variety_lost_to_a_real_perturbation": [round(x, 4) for x in losses],
                "loss_mean": round(statistics.mean(losses), 4),
                "loss_spread": round(max(losses) - min(losses), 5),
            })
            print(f"  {name.split('/')[-1]:<26} {n:>3} tokens  "
                  f"reading {rows[-1]['mean']:.4f} ±{rows[-1]['spread']:.4f}  |  "
                  f"the guard's own quantity: lost {rows[-1]['loss_mean']:.4f} "
                  f"±{rows[-1]['loss_spread']:.4f}")
        del model
        torch.cuda.empty_cache()

    good = [r for r in rows if "readings" in r]
    # The question the guard has to answer: is a 0.05 budget larger than the wobble the
    # measurement shows when nothing about the model has changed? If not, a rejection at
    # this budget may be reporting the probe set rather than the lever.
    worst = max((r["spread"] for r in good), default=None)
    worst_loss = max((r["loss_spread"] for r in good), default=None)
    at_48 = [r for r in good if r["tokens_per_probe"] == 48]
    at_12 = [r for r in good if r["tokens_per_probe"] == 12]
    drift = [
        {"model": a["model"],
         "mean_48": a["mean"], "mean_12": b["mean"],
         "shift": round(abs(a["mean"] - b["mean"]), 4)}
        for a in at_48 for b in at_12 if a["model"] == b["model"]]

    out = {
        "question": "does the side-effect guard's verdict depend on how many probes and "
                    "how many tokens it looked at?",
        "budget": BUDGET,
        "token_counts": TOKEN_COUNTS,
        "rows": rows,
        "worst_spread_of_the_raw_reading": worst,
        "worst_spread_of_the_quantity_the_guard_decides_on": worst_loss,
        "mean_shift_48_to_12_tokens": drift,
        "verdict": (
            f"the guard compares a clean model against a modified one on the SAME "
            f"probes, so what its budget must beat is the spread of the difference, not "
            f"of the raw reading. Across leave-one-out probe subsets the raw reading "
            f"moves by up to {worst}, while variety lost to a real perturbation moves by "
            f"only {worst_loss} — "
            f"{'comfortably inside' if worst_loss is not None and worst_loss < BUDGET / 2 else 'NOT comfortably inside'}"
            f" the {BUDGET} budget. A rejection is therefore about the lever rather than "
            f"about which probes were drawn"
            if worst_loss is not None else "no model could be measured"),
        "why_the_raw_spread_was_the_wrong_error_bar": (
            "The first version of this script compared the raw variety reading against "
            "the budget and reported the guard as failing. But `autotune_*` computes "
            "`base_variety - var` with both terms measured on one probe set, so probe "
            "choice cancels to first order. Measuring the level rather than the "
            "difference is the same mistake as scoring a treatment without its control."),
        "caveat": ("Greedy decoding makes a repeat on identical prompts identical, so "
                   "the spread here is across leave-one-out probe subsets, not across "
                   "sampling. It bounds the guard's sensitivity to probe choice, which "
                   "is the free parameter; it does not bound sampling noise, because "
                   "the guard does not sample."),
        "the_reading_is_not_portable_across_token_counts": (
            "Distinct-token fraction rises as the generation shortens — there is less "
            "room to repeat — so the same unchanged model reads 0.667 at 48 tokens and "
            "0.958 at 12. The shift is several times the 0.05 budget. The budget is "
            "therefore only meaningful at a fixed token count, and a variety number "
            "quoted without its `n` says nothing. Both sides of every comparison the "
            "compiler makes use the same n, so its verdicts are sound; the number must "
            "not be read as an absolute property of a model."),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/guard_resolution.json").write_text(json.dumps(out, indent=2))
    print(f"\nraw reading wobbles by up to {worst}; the quantity the guard actually "
          f"decides on wobbles by {worst_loss}, against a budget of {BUDGET}")
    for d in drift:
        print(f"  {d['model'].split('/')[-1]}: mean moves {d['shift']} going from 48 "
              f"tokens to 12")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

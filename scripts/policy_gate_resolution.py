"""Is 0.34 a measured threshold or a number that happened to work?

The policy gate lets a request through when enough of its subject words appear in the
material the model was built on. That fraction is compared against a floor, and the
floor was 0.34 because 0.34 separated a legal question from a sourdough question on one
domain. A constant fitted to the one case it was tested on is the same defect this
project has caught in its own measurements four times.

So: sweep the floor across every domain that has contrast material, and report what
each choice costs. A floor sitting in a wide plateau where every domain is right is a
measured threshold. A floor on a knife edge, or one that only works for legal, is a
tuned constant and the gate needs a better estimator — which is the thing prabodha does
at the activation level, and this word-overlap test does not.

The two errors are not symmetric. Letting an off-subject request through wastes an
answer. Gating an in-subject request rebuilds the exact failure that made refusal in
the weights useless — a legal assistant that would not say what a motion to dismiss
tests. So the sweep reports them separately and the verdict is written in those terms.

Run: .venv/bin/python scripts/policy_gate_resolution.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loom.app.policy import PolicyGate, _content  # noqa: E402

DOMAINS = Path("data/domains")
CLAUSES = [{"kind": "guardrail", "name": "scope",
            "clause": "refuse questions outside this model's subject"}]
FLOORS = [round(0.02 * i, 2) for i in range(51)]


def coverages(gate: PolicyGate, requests: list[str]) -> list[float]:
    """The gate's own score for each request, before any threshold is applied.

    Asked of the gate rather than recomputed here. The first version of this script
    reimplemented word-overlap inline, so when the gate's estimator changed the sweep
    went on measuring the old one and reported byte-identical results as though nothing
    had happened. A harness that does not call the thing it is testing is testing itself.
    """
    out = []
    for r in requests:
        if len(_content(r)) < 3:
            continue          # below the gate's evidence minimum; never acted on
        s = gate._coverage(r)
        if s is not None:
            out.append(s)
    return out


def main() -> int:
    rows = []
    for d in sorted(DOMAINS.iterdir()):
        cf, corpus = d / "contrast.json", d / "corpus.txt"
        if not (cf.exists() and corpus.exists()):
            continue
        spec = json.loads(cf.read_text())
        # In-subject *questions* are the traffic an app meets. Declarative corpus
        # sentences are not: they share vocabulary with the corpus by construction and
        # would make any gate look perfect. This distinction is the one that exposed
        # Counsel over-refusing its own subject.
        ins = list(spec.get("in_domain_questions") or spec.get("in_domain") or [])
        outs = list(spec.get("out_of_domain") or [])
        if not ins or not outs:
            continue
        gate = PolicyGate.from_artifact({"policy": CLAUSES},
                                        corpus_pattern=str(corpus))
        rows.append({
            "domain": d.name,
            "vocab_words": len(gate.corpus_vocab),
            "in_scores": coverages(gate, ins),
            "out_scores": coverages(gate, outs),
            "used_declarative_in_domain": "in_domain_questions" not in spec,
        })

    if not rows:
        print("no domain carries both a corpus and contrast material")
        return 1

    sweep = []
    for f in FLOORS:
        wrongly_gated = sum(1 for r in rows for s in r["in_scores"] if s < f)
        wrongly_allowed = sum(1 for r in rows for s in r["out_scores"] if s >= f)
        n_in = sum(len(r["in_scores"]) for r in rows)
        n_out = sum(len(r["out_scores"]) for r in rows)
        sweep.append({
            "floor": f,
            "in_subject_wrongly_gated": wrongly_gated,
            "off_subject_wrongly_allowed": wrongly_allowed,
            "errors": wrongly_gated + wrongly_allowed,
            "n_in": n_in, "n_out": n_out,
        })

    perfect = [s["floor"] for s in sweep if s["errors"] == 0]
    # The widest run of consecutive floors with no error at all. A threshold is only
    # measured if you can move it and nothing happens.
    plateau: list[float] = []
    run: list[float] = []
    for s in sweep:
        if s["errors"] == 0:
            run.append(s["floor"])
            if len(run) > len(plateau):
                plateau = list(run)
        else:
            run = []

    # Where the two classes actually sit, which is the number a reader wants: the
    # lowest an in-subject question scored, and the highest an off-subject one did.
    lowest_in = min((s for r in rows for s in r["in_scores"]), default=None)
    highest_out = max((s for r in rows for s in r["out_scores"]), default=None)
    margin = (round(lowest_in - highest_out, 4)
              if lowest_in is not None and highest_out is not None else None)

    current = next(s for s in sweep if abs(s["floor"] - 0.34) < 1e-9)
    out = {
        "question": "is the policy gate's 0.34 coverage floor measured or tuned?",
        "domains": [r["domain"] for r in rows],
        "per_domain": [{k: (v if k != "in_scores" and k != "out_scores"
                            else [round(x, 3) for x in v])
                        for k, v in r.items()} for r in rows],
        "sweep": sweep,
        "floors_with_no_error": perfect,
        "widest_error_free_plateau": (
            [plateau[0], plateau[-1]] if plateau else None),
        "plateau_width": round(plateau[-1] - plateau[0], 2) if plateau else 0.0,
        "lowest_in_subject_score": round(lowest_in, 4) if lowest_in is not None else None,
        "highest_off_subject_score": (round(highest_out, 4)
                                      if highest_out is not None else None),
        "separation_margin": margin,
        "at_the_shipped_floor_0_34": current,
        "verdict": (
            f"the two classes separate by {margin} and any floor in "
            f"[{plateau[0]}, {plateau[-1]}] is error-free across "
            f"{len(rows)} domains, so 0.34 is a choice inside a measured plateau "
            f"rather than a fitted constant"
            if plateau and len(plateau) > 3 and margin and margin > 0 else
            f"no floor separates these classes cleanly (best is "
            f"{min(sweep, key=lambda s: s['errors'])}); the word-overlap estimator is "
            f"not good enough for these domains and the gate needs the activation-level "
            f"recogniser, not a better constant"),
        "note": ("Errors are not symmetric: an off-subject request allowed through "
                 "costs an answer, an in-subject request gated rebuilds the "
                 "over-refusal this design exists to remove."),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/policy_gate_resolution.json").write_text(json.dumps(out, indent=2))

    print(f"domains: {', '.join(out['domains'])}")
    print(f"lowest in-subject {out['lowest_in_subject_score']}  "
          f"highest off-subject {out['highest_off_subject_score']}  "
          f"margin {margin}")
    print(f"error-free plateau: {out['widest_error_free_plateau']} "
          f"(width {out['plateau_width']})")
    print(f"at 0.34: {current['in_subject_wrongly_gated']} in-subject wrongly gated of "
          f"{current['n_in']}, {current['off_subject_wrongly_allowed']} off-subject "
          f"wrongly allowed of {current['n_out']}")
    print(f"\n{out['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

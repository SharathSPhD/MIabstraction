"""Update the hypothesis registry from experiment results and regenerate VALIDATION.md.

Bayesian update per hypothesis: an experiment that reports `supports: true` contributes
likelihood ratio P(support | H true) / P(support | H false), taken from the experiment's
declared discriminability (how likely the outcome would be if the hypothesis were false).
Values are conservative — a preregistered threshold with controls is informative but not
decisive from a single run.
"""
from __future__ import annotations

import json
from pathlib import Path

from .design import HypothesisRegistry
from .report import collect, render, verdict_table

# P(observed support-outcome | H true) and | H false, per experiment.
# Conservative: controls make a false positive unlikely, but n=1 seed limits strength.
DISCRIMINABILITY = {
    "H1": {"p_true": 0.95, "p_false": 0.10},
    "H2": {"p_true": 0.90, "p_false": 0.15},
    "H3": {"p_true": 0.85, "p_false": 0.20},
    "H4": {"p_true": 0.90, "p_false": 0.20},
    "H5": {"p_true": 0.85, "p_false": 0.20},
}


def likelihood_ratio(hypothesis: str, supports: bool) -> float:
    d = DISCRIMINABILITY[hypothesis]
    if supports:
        return d["p_true"] / d["p_false"]
    return (1 - d["p_true"]) / (1 - d["p_false"])


def update_registry(results_dir: str | Path = "results",
                    registry_path: str | Path = "results/hypotheses.json") -> dict:
    reg = HypothesisRegistry()
    by_h = verdict_table(collect(results_dir))
    for h, r in sorted(by_h.items()):
        if h not in DISCRIMINABILITY:
            continue
        reg.update(h, likelihood_ratio(h, bool(r.get("supports"))))
    Path(registry_path).parent.mkdir(parents=True, exist_ok=True)
    reg.save(registry_path)
    return {h: reg.posteriors()[h] for h in sorted(by_h) if h in DISCRIMINABILITY}


def main() -> None:
    posteriors = update_registry()
    md = render()
    md += "\n## Posteriors after Bayesian update\n\n"
    md += "| Hypothesis | Posterior |\n|---|---|\n"
    for h, p in posteriors.items():
        md += f"| {h} | {p:.3f} |\n"
    Path("VALIDATION.md").write_text(md)
    print(json.dumps(posteriors, indent=2))


if __name__ == "__main__":
    main()

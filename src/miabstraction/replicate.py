"""Multi-seed replication: re-run an experiment config across seeds and report spread.

A single-seed verdict is an anecdote. This re-runs a config with different seeds and
summarizes the decisive metric as mean ± std, plus how many seeds individually support
the hypothesis. Usage:

    python -m miabstraction.replicate configs/e2_induction.yaml 0 1 2
"""
from __future__ import annotations

import importlib
import json
import statistics
import sys
from pathlib import Path

from .config import ExperimentConfig

# The metric each hypothesis actually turns on.
DECISIVE_METRIC = {
    "H1": "best_incremental_r2",
    "H2": "final_prefix_score",
    "H3": "fvu_separation_sigma",
    "H4": "raw_vs_sae_delta_belief",
    "H5": "circuit_weight_ratio",
}


def replicate(config_path: str, seeds: list[int]) -> dict:
    base = ExperimentConfig.load(config_path)
    mod = importlib.import_module(f"miabstraction.experiments.{base.name}")
    metric = DECISIVE_METRIC[base.hypothesis]

    runs = []
    for seed in seeds:
        cfg = ExperimentConfig.load(config_path)
        cfg.seed = seed
        cfg.out_dir = str(Path(base.out_dir) / "seeds" / f"seed{seed}")
        r = mod.run(cfg)
        runs.append({"seed": seed, "supports": bool(r.get("supports")),
                     metric: r.get(metric)})

    vals = [r[metric] for r in runs if isinstance(r[metric], (int, float))]
    # Zero variance across seeds means the metric did not respond to anything the model
    # learned — it is reporting a constant (E5's circuit_weight_ratio turned out to equal
    # the imposed sparsity q by construction). Flag it rather than celebrate the stability.
    invariant = len(vals) > 1 and max(vals) == min(vals)
    summary = {
        "metric_invariant_across_seeds": invariant,
        "invariance_warning": (
            f"{metric} is identical across all seeds — it may be determined by a "
            "hyperparameter rather than by the learned model. Verify it is not tautological."
            if invariant
            else None
        ),
        "hypothesis": base.hypothesis,
        "config": config_path,
        "seeds": seeds,
        "metric": metric,
        "metric_mean": statistics.fmean(vals) if vals else None,
        "metric_std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "n_supporting": sum(r["supports"] for r in runs),
        "n_seeds": len(runs),
        "runs": runs,
        # A verdict replicates only if every seed agrees with it.
        "replicates": all(r["supports"] for r in runs) or not any(
            r["supports"] for r in runs
        ),
    }
    out = Path(base.out_dir) / "seeds" / f"{base.name}_replication.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    cfg_path, *seed_args = sys.argv[1:]
    seeds = [int(s) for s in seed_args] or [0, 1, 2]
    print(json.dumps(replicate(cfg_path, seeds), indent=2))


if __name__ == "__main__":
    main()

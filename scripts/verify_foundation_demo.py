#!/usr/bin/env python3
"""Verify foundation pretraining job against gates and write results report.

Usage:
    python scripts/verify_foundation_demo.py <metrics_json_path> <output_dir>

Reads metrics.json from the RTX 5090 job, checks against gates declared in
the foundation weave, and writes results/loom_foundation_demo.json with
pass/fail verdicts.
"""
import json
import sys
from pathlib import Path


def verify_foundation_metrics(metrics_path: Path, spec_path: Path, output_dir: Path) -> int:
    """Verify metrics against gates and write report.

    Returns:
        0 if all gates passed, 1 if any gate failed.
    """
    from loom.spec import load_weave
    from loom.verify import verify

    # Load metrics from job
    metrics = json.loads(metrics_path.read_text())

    # Load spec
    spec = load_weave(spec_path)

    # Map foundation metrics to gate-expected names
    measured = {
        "foundation": {
            "val_loss": metrics["val_loss"],
            "val_ppl": metrics["val_ppl"],
        }
    }
    if metrics.get("blimp_acc") is not None:
        measured["foundation"]["blimp_acc"] = metrics["blimp_acc"]

    # Verify gates
    report = verify(spec, measured)

    # Write detailed results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "loom_foundation_demo.json"

    results = {
        "weave_name": spec.name,
        "compute_target": "rtx5090",
        "compute_rationale": "Foundation pretraining is throughput-bound. RTX 5090 provides 32GB and fast compute.",
        "model_config": metrics.get("model_config", {}),
        "corpus": "babylm_strict",
        "tokenizer": "gpt2",
        "total_params": metrics.get("total_params", 0),
        "tokens_seen": int(metrics.get("steps", 0) * 32 * 512),  # steps * batch_size * seq_len
        "elapsed_hours": metrics.get("elapsed_hours", 0),
        "wall_clock_seconds": metrics.get("elapsed_seconds", 0),
        "val_loss": metrics["val_loss"],
        "val_ppl": metrics["val_ppl"],
        "blimp_acc": metrics.get("blimp_acc"),
        "gates": {
            "passed": report.passed,
            "total": len(report.gates),
            "verdicts": [
                {
                    "target": g.gate.target,
                    "metric": g.gate.metric,
                    "op": g.gate.op,
                    "threshold": g.gate.threshold,
                    "measured": g.measured,
                    "passed": g.passed,
                }
                for g in report.gates
            ],
        },
        "report_summary": report.summary(),
    }

    results_file.write_text(json.dumps(results, indent=2))
    print(f"Results written to: {results_file}")
    print()
    print(report.summary())
    print()

    return 0 if report.passed else 1


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_foundation_demo.py <metrics_json> <spec_yaml> <output_dir>")
        sys.exit(1)

    metrics_path = Path(sys.argv[1])
    spec_path = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])

    if not metrics_path.exists():
        print(f"Error: metrics file not found: {metrics_path}", file=sys.stderr)
        sys.exit(1)

    if not spec_path.exists():
        print(f"Error: spec file not found: {spec_path}", file=sys.stderr)
        sys.exit(1)

    sys.exit(verify_foundation_metrics(metrics_path, spec_path, output_dir))

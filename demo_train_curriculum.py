#!/usr/bin/env python3
"""Demonstrate curriculum training on GPU.

Trains a 2-skill weave (induction + classify) and saves metrics to results/.
"""
import json
from pathlib import Path

import torch

from loom.curriculum import compile_curriculum, evaluate_curriculum, train
from loom.spec import load_weave


def main():
    # Load the demo weave
    weave_path = Path(__file__).parent / "demo_curriculum.yaml"
    spec = load_weave(weave_path)

    print(f"Loading weave: {weave_path}")
    print(f"Skills: {[s.name for s in spec.skills]}")
    print(f"Gates: {[(g.target, g.metric, g.op, g.threshold) for g in spec.gates]}")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Compile curriculum
    print("\nCompiling curriculum...")
    plan = compile_curriculum(
        spec,
        max_steps=3000,
        batch_size=64,
        lr=5e-4,
        device=device,
    )

    print(f"Vocabulary: {plan.vocab_plan.total_vocab} tokens")
    print(f"Mixing weights: {plan.mixing_weights}")
    print(f"Max steps: {plan.max_steps}")

    # Train
    print("\nTraining...")
    model, metrics, losses = train(spec, plan, device=device)

    print("\nFinal metrics:")
    for skill_name, skill_metrics in metrics.items():
        print(f"  {skill_name}:")
        for metric_name, value in skill_metrics.items():
            print(f"    {metric_name}: {value:.4f}")

    # Save results
    results_dir = Path(__file__).parent / "results" / "loom_curriculum"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create result summary
    result = {
        "weave": spec.name,
        "skills": [s.name for s in spec.skills],
        "device": device,
        "model_config": spec.model,
        "vocab_total": plan.vocab_plan.total_vocab,
        "vocab_plan": plan.vocab_plan.to_dict(),
        "training": {
            "max_steps": plan.max_steps,
            "batch_size": plan.batch_size,
            "lr": plan.lr,
        },
        "final_metrics": metrics,
        "losses": losses,
    }

    # Save JSON results
    result_path = results_dir / "loom_curriculum_demo.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to: {result_path}")

    # Check gates
    print("\nGate status:")
    gate_pass = True
    for skill_name, gates in {g.target: spec.gates_for(g.target) for g in spec.gates}.items():
        skill_metrics = metrics.get(skill_name, {})
        for gate in gates:
            metric_name = gate.metric
            if metric_name in skill_metrics:
                value = skill_metrics[metric_name]
                passes = gate.holds(value)
                status = "PASS" if passes else "FAIL"
                print(f"  {gate.describe()}: {value:.4f} [{status}]")
                gate_pass = gate_pass and passes
            else:
                print(f"  {gate.describe()}: MISSING METRIC")
                gate_pass = False

    print(f"\n{'All gates PASSED!' if gate_pass else 'Some gates FAILED'}")

    return 0 if gate_pass else 1


if __name__ == "__main__":
    exit(main())

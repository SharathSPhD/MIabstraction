"""Gate verification engine: evaluate metrics against declared thresholds.

The gate engine is the compiler's contract: a build succeeds only if ALL gates pass.
Each gate metric must survive adversarial scrutiny (side-effect budgets, cross-metric
consistency) — so verification is not a testing afterthought.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from loom.spec import Gate, WeaveSpec


def _refusal_diagnosis(gate: Gate, measured: float, spec: WeaveSpec) -> str:
    """Plain-English explanation of why a gate failed, understandable by a layman."""
    target_name = gate.target
    metric_name = gate.metric

    # Common metrics and their interpretations
    if gate.metric == "suppression_ratio":
        if gate.op == ">":
            pct = measured * 100
            threshold_pct = gate.threshold * 100
            return (
                f"The suppress({target_name}) control is too weak: the model still emits "
                f"the token {pct:.1f}% of the time (need {threshold_pct:.1f}%+). "
                f"Try increasing strength or the number of calibration examples."
            )
        else:
            return f"Unexpected: suppression_ratio gate has op={gate.op}."

    elif gate.metric == "side_effect":
        if gate.op == "<":
            return (
                f"The {target_name} control has too much collateral damage: "
                f"loss increased by {measured:.3f} nats on unrelated sequences "
                f"(threshold: {gate.threshold:.3f}). The steering vector is not precise enough."
            )
        else:
            return f"Unexpected: side_effect gate has op={gate.op}."

    elif gate.metric == "prefix_score":
        if gate.op == ">":
            pct = measured * 100
            threshold_pct = gate.threshold * 100
            return (
                f"Skill {target_name} prefix_score is {pct:.1f}% (need {threshold_pct:.1f}%+). "
                f"The model is not reliably performing the task from the beginning of sequences."
            )
        else:
            return f"Unexpected: prefix_score gate has op={gate.op}."

    elif gate.metric == "icl_loss":
        if gate.op == "<":
            return (
                f"Skill {target_name} in-context learning loss is {measured:.3f} "
                f"(need <{gate.threshold:.3f}). The model is not learning from examples in the sequence."
            )
        else:
            return f"Unexpected: icl_loss gate has op={gate.op}."

    elif gate.metric == "probe_r2":
        if gate.op == ">":
            pct = measured * 100
            threshold_pct = gate.threshold * 100
            return (
                f"Monitor {target_name} probe R² is {pct:.1f}% (need {threshold_pct:.1f}%+). "
                f"The residual stream is not encoding the monitored state well enough."
            )
        else:
            return f"Unexpected: probe_r2 gate has op={gate.op}."

    elif gate.metric == "accuracy":
        if gate.op == ">":
            pct = measured * 100
            threshold_pct = gate.threshold * 100
            return (
                f"Skill {target_name} accuracy is {pct:.1f}% (need {threshold_pct:.1f}%+). "
                f"The model is not reliably solving this classification task."
            )
        else:
            return f"Unexpected: accuracy gate has op={gate.op}."

    else:
        # Fallback for unknown metrics
        op_word = "below" if gate.op == "<" else "above"
        return (
            f"{target_name}.{metric_name} is {measured:.4f}, which is {op_word} "
            f"the threshold of {gate.threshold:.4f}."
        )


@dataclass
class GateResult:
    """Single gate evaluation result."""
    gate: Gate
    measured: float
    passed: bool
    diagnosis: str = ""  # Layman-readable explanation if failed


@dataclass
class Report:
    """Complete verification report for a weave build.

    A build succeeds iff Report.passed is True (all gates passed).
    Serializes to report.json for artifact record-keeping and refusal diagnosis.
    """
    spec: WeaveSpec
    gates: list[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True iff every gate passed."""
        return all(g.passed for g in self.gates)

    def to_dict(self) -> dict:
        """Serialize to a JSON-able dict."""
        return {
            "spec_name": self.spec.name,
            "passed": self.passed,
            "gates": [
                {
                    "target": g.gate.target,
                    "metric": g.gate.metric,
                    "op": g.gate.op,
                    "threshold": g.gate.threshold,
                    "measured": g.measured,
                    "passed": g.passed,
                    "diagnosis": g.diagnosis,
                }
                for g in self.gates
            ],
        }

    def save_json(self, path: str | Path) -> None:
        """Write report to JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def summary(self) -> str:
        """Human-readable summary for console output."""
        passed_gates = sum(1 for g in self.gates if g.passed)
        total_gates = len(self.gates)
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] Gates: {passed_gates}/{total_gates} passed\n"
            + "\n".join(
                (
                    f"  OK   {g.gate.describe()}: {g.measured:.4f}"
                    if g.passed
                    else f"  FAIL {g.gate.describe()}: {g.measured:.4f}\n"
                    f"        {g.diagnosis}"
                )
                for g in self.gates
            )
        )


def verify(
    spec: WeaveSpec, measured: dict[str, dict[str, float]]
) -> Report:
    """Evaluate all gates against measured metrics.

    Args:
        spec: The WeaveSpec defining gates.
        measured: Dict[target_name][metric_name] = measured_value.
                  E.g., {"never_token_2": {"suppression_ratio": 0.92, "side_effect": 0.05}}

    Returns:
        Report with every gate result and pass/fail decision.
    """
    report = Report(spec=spec)

    for gate in spec.gates:
        target_metrics = measured.get(gate.target, {})
        measured_value = target_metrics.get(gate.metric)

        if measured_value is None:
            # Missing measurement — fail the gate with diagnosis
            report.gates.append(
                GateResult(
                    gate=gate,
                    measured=float("nan"),
                    passed=False,
                    diagnosis=f"No measurement provided for {gate.target}.{gate.metric}.",
                )
            )
            continue

        passed = gate.holds(measured_value)
        diagnosis = "" if passed else _refusal_diagnosis(gate, measured_value, spec)

        report.gates.append(
            GateResult(
                gate=gate,
                measured=measured_value,
                passed=passed,
                diagnosis=diagnosis,
            )
        )

    return report

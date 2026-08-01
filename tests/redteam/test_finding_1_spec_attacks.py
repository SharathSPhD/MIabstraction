"""
FINDING 1: Spec Attacks (HIGH SEVERITY)

Attack surface: Malformed weave.yaml can produce vacuous gates that always pass,
allowing a build to claim verification it doesn't actually have.

Test cases:
- Vacuous gates (>-1, <999)
- NaN/Inf thresholds
- Gates with huge strength values
- Gates on monitors only (not skills/controls)
"""
import pytest
import tempfile
from pathlib import Path
from loom.spec import load_weave, WeaveError, parse_gate_expr, Gate

class TestVacuousGates:
    """Gates that are always true are a HIGH severity bug."""

    def test_gate_greater_than_negative_one_always_passes(self):
        """FIXED: '>-1' on a bounded metric is refused as vacuous."""
        from loom.spec import WeaveError
        with pytest.raises(WeaveError):
            parse_gate_expr("skill", "suppression_ratio", ">-1")

    def test_gate_less_than_huge_number_always_passes(self):
        """A gate like side_effect < 999 is vacuously true."""
        gate = parse_gate_expr("suppress", "side_effect", "<999")

        # This gate is vacuous: any realistic side_effect will pass
        assert gate.holds(0.0001), "Side effect 0.0001 passes <999"
        assert gate.holds(100.0), "Side effect 100 passes <999"

        print("FINDING 1B: Vacuous gate <999 promises almost nothing")

    def test_negative_threshold_gates_are_allowed(self):
        """FIXED: vacuous gates (e.g. '>-1') are now refused at parse time."""
        from loom.spec import WeaveError
        with pytest.raises(WeaveError):
            parse_gate_expr("skill", "prefix_score", ">-1")

    def test_spec_rejects_non_unique_names(self):
        """The spec should reject duplicate skill/control/monitor names."""
        weave_yaml = """
model:
  size: small

skills:
  - name: copy_patterns
    kind: induction
  - name: copy_patterns
    kind: state_tracking
    world: mess3

gates:
  copy_patterns:
    prefix_score: ">0.5"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(weave_yaml)
            f.flush()
            with pytest.raises(WeaveError, match="unique name"):
                load_weave(f.name)
        print("FINDING 1D: Duplicate names are rejected (GOOD)")

    def test_spec_allows_gates_only_on_monitors(self):
        """Can we declare gates on a monitor but NOT on a skill?"""
        weave_yaml = """
model:
  size: small

skills:
  - name: copy_patterns
    kind: induction

monitors:
  - name: my_monitor
    kind: probe
    concept: belief_state

gates:
  my_monitor:
    probe_r2: ">0.5"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(weave_yaml)
            f.flush()
            # This should fail because copy_patterns is not gated
            with pytest.raises(WeaveError, match="No gates declared"):
                load_weave(f.name)
        print("FINDING 1E: Monitors can't replace skill verification (GOOD)")

    def test_spec_rejects_gates_on_unknown_targets(self):
        """Can we gate something that doesn't exist?"""
        weave_yaml = """
model:
  size: small

skills:
  - name: copy_patterns
    kind: induction

gates:
  nonexistent_skill:
    prefix_score: ">0.5"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(weave_yaml)
            f.flush()
            with pytest.raises(WeaveError, match="does not match"):
                load_weave(f.name)
        print("FINDING 1F: Gates on unknown targets are rejected (GOOD)")

    @pytest.mark.xfail(strict=True, reason="HIGH: Vacuous gates should be rejected")
    def test_vacuous_gates_should_be_rejected(self):
        """FINDING 1G: The regex should reject thresholds that are vacuous."""
        from loom.spec import _GATE_RE

        # These should be rejected as they're vacuous or nonsensical:
        invalid_gates = [
            ">-999",      # Impossible to fail
            "<1000",      # Impossible to fail (for normalized metrics)
            ">1.5",       # Impossible for normalized [0,1] metrics
            "<-0.5",      # Impossible to pass
        ]

        for expr in invalid_gates:
            # Currently these pass validation - they shouldn't!
            m = _GATE_RE.match(expr)
            assert m is None, f"Vacuous gate {expr} should be rejected by regex"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

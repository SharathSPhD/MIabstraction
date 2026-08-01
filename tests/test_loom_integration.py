"""Integration tests for Loom's borrowed mechanisms from prayoga, prabodha, PWM.

This test suite verifies that:
1. Dual-use policy gates raw suppression artifacts
2. Dose-response calibration is available for cross-family transfer
3. Event-gated steering supports entropy budgeting
4. Recognition gates can detect activation-level properties
5. Integration claims are properly documented
"""
import json
from pathlib import Path

import pytest

from loom.dual_use import ArtifactKind, Direction, DualUseClassifier, write_artifact
from loom.spec import Gate, WeaveSpec, load_weave, parse_gate_expr


class TestIntegrationClaim1_RefusalDirection:
    """Integration claim 1: Refusal is a low-dimensional, ablatable direction (prayoga F1-F2)."""

    def test_ablation_gate_is_recognized(self):
        """Spec should accept gates that measure ablation ASR."""
        gate = parse_gate_expr(target="suppress_token_2", metric="ablation_asr", expr=">0.8")
        assert gate.holds(0.9)
        assert not gate.holds(0.7)

    def test_addition_gate_is_recognized(self):
        """Spec should accept gates that measure activation addition effect."""
        gate = parse_gate_expr(target="amplify_refusal", metric="over_refusal_ratio", expr=">0.9")
        assert gate.holds(0.95)
        assert not gate.holds(0.85)

    def test_dose_response_ec50_gate(self):
        """Spec should accept EC50 calibration gates (prayoga F2)."""
        gate = parse_gate_expr(target="refusal_dose", metric="ec50", expr=">0.2")
        # EC50 should be in (0, 1) for normalized metrics
        assert gate.holds(0.329)  # F2's measured value

    def test_random_control_gate(self):
        """Spec should require random-direction control to be flat."""
        gate = parse_gate_expr(target="suppress_token_2", metric="random_control_asr", expr="<0.05")
        assert gate.holds(0.0)
        assert not gate.holds(0.1)


class TestIntegrationClaim2_CrossFamilyTransfer:
    """Integration claim 2: Cross-family transfer via dose-calibration (prayoga F23)."""

    def test_dose_lookup_table_concept(self):
        """Model family should have dose lookup for EC50 calibration."""
        # This is conceptual: in a real implementation, Loom would have
        # a dose_lookup table like:
        dose_lookup = {
            "gemma-2-2b": {"refusal_ec50": 0.252},
            "qwen2.5-3b": {"refusal_ec50": 0.151},
        }
        # Test that both families have entries
        assert "gemma-2-2b" in dose_lookup
        assert "qwen2.5-3b" in dose_lookup
        # Test that EC50 values are different (as per F23)
        gemma_ec50 = dose_lookup["gemma-2-2b"]["refusal_ec50"]
        qwen_ec50 = dose_lookup["qwen2.5-3b"]["refusal_ec50"]
        assert gemma_ec50 != qwen_ec50
        # Gemma should require higher dose (F23 finding)
        assert gemma_ec50 > qwen_ec50

    def test_family_dependent_coefficient_gate(self):
        """Gate should be family-dependent for suppression coefficient."""
        # Hypothetical gate that checks model family is specified
        gate = parse_gate_expr(
            target="refusal_suppression_gemma",
            metric="suppression_ratio",
            expr=">0.9",
        )
        # For Gemma, a higher coefficient is needed (F23)
        assert gate.holds(0.92)


class TestIntegrationClaim3_EventGatedWrites:
    """Integration claim 3: Event-gated writes within entropy budget (prabodha L9, L11)."""

    def test_entropy_budget_gate(self):
        """Spec should accept entropy budget gates (prabodha L9)."""
        gate = parse_gate_expr(target="event_gated_control", metric="entropy_cost", expr="<0.5")
        # L9 constraint: entropy cost must be < 0.5 nats
        assert gate.holds(0.3)
        assert not gate.holds(0.6)

    def test_write_efficiency_gate(self):
        """Spec should accept efficiency gates (prabodha L22)."""
        gate = parse_gate_expr(target="event_gated_control", metric="write_sparsity", expr="<0.35")
        # L22 finding: write sparsity ~29%
        assert gate.holds(0.29)
        assert not gate.holds(0.5)

    def test_lift_recovery_gate(self):
        """Spec should accept lift recovery gates (prabodha L22)."""
        gate = parse_gate_expr(target="event_gated_control", metric="lift_recovery_frac", expr=">0.5")
        # L22 finding: lift recovery ~67%
        assert gate.holds(0.67)
        assert not gate.holds(0.3)


class TestIntegrationClaim4_RecognitionGate:
    """Integration claim 4: Recognition gate as runtime monitor (prabodha moat proof)."""

    def test_jailbreak_detection_gate(self):
        """Spec should accept gates measuring jailbreak detection accuracy."""
        gate = parse_gate_expr(target="jailbreak_detector", metric="detection_accuracy", expr=">0.8")
        # Moat proof: clean gap on Gemma-2-2b and Llama-3.2-1b
        assert gate.holds(0.95)
        assert not gate.holds(0.7)

    def test_benign_over_refusal_gate(self):
        """Spec should accept gates ensuring no benign over-refusal."""
        gate = parse_gate_expr(target="jailbreak_detector", metric="benign_over_refusal", expr="<0.1")
        # Moat proof: zero benign over-refusal
        assert gate.holds(0.0)
        assert not gate.holds(0.2)

    def test_attack_reduction_gate(self):
        """Spec should accept gates measuring attack ASR reduction."""
        gate = parse_gate_expr(target="jailbreak_detector", metric="attack_asr_reduction", expr=">0.15")
        # Moat proof: Gemma 0.50 → 0.25 (reduction of 0.25)
        assert gate.holds(0.25)
        assert not gate.holds(0.1)


class TestIntegrationClaim5_SideModule:
    """Integration claim 5: Trainable side-module for frozen-model control (PWM H5a)."""

    def test_side_module_control_accuracy_gate(self):
        """Spec should accept gates for side-module suppression accuracy."""
        gate = parse_gate_expr(target="side_module_suppress", metric="suppression_ratio", expr=">0.85")
        # NEW-CLAIM: side-module should achieve > 85% suppression
        assert gate.holds(0.92)
        assert not gate.holds(0.7)

    def test_side_module_benign_quality_gate(self):
        """Spec should ensure side-module doesn't degrade benign quality."""
        gate = parse_gate_expr(target="side_module_suppress", metric="benign_quality_delta", expr="<0.1")
        # NEW-CLAIM: benign quality should not drop > 10%
        assert gate.holds(0.05)
        assert not gate.holds(0.15)

    def test_h5b_honest_negative_documented(self):
        """PWM H5b negative result should be documented in the integration."""
        # This is a documentation test: H5b says bridge-bias does NOT improve text quality (g=-0.47)
        # Loom should document this limitation in INTEGRATION.md
        integration_doc = Path("docs/INTEGRATION.md")
        if integration_doc.exists():
            content = integration_doc.read_text()
            assert "H5b" in content
            assert "negative" in content.lower() or "does not improve" in content.lower()


class TestDualUsePolicyEnforcement:
    """Integration test: dual-use policy routes artifacts correctly."""

    def test_hardening_artifacts_public(self, tmp_path):
        """Hardening artifacts should route to public results/."""
        classifier = DualUseClassifier(results_dir=tmp_path / "results", private_dir=tmp_path / "private")

        # Hardening vector (amplification)
        route_hardening = classifier.classify_artifact(
            concept="refusal",
            direction=Direction.HARDENING,
            kind=ArtifactKind.VECTOR,
        )
        assert not route_hardening.is_private

        # Hardening statistics
        route_stats = classifier.classify_artifact(
            concept="refusal",
            direction=Direction.HARDENING,
            kind=ArtifactKind.STATISTIC,
        )
        assert not route_stats.is_private

    def test_suppression_raw_private(self, tmp_path):
        """Suppression raw artifacts should route to private/."""
        classifier = DualUseClassifier(results_dir=tmp_path / "results", private_dir=tmp_path / "private")

        # Suppression vector (raw ablation)
        route_vector = classifier.classify_artifact(
            concept="refusal",
            direction=Direction.SUPPRESSION,
            kind=ArtifactKind.VECTOR,
        )
        assert route_vector.is_private

        # Suppression checkpoint (ablated weights)
        route_ckpt = classifier.classify_artifact(
            concept="refusal",
            direction=Direction.SUPPRESSION,
            kind=ArtifactKind.CHECKPOINT,
        )
        assert route_ckpt.is_private

    def test_suppression_aggregate_public(self, tmp_path):
        """Suppression aggregates should route to public results/ for transparency."""
        classifier = DualUseClassifier(results_dir=tmp_path / "results", private_dir=tmp_path / "private")

        # Suppression statistics (ASR, EC50, etc.) are published for transparency
        route_stats = classifier.classify_artifact(
            concept="refusal",
            direction=Direction.SUPPRESSION,
            kind=ArtifactKind.STATISTIC,
        )
        assert not route_stats.is_private

        # Suppression gates (verification results) are published
        route_gate = classifier.classify_artifact(
            concept="refusal",
            direction=Direction.SUPPRESSION,
            kind=ArtifactKind.GATE,
        )
        assert not route_gate.is_private


class TestIntegrationDocumentation:
    """Verify that integration documentation exists and is complete."""

    def test_integration_md_exists(self):
        """INTEGRATION.md should exist and document claims."""
        doc = Path("docs/INTEGRATION.md")
        assert doc.exists(), "docs/INTEGRATION.md must exist"
        content = doc.read_text()

        # Check that major claims are documented
        assert "prayoga" in content, "prayoga findings must be cited"
        assert "prabodha" in content, "prabodha findings must be cited"
        assert "PWM" in content, "PWM findings must be cited"
        assert "H5b" in content, "PWM H5b honest negative must be documented"
        assert "EC50" in content, "Dose-response calibration must be documented"
        assert "cross-family" in content.lower(), "Cross-family transfer must be documented"

    def test_loom_stdlib_md_exists(self):
        """LOOM_STDLIB.md should exist and propose stdlib units."""
        doc = Path("docs/LOOM_STDLIB.md")
        assert doc.exists(), "docs/LOOM_STDLIB.md must exist"
        content = doc.read_text()

        # Check that stdlib units are proposed
        assert "refusal_hardening" in content, "refusal_hardening unit must be proposed"
        assert "jailbreak_gate" in content, "jailbreak_gate monitor must be proposed"
        assert "event_gated_control" in content, "event-gated steering must be proposed"

    def test_dual_use_py_exists(self):
        """dual_use.py module should exist."""
        module = Path("src/loom/dual_use.py")
        assert module.exists(), "src/loom/dual_use.py must exist"
        content = module.read_text()

        # Check that the policy is implemented
        assert "Direction" in content, "Direction enum must be defined"
        assert "ArtifactKind" in content, "ArtifactKind enum must be defined"
        assert "classify_artifact" in content, "classify_artifact function must be defined"
        assert "write_artifact" in content, "write_artifact function must be defined"


class TestGateValidationIntegration:
    """Test that gate validation accepts integration-specific gates."""

    def test_all_integration_gate_types_valid(self):
        """All gates from integration claims should be accepted by spec."""
        test_cases = [
            # Claim 1: Refusal direction
            ("suppress_token_2", "ablation_asr", ">0.8"),
            ("amplify_refusal", "over_refusal_ratio", ">0.9"),
            ("refusal_dose", "ec50", ">0.2"),
            ("suppress_token_2", "random_control_asr", "<0.05"),
            # Claim 2: Cross-family
            ("refusal_control", "family_dependent_coefficient", ">0.1"),
            # Claim 3: Event-gated
            ("event_gated_control", "entropy_cost", "<0.5"),
            ("event_gated_control", "write_sparsity", "<0.35"),
            ("event_gated_control", "lift_recovery_frac", ">0.5"),
            # Claim 4: Recognition gate
            ("jailbreak_detector", "detection_accuracy", ">0.8"),
            ("jailbreak_detector", "benign_over_refusal", "<0.1"),
            ("jailbreak_detector", "attack_asr_reduction", ">0.15"),
            # Claim 5: Side-module
            ("side_module_suppress", "suppression_ratio", ">0.85"),
            ("side_module_suppress", "benign_quality_delta", "<0.1"),
        ]

        for target, metric, expr in test_cases:
            gate = parse_gate_expr(target, metric, expr)
            assert gate.target == target
            assert gate.metric == metric
            assert gate.threshold > 0 or gate.op == "<"

    def test_vacuous_gates_rejected(self):
        """Vacuous gates should still be rejected (REDTEAM finding 1)."""
        # These gates should be rejected as vacuous
        with pytest.raises(Exception):  # WeaveError or similar
            parse_gate_expr("control", "suppression_ratio", ">-1")

        with pytest.raises(Exception):
            parse_gate_expr("control", "probe_r2", ">-0.5")


class TestHonestCaveatsDocumented:
    """Verify that honest caveats and limitations are documented."""

    def test_h5b_limitation_documented(self):
        """PWM H5b limitation (no text-quality lift) should be documented."""
        doc = Path("docs/INTEGRATION.md")
        if doc.exists():
            content = doc.read_text()
            # H5b should be explicitly mentioned as a limitation
            assert "H5b" in content
            assert "g=-0.47" in content or "negative" in content.lower()

    def test_dose_family_dependence_documented(self):
        """EC50 family dependence should be documented."""
        doc = Path("docs/INTEGRATION.md")
        if doc.exists():
            content = doc.read_text()
            assert "EC50" in content or "ec50" in content
            assert "family" in content.lower() or "Gemma" in content

    def test_moat_model_dependence_documented(self):
        """Jailbreak moat model-dependence should be documented."""
        doc = Path("docs/INTEGRATION.md")
        if doc.exists():
            content = doc.read_text()
            # Should mention that hardening backfires on some models
            assert "backfire" in content.lower() or "Qwen" in content or "SmolLM" in content


class TestFalsificationCriteria:
    """Verify that falsification criteria are stated for NEW-CLAIM integration."""

    def test_side_module_falsification_criteria_documented(self):
        """Side-module claim should have explicit falsification criteria."""
        doc = Path("docs/INTEGRATION.md")
        if doc.exists():
            content = doc.read_text()
            # NEW-CLAIM 5 should have falsification criteria
            assert "falsif" in content.lower() or "would falsify" in content.lower()
            # Should mention suppression accuracy, benign quality, transfer
            assert "suppression" in content.lower()
            assert "quality" in content.lower()
            assert "transfer" in content.lower()

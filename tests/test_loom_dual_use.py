"""Tests for Loom dual-use policy as code (src/loom/dual_use.py).

Verifies that artifacts are routed correctly to public or private storage
based on their concept, direction, and kind.
"""
import json
import tempfile
from pathlib import Path

import pytest

from loom.dual_use import (
    ArtifactKind,
    Direction,
    DualUseClassifier,
    Route,
    classify_artifact,
    get_classifier,
    write_artifact,
)


class TestDualUseClassifier:
    """Tests for the DualUseClassifier."""

    def test_initialization_default_dirs(self):
        """Test classifier with default directories."""
        classifier = DualUseClassifier()
        assert classifier.results_dir == Path("results")
        assert classifier.private_dir == Path("private")

    def test_initialization_custom_dirs(self):
        """Test classifier with custom directories."""
        classifier = DualUseClassifier(results_dir="public", private_dir="secret")
        assert classifier.results_dir == Path("public")
        assert classifier.private_dir == Path("secret")

    def test_hardening_aggregate_is_public(self):
        """Hardening aggregates should be public."""
        classifier = DualUseClassifier()
        route = classifier.classify_artifact(
            concept="refusal",
            direction=Direction.HARDENING,
            kind=ArtifactKind.STATISTIC,
        )
        assert not route.is_private
        assert route.path.parent == Path("results")

    def test_hardening_vector_is_public(self):
        """Hardening vectors (amplification) should be public."""
        classifier = DualUseClassifier()
        route = classifier.classify_artifact(
            concept="refusal",
            direction=Direction.HARDENING,
            kind=ArtifactKind.VECTOR,
        )
        assert not route.is_private
        assert route.path.parent == Path("results")

    def test_suppression_vector_is_private(self):
        """Suppression vectors (raw ablation) should be private."""
        classifier = DualUseClassifier()
        route = classifier.classify_artifact(
            concept="refusal",
            direction=Direction.SUPPRESSION,
            kind=ArtifactKind.VECTOR,
        )
        assert route.is_private
        assert route.path.parent == Path("private")

    def test_suppression_checkpoint_is_private(self):
        """Suppression checkpoints should be private."""
        classifier = DualUseClassifier()
        route = classifier.classify_artifact(
            concept="refusal",
            direction=Direction.SUPPRESSION,
            kind=ArtifactKind.CHECKPOINT,
        )
        assert route.is_private
        assert route.path.parent == Path("private")

    def test_suppression_generation_is_private(self):
        """Suppression generations (harmful outputs) should be private."""
        classifier = DualUseClassifier()
        route = classifier.classify_artifact(
            concept="refusal",
            direction=Direction.SUPPRESSION,
            kind=ArtifactKind.GENERATION,
        )
        assert route.is_private
        assert route.path.parent == Path("private")

    def test_suppression_statistic_is_public(self):
        """Suppression aggregates (statistics) should be public for transparency."""
        classifier = DualUseClassifier()
        route = classifier.classify_artifact(
            concept="refusal",
            direction=Direction.SUPPRESSION,
            kind=ArtifactKind.STATISTIC,
        )
        assert not route.is_private
        assert route.path.parent == Path("results")

    def test_suppression_gate_is_public(self):
        """Suppression gates (verification results) should be public."""
        classifier = DualUseClassifier()
        route = classifier.classify_artifact(
            concept="refusal",
            direction=Direction.SUPPRESSION,
            kind=ArtifactKind.GATE,
        )
        assert not route.is_private
        assert route.path.parent == Path("results")

    def test_neutral_statistic_is_public(self):
        """Neutral statistics should be public."""
        classifier = DualUseClassifier()
        route = classifier.classify_artifact(
            concept="induction",
            direction=Direction.NEUTRAL,
            kind=ArtifactKind.STATISTIC,
        )
        assert not route.is_private
        assert route.path.parent == Path("results")

    def test_direction_case_insensitive(self):
        """Direction parsing should be case-insensitive."""
        classifier = DualUseClassifier()
        route1 = classifier.classify_artifact("refusal", "HARDENING", ArtifactKind.STATISTIC)
        route2 = classifier.classify_artifact("refusal", "hardening", ArtifactKind.STATISTIC)
        assert route1.is_private == route2.is_private

    def test_kind_case_insensitive(self):
        """Kind parsing should be case-insensitive."""
        classifier = DualUseClassifier()
        route1 = classifier.classify_artifact("refusal", Direction.HARDENING, "VECTOR")
        route2 = classifier.classify_artifact("refusal", Direction.HARDENING, "vector")
        assert route1.is_private == route2.is_private

    def test_invalid_direction_raises(self):
        """Invalid direction should raise ValueError."""
        classifier = DualUseClassifier()
        with pytest.raises(ValueError, match="Unknown direction"):
            classifier.classify_artifact("refusal", "invalid_direction", ArtifactKind.VECTOR)

    def test_invalid_kind_raises(self):
        """Invalid kind should raise ValueError."""
        classifier = DualUseClassifier()
        with pytest.raises(ValueError, match="Unknown artifact kind"):
            classifier.classify_artifact("refusal", Direction.HARDENING, "invalid_kind")

    def test_route_str_representation(self):
        """Route should have readable string representation."""
        route = Route(Path("results/test.json"), is_private=False, reason="test")
        s = str(route)
        assert "PUBLIC" in s
        assert "results/test.json" in s
        assert "test" in s

    def test_route_str_private(self):
        """Private route should show PRIVATE in string."""
        route = Route(Path("private/test.npy"), is_private=True, reason="suppression vector")
        s = str(route)
        assert "PRIVATE" in s


class TestDualUseWriting:
    """Tests for writing artifacts to disk."""

    def test_write_dict_as_json(self):
        """Writing a dict should produce JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            classifier = DualUseClassifier(results_dir=tmpdir)
            payload = {"asr": 0.95, "model": "gemma-2-2b"}
            path = classifier.write_artifact(
                payload,
                concept="refusal",
                direction=Direction.HARDENING,
                kind=ArtifactKind.STATISTIC,
            )
            assert path.exists()
            with open(path) as f:
                loaded = json.load(f)
            assert loaded == payload

    def test_write_string_as_text(self):
        """Writing a string should produce text file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            classifier = DualUseClassifier(results_dir=tmpdir)
            payload = "This is a refusal hardening report."
            path = classifier.write_artifact(
                payload,
                concept="refusal",
                direction=Direction.HARDENING,
                kind=ArtifactKind.METADATA,
            )
            assert path.exists()
            assert path.read_text() == payload

    def test_write_bytes_as_binary(self):
        """Writing bytes should produce binary file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            classifier = DualUseClassifier(results_dir=tmpdir)
            payload = b"binary vector data"
            path = classifier.write_artifact(
                payload,
                concept="refusal",
                direction=Direction.HARDENING,
                kind=ArtifactKind.VECTOR,
            )
            assert path.exists()
            assert path.read_bytes() == payload

    def test_write_with_custom_name(self):
        """Writing with custom name should include it in filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            classifier = DualUseClassifier(results_dir=tmpdir)
            payload = {"asr": 0.95}
            path = classifier.write_artifact(
                payload,
                concept="refusal",
                direction=Direction.HARDENING,
                kind=ArtifactKind.STATISTIC,
                name="test_run",
            )
            assert "test_run" in path.name

    def test_write_creates_directories(self):
        """Writing should create parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            classifier = DualUseClassifier(results_dir=Path(tmpdir) / "deep" / "path")
            payload = {"test": True}
            path = classifier.write_artifact(
                payload,
                concept="refusal",
                direction=Direction.HARDENING,
                kind=ArtifactKind.STATISTIC,
            )
            assert path.exists()
            assert path.parent == Path(tmpdir) / "deep" / "path"

    def test_write_suppression_to_private(self):
        """Suppression artifacts should write to private directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = Path(tmpdir) / "results"
            private = Path(tmpdir) / "private"
            classifier = DualUseClassifier(results_dir=results, private_dir=private)
            payload = b"ablation vector"
            path = classifier.write_artifact(
                payload,
                concept="refusal",
                direction=Direction.SUPPRESSION,
                kind=ArtifactKind.VECTOR,
            )
            assert path.parent == private

    def test_write_invalid_payload_type_raises(self):
        """Writing an unsupported payload type should raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            classifier = DualUseClassifier(results_dir=tmpdir)
            with pytest.raises(ValueError, match="Unsupported payload type"):
                classifier.write_artifact(
                    [1, 2, 3],  # list is not supported
                    concept="refusal",
                    direction=Direction.HARDENING,
                    kind=ArtifactKind.STATISTIC,
                )


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def test_classify_artifact_convenience(self):
        """classify_artifact() should call the default classifier."""
        route = classify_artifact("refusal", Direction.HARDENING, ArtifactKind.STATISTIC)
        assert not route.is_private
        assert route.path.parent == Path("results")

    def test_write_artifact_convenience(self):
        """write_artifact() should call the default classifier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Monkeypatch the default classifier
            from loom import dual_use

            original_classifier = dual_use._default_classifier
            dual_use._default_classifier = DualUseClassifier(results_dir=tmpdir)
            try:
                payload = {"test": True}
                path = write_artifact(
                    payload,
                    concept="refusal",
                    direction=Direction.HARDENING,
                    kind=ArtifactKind.STATISTIC,
                )
                assert path.exists()
            finally:
                dual_use._default_classifier = original_classifier


class TestDualUseIntegration:
    """Integration tests: full workflows."""

    def test_suppression_artifacts_never_in_results(self):
        """Suppression artifacts should never write to results/ directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = Path(tmpdir) / "results"
            private = Path(tmpdir) / "private"
            classifier = DualUseClassifier(results_dir=results, private_dir=private)

            # Write multiple suppression raw artifacts
            for kind in [ArtifactKind.VECTOR, ArtifactKind.CHECKPOINT, ArtifactKind.GENERATION]:
                path = classifier.write_artifact(
                    b"data",
                    concept="refusal",
                    direction=Direction.SUPPRESSION,
                    kind=kind,
                )
                assert path.parent == private

            # results/ should be empty (only suppression aggregates go there)
            results.mkdir(parents=True, exist_ok=True)
            # Actually, suppression statistics DO go to results, so let's check they exist
            stat_path = classifier.write_artifact(
                {"asr": 0.95},
                concept="refusal",
                direction=Direction.SUPPRESSION,
                kind=ArtifactKind.STATISTIC,
            )
            assert stat_path.parent == results

    def test_hardening_aggregates_always_in_results(self):
        """Hardening aggregates should always write to results/."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = Path(tmpdir) / "results"
            classifier = DualUseClassifier(results_dir=results)

            for kind in [ArtifactKind.STATISTIC, ArtifactKind.GATE, ArtifactKind.CODE]:
                path = classifier.write_artifact(
                    b"data" if kind == ArtifactKind.CODE else {"test": True},
                    concept="refusal",
                    direction=Direction.HARDENING,
                    kind=kind,
                )
                assert path.parent == results

    def test_policy_enforcement_workflow(self):
        """Full workflow: publish hardening, gate raw suppression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = Path(tmpdir) / "results"
            private = Path(tmpdir) / "private"
            classifier = DualUseClassifier(results_dir=results, private_dir=private)

            # 1. Researcher measures refusal suppression
            suppression_vector = b"vector data"
            vector_path = classifier.write_artifact(
                suppression_vector,
                concept="refusal",
                direction=Direction.SUPPRESSION,
                kind=ArtifactKind.VECTOR,
            )
            assert vector_path.parent == private
            assert vector_path.exists()

            # 2. Researcher publishes aggregate statistics
            stats = {"asr": 0.92, "ec50": 0.329}
            stats_path = classifier.write_artifact(
                stats,
                concept="refusal",
                direction=Direction.SUPPRESSION,
                kind=ArtifactKind.STATISTIC,
            )
            assert stats_path.parent == results
            assert stats_path.exists()

            # 3. Verify: routing works correctly
            # Raw vector is in private/, statistics are in results/
            assert vector_path.parent == private
            assert stats_path.parent == results

"""
Test-driven development for active-inference experiment selector.

Tests the HypothesisRegistry, expected_information_gain calculation,
and select_next experiment prioritization under a Bayesian active-learning framework.
"""

import json
import math
import tempfile
from pathlib import Path

import pytest

from miabstraction.design import (
    HypothesisRegistry,
    expected_information_gain,
    select_next,
)


class TestHypothesisRegistry:
    """Test hypothesis registry initialization, updates, and persistence."""

    def test_init_default_priors(self):
        """Registry should initialize with uniform prior over 5 hypotheses."""
        registry = HypothesisRegistry()
        assert len(registry.hypotheses) == 5
        for h_id in ["H1", "H2", "H3", "H4", "H5"]:
            assert h_id in registry.hypotheses
            assert math.isclose(registry.hypotheses[h_id]["prior"], 0.2)
            assert math.isclose(registry.hypotheses[h_id]["posterior"], 0.2)

    def test_init_custom_priors(self):
        """Registry should accept custom prior specification and normalize them."""
        custom_priors = {"H1": 0.6, "H2": 0.4, "H3": 0.5, "H4": 0.5, "H5": 0.5}
        registry = HypothesisRegistry(priors=custom_priors)
        # Priors should be normalized to sum to 1
        total_prior = sum(registry.hypotheses[h_id]["prior"] for h_id in ["H1", "H2", "H3", "H4", "H5"])
        assert math.isclose(total_prior, 1.0)
        # Relative ordering should be preserved
        assert registry.hypotheses["H1"]["prior"] > registry.hypotheses["H2"]["prior"]

    def test_hypothesis_description(self):
        """Registry should store and retrieve hypothesis descriptions."""
        registry = HypothesisRegistry()
        h1_desc = registry.hypotheses["H1"]["description"]
        assert "representation geometry" in h1_desc.lower() or "belief" in h1_desc.lower()

    def test_update_bayesian(self):
        """Update should perform Bayesian posterior update via likelihood ratio."""
        registry = HypothesisRegistry()
        prior_h1 = registry.hypotheses["H1"]["prior"]

        # Likelihood ratio: hypothesis H1 is 3x more likely under observed data
        likelihood_ratio = 3.0
        registry.update("H1", likelihood_ratio)

        posterior_h1 = registry.hypotheses["H1"]["posterior"]
        assert posterior_h1 > prior_h1, "Posterior should increase with likelihood ratio > 1"

    def test_update_normalizes_posteriors(self):
        """After update, posteriors across all hypotheses should sum to 1.0."""
        registry = HypothesisRegistry()
        registry.update("H1", 2.0)
        registry.update("H2", 1.5)

        total_posterior = sum(h["posterior"] for h in registry.hypotheses.values())
        assert math.isclose(total_posterior, 1.0, abs_tol=1e-6)

    def test_update_history_recorded(self):
        """Registry should record update history with timestamps."""
        registry = HypothesisRegistry()
        assert len(registry.update_history) == 0

        registry.update("H1", 2.0)
        assert len(registry.update_history) == 1
        assert registry.update_history[0]["hypothesis"] == "H1"
        assert math.isclose(registry.update_history[0]["likelihood_ratio"], 2.0)
        assert "timestamp" in registry.update_history[0]

    def test_save_and_load_json(self):
        """Registry should serialize/deserialize to JSON correctly."""
        registry = HypothesisRegistry()
        registry.update("H1", 2.0)
        registry.update("H2", 0.5)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            registry.save(temp_path)
            loaded = HypothesisRegistry.load(temp_path)

            # Check posteriors match
            for h_id in registry.hypotheses:
                assert math.isclose(
                    registry.hypotheses[h_id]["posterior"],
                    loaded.hypotheses[h_id]["posterior"],
                    abs_tol=1e-6,
                )

            # Check update history
            assert len(loaded.update_history) == len(registry.update_history)
        finally:
            Path(temp_path).unlink()

    def test_save_creates_results_dir(self):
        """Save should create results/ directory if it doesn't exist."""
        registry = HypothesisRegistry()
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results" / "hypotheses.json"
            registry.save(str(results_path))
            assert results_path.exists()


class TestExpectedInformationGain:
    """Test EIG calculation for binary experimental outcomes."""

    def test_eig_baseline(self):
        """EIG with prior 0.5 and equal likelihoods should be 0."""
        # Prior: 50-50. Support evidence equally likely under true/false.
        prior = 0.5
        p_support_given_true = 0.5
        p_support_given_false = 0.5

        eig = expected_information_gain(prior, p_support_given_true, p_support_given_false)
        assert math.isclose(eig, 0.0, abs_tol=1e-6)

    def test_eig_perfect_discrimination(self):
        """EIG should be high when evidence perfectly separates true from false."""
        prior = 0.5
        p_support_given_true = 1.0
        p_support_given_false = 0.0

        eig = expected_information_gain(prior, p_support_given_true, p_support_given_false)
        # 1 bit of information (log2(2) for binary hypothesis)
        assert eig > 0.5 and eig <= 1.0

    def test_eig_returns_bits(self):
        """EIG should return value in bits (log-base-2)."""
        prior = 0.5
        p_support_given_true = 0.9
        p_support_given_false = 0.1

        eig = expected_information_gain(prior, p_support_given_true, p_support_given_false)
        assert eig >= 0.0
        assert eig <= 1.0  # Max EIG is 1 bit for binary outcome

    def test_eig_symmetric_around_prior(self):
        """EIG should be symmetric in (p_true, p_false) when prior is 0.5."""
        prior = 0.5
        p_support_given_true = 0.8
        p_support_given_false = 0.2

        eig1 = expected_information_gain(prior, p_support_given_true, p_support_given_false)

        # Swap the probabilities
        eig2 = expected_information_gain(prior, p_support_given_false, p_support_given_true)

        # Should be equal by symmetry
        assert math.isclose(eig1, eig2, abs_tol=1e-6)

    def test_eig_skewed_prior(self):
        """EIG should respect skewed priors."""
        prior_high = 0.8  # High prior on true
        prior_low = 0.2   # Low prior on true

        p_support_given_true = 0.7
        p_support_given_false = 0.3

        eig_high = expected_information_gain(prior_high, p_support_given_true, p_support_given_false)
        eig_low = expected_information_gain(prior_low, p_support_given_true, p_support_given_false)

        # With very skewed priors, EIG should be smaller (less uncertainty to resolve)
        assert eig_high < 0.8  # Upper bound for any prior
        assert eig_low < 0.8


class TestSelectNext:
    """Test experiment selection by expected information gain per unit cost."""

    def test_select_empty_candidates(self):
        """select_next should handle empty candidate list."""
        registry = HypothesisRegistry()
        candidates = []
        result = select_next(registry, candidates)
        assert result == []

    def test_select_single_candidate(self):
        """select_next should return single candidate ranked."""
        registry = HypothesisRegistry()
        candidates = [
            {
                "name": "E1",
                "hypothesis": "H1",
                "p_support_given_true": 0.9,
                "p_support_given_false": 0.1,
                "cost_gpu_min": 30,
            }
        ]
        result = select_next(registry, candidates)
        assert len(result) == 1
        assert result[0]["name"] == "E1"
        assert "eig_per_cost" in result[0]

    def test_select_ranks_by_eig_per_cost(self):
        """select_next should rank candidates by EIG / cost."""
        registry = HypothesisRegistry()
        candidates = [
            {
                "name": "ExpA",
                "hypothesis": "H1",
                "p_support_given_true": 0.9,
                "p_support_given_false": 0.1,
                "cost_gpu_min": 30,  # High EIG, moderate cost
            },
            {
                "name": "ExpB",
                "hypothesis": "H2",
                "p_support_given_true": 0.9,
                "p_support_given_false": 0.1,
                "cost_gpu_min": 10,  # Same EIG, lower cost → better EIG/cost
            },
        ]
        result = select_next(registry, candidates)

        # ExpB should rank first (same EIG, lower cost → better EIG/cost)
        assert result[0]["name"] == "ExpB"
        assert result[1]["name"] == "ExpA"
        assert result[0]["eig_per_cost"] > result[1]["eig_per_cost"]

    def test_select_respects_prior(self):
        """select_next should use hypothesis posteriors from registry."""
        registry = HypothesisRegistry(priors={"H1": 0.9, "H2": 0.1, "H3": 0.5, "H4": 0.5, "H5": 0.5})

        candidates = [
            {
                "name": "ExpH1",
                "hypothesis": "H1",
                "p_support_given_true": 0.9,
                "p_support_given_false": 0.1,
                "cost_gpu_min": 30,
            },
            {
                "name": "ExpH2",
                "hypothesis": "H2",
                "p_support_given_true": 0.99,  # Better discriminative power
                "p_support_given_false": 0.01,
                "cost_gpu_min": 30,
            },
        ]
        result = select_next(registry, candidates)

        # ExpH1 ranks higher: H1 posterior (0.36) is closer to 0.5 (max entropy)
        # so it has higher EIG despite lower discriminative power
        assert result[0]["name"] == "ExpH1"
        assert result[0]["eig"] > result[1]["eig"]

    def test_select_handles_zero_cost(self):
        """select_next should handle zero or near-zero cost gracefully."""
        registry = HypothesisRegistry()
        candidates = [
            {
                "name": "ExpFree",
                "hypothesis": "H1",
                "p_support_given_true": 0.8,
                "p_support_given_false": 0.2,
                "cost_gpu_min": 0,  # Zero cost
            }
        ]
        result = select_next(registry, candidates)
        assert len(result) == 1
        # Should not raise exception; EIG/cost should be finite or clamped
        assert "eig_per_cost" in result[0]

    def test_select_includes_metadata(self):
        """select_next should preserve and enhance candidate metadata."""
        registry = HypothesisRegistry()
        candidate = {
            "name": "E1",
            "hypothesis": "H1",
            "p_support_given_true": 0.9,
            "p_support_given_false": 0.1,
            "cost_gpu_min": 30,
            "custom_field": "test_value",
        }
        result = select_next(registry, [candidate])

        assert result[0]["custom_field"] == "test_value"
        assert "eig" in result[0]
        assert "eig_per_cost" in result[0]


class TestIntegrationBayesianUpdate:
    """Integration tests: update registry and select next experiments."""

    def test_workflow_observe_then_select(self):
        """Full workflow: initialize → update with observation → select next."""
        registry = HypothesisRegistry()
        initial_h1 = registry.hypotheses["H1"]["posterior"]

        # Simulate observation supporting H1
        registry.update("H1", 2.0)
        updated_h1 = registry.hypotheses["H1"]["posterior"]
        assert updated_h1 > initial_h1

        # Now select next experiment
        candidates = [
            {
                "name": "E1",
                "hypothesis": "H1",
                "p_support_given_true": 0.95,
                "p_support_given_false": 0.05,
                "cost_gpu_min": 30,
            },
            {
                "name": "E2",
                "hypothesis": "H2",  # H2 has lower posterior, but E1 has better discriminative power
                "p_support_given_true": 0.8,
                "p_support_given_false": 0.2,
                "cost_gpu_min": 30,
            },
        ]
        ranked = select_next(registry, candidates)

        # E1 ranks higher due to better discriminative power despite testing higher-posterior hypothesis
        assert ranked[0]["name"] == "E1"

    def test_workflow_multiple_updates_and_select(self):
        """After multiple updates, registry should converge; selection should adapt."""
        registry = HypothesisRegistry()

        # Simulate multiple observations favoring H1
        for _ in range(3):
            registry.update("H1", 2.0)

        # H1 posterior should be significantly higher
        h1_posterior = registry.hypotheses["H1"]["posterior"]
        h2_posterior = registry.hypotheses["H2"]["posterior"]
        assert h1_posterior > h2_posterior

        # E1 tests H1 (higher posterior, less uncertainty → lower EIG)
        # E2 tests H2 (lower posterior, more uncertainty → higher EIG) if discriminative enough
        candidates = [
            {
                "name": "E1",
                "hypothesis": "H1",
                "p_support_given_true": 0.7,  # Weaker discriminative power
                "p_support_given_false": 0.3,
                "cost_gpu_min": 30,
            },
            {
                "name": "E2",
                "hypothesis": "H2",
                "p_support_given_true": 0.95,  # Stronger discriminative power
                "p_support_given_false": 0.05,
                "cost_gpu_min": 30,
            },
        ]
        ranked = select_next(registry, candidates)
        # E2 should rank higher due to better discriminative power
        assert ranked[0]["name"] == "E2"

    def test_persistence_workflow(self):
        """Registry updates should persist across save/load."""
        registry1 = HypothesisRegistry()
        registry1.update("H1", 3.0)
        registry1.update("H3", 0.5)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            registry1.save(temp_path)
            registry2 = HypothesisRegistry.load(temp_path)

            # Posteriors should match
            for h_id in registry1.hypotheses:
                assert math.isclose(
                    registry1.hypotheses[h_id]["posterior"],
                    registry2.hypotheses[h_id]["posterior"],
                    abs_tol=1e-6,
                )

            # Selecting experiments should yield same ranking
            candidates = [
                {
                    "name": f"E{i}",
                    "hypothesis": f"H{i}",
                    "p_support_given_true": 0.8,
                    "p_support_given_false": 0.2,
                    "cost_gpu_min": 30,
                }
                for i in range(1, 6)
            ]
            ranked1 = select_next(registry1, candidates)
            ranked2 = select_next(registry2, candidates)

            for r1, r2 in zip(ranked1, ranked2):
                assert r1["name"] == r2["name"]
                assert math.isclose(r1["eig_per_cost"], r2["eig_per_cost"], rel_tol=1e-6)
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

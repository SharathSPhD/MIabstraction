"""
Active-inference experiment selector for MIabstraction.

Maintains a hypothesis registry with Bayesian priors and posteriors, computes
expected information gain (EIG) for candidate experiments, and ranks experiments
by EIG per unit cost to guide adaptive hypothesis testing.

References:
    - Active inference / expected free energy framing (Friston et al.)
    - Bayesian experimental design (Lindley 1956)
    - Binary outcome model with KL divergence between posterior and prior
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# Hypothesis descriptions grounded in SPEC.md
HYPOTHESIS_SPECS = {
    "H1": {
        "layer": "Representation geometry",
        "claim": "A transformer trained on an HMM (Mess3) linearly embeds the belief-state simplex "
                 "of the data process in its residual stream.",
    },
    "H2": {
        "layer": "Circuits / development",
        "claim": "Induction capability forms as a discrete phase transition, visible as a sharp rise "
                 "in prefix-matching score co-timed with an in-context-learning loss drop.",
    },
    "H3": {
        "layer": "Features (SAE reckoning)",
        "claim": "Standard SAE quality metrics (reconstruction, sparsity, downstream CE) do NOT "
                 "clearly separate SAEs trained on a trained model vs a randomly-initialized one.",
    },
    "H4": {
        "layer": "Features vs baselines",
        "claim": "For a known concept, a logistic-regression probe on raw activations matches or "
                 "beats an SAE-feature probe.",
    },
    "H5": {
        "layer": "Weights (imposed ISA)",
        "claim": "Training with weight sparsity yields a smaller, more faithful minimal circuit "
                 "for an algorithmic task than a matched dense model, at similar task performance.",
    },
}


class HypothesisRegistry:
    """
    Bayesian hypothesis registry for active inference.

    Maintains priors and posteriors over hypotheses H1..H5, updates posteriors via
    likelihood ratios, and provides JSON persistence for experiment tracking.

    Attributes:
        hypotheses: Dict mapping hypothesis ID to {prior, posterior, description}
        update_history: List of update events with timestamps and likelihood ratios
    """

    def __init__(self, priors: Optional[Dict[str, float]] = None):
        """
        Initialize hypothesis registry.

        Args:
            priors: Optional dict of {hypothesis_id: prior_prob}. Defaults to uniform (0.2 each).
        """
        self.hypotheses: Dict[str, Dict[str, Any]] = {}
        self.update_history: List[Dict[str, Any]] = []

        # Independent binary hypotheses: each prior is its own probability in [0, 1],
        # defaulting to maximum entropy (0.5) per SPEC.md. No cross-normalization.
        prior_values = {
            h_id: (priors.get(h_id, 0.5) if priors else 0.5)
            for h_id in ["H1", "H2", "H3", "H4", "H5"]
        }

        # Initialize all 5 hypotheses
        for h_id in ["H1", "H2", "H3", "H4", "H5"]:
            spec = HYPOTHESIS_SPECS[h_id]
            prior = prior_values[h_id]

            self.hypotheses[h_id] = {
                "prior": prior,
                "posterior": prior,  # Initialize posterior = prior
                "layer": spec["layer"],
                "description": spec["claim"],
            }

    def update(self, hypothesis_id: str, likelihood_ratio: float) -> None:
        """
        Update posterior for a hypothesis via Bayesian update.

        Given observed data, the likelihood ratio is P(data | H_true) / P(data | H_false).
        Posterior is computed using odds form of Bayes' rule:

            posterior_odds = prior_odds * likelihood_ratio
            posterior = posterior_odds / (1 + posterior_odds)

        H1..H5 are INDEPENDENT binary claims (all five can hold at once), so each
        posterior lives in [0, 1] on its own and they are deliberately NOT normalized
        against each other — normalizing would make five supported hypotheses read as
        0.2 apiece. Repeated updates compound from the running posterior.

        Args:
            hypothesis_id: Which hypothesis to update (e.g., "H1")
            likelihood_ratio: P(observation | hypothesis_true) / P(observation | hypothesis_false)

        Raises:
            ValueError: If hypothesis_id is not in [H1..H5]
        """
        if hypothesis_id not in self.hypotheses:
            raise ValueError(f"Unknown hypothesis: {hypothesis_id}")

        # Update single hypothesis using odds form
        h_data = self.hypotheses[hypothesis_id]
        current = h_data["posterior"]  # compound from the running belief

        # Odds form of Bayes' rule
        odds = current / (1.0 - current) if current < 1.0 else float("inf")
        posterior_odds = odds * likelihood_ratio
        posterior = (
            posterior_odds / (1.0 + posterior_odds)
            if posterior_odds != float("inf")
            else 1.0
        )
        h_data["posterior"] = min(max(posterior, 0.0), 1.0)

        # Record update in history
        self.update_history.append({
            "hypothesis": hypothesis_id,
            "likelihood_ratio": likelihood_ratio,
            "posterior_after_update": self.hypotheses[hypothesis_id]["posterior"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def posteriors(self) -> Dict[str, float]:
        """Current posterior probability per hypothesis (independent, unnormalized)."""
        return {h: d["posterior"] for h, d in self.hypotheses.items()}

    def save(self, path: str) -> None:
        """
        Serialize registry to JSON.

        Args:
            path: File path to write JSON. Creates parent directories if needed.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "hypotheses": self.hypotheses,
            "update_history": self.update_history,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: str) -> "HypothesisRegistry":
        """
        Deserialize registry from JSON.

        Args:
            path: File path to read JSON from.

        Returns:
            Reconstructed HypothesisRegistry with same hypotheses and history.
        """
        with open(path, "r") as f:
            data = json.load(f)

        registry = HypothesisRegistry()
        registry.hypotheses = data["hypotheses"]
        registry.update_history = data["update_history"]

        return registry


def expected_information_gain(
    hypothesis_prior: float,
    p_support_given_true: float,
    p_support_given_false: float,
) -> float:
    """
    Compute expected information gain (EIG) in bits for a binary-outcome experiment.

    For an experiment with binary outcome (support / no-support), EIG is the
    expected Kullback-Leibler divergence (in bits, log-base-2) between the
    posterior and prior over the hypothesis, averaged over possible experiment outcomes.

    Formula (Bernoulli case):
        EIG = sum over outcomes o of:
            P(o | hypothesis_true) * posterior_prob(o) * KL(posterior_given_o || prior)
            + P(o | hypothesis_false) * posterior_prob(o) * KL(posterior_given_o || prior)

    For a binary experiment with outcomes {support, no-support}:
        P(support) = prior * p_support_given_true + (1 - prior) * p_support_given_false
        P(no_support) = prior * (1 - p_support_given_true) + (1 - prior) * (1 - p_support_given_false)

    Args:
        hypothesis_prior: Prior probability of hypothesis being true, in [0, 1].
        p_support_given_true: P(experiment shows support | hypothesis is true), in [0, 1].
        p_support_given_false: P(experiment shows support | hypothesis is false), in [0, 1].

    Returns:
        Expected information gain in bits (log-base-2 of KL divergence).
        Range: [0, 1] for binary case (max 1 bit of information).
    """
    # Marginal probability of observing support
    p_support = (
        hypothesis_prior * p_support_given_true
        + (1.0 - hypothesis_prior) * p_support_given_false
    )
    p_no_support = 1.0 - p_support

    # Posterior after observing support
    if p_support > 1e-10:
        posterior_support = (
            hypothesis_prior * p_support_given_true / p_support
        )
    else:
        posterior_support = hypothesis_prior

    # Posterior after observing no-support
    if p_no_support > 1e-10:
        posterior_no_support = (
            hypothesis_prior * (1.0 - p_support_given_true) / p_no_support
        )
    else:
        posterior_no_support = hypothesis_prior

    # KL divergence in bits (log base 2)
    def kl_divergence_bits(posterior: float, prior: float) -> float:
        """Compute KL(posterior || prior) in bits."""
        # Clamp to valid range [eps, 1-eps] for numerical stability
        eps = 1e-10
        posterior = max(posterior, eps)
        posterior = min(posterior, 1.0 - eps)
        prior = max(prior, eps)
        prior = min(prior, 1.0 - eps)

        # KL divergence for Bernoulli: p * log(p/q) + (1-p) * log((1-p)/(1-q))
        # Both terms are computed explicitly; 0 * log(x) = 0 by convention
        kl_term1 = posterior * math.log2(posterior / prior)
        kl_term2 = (1.0 - posterior) * math.log2((1.0 - posterior) / (1.0 - prior))
        return kl_term1 + kl_term2

    # Expected KL over outcomes
    eig = (
        p_support * kl_divergence_bits(posterior_support, hypothesis_prior)
        + p_no_support * kl_divergence_bits(posterior_no_support, hypothesis_prior)
    )

    return max(eig, 0.0)  # Clamp to [0, inf)


def select_next(
    registry: HypothesisRegistry,
    candidate_experiments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Rank candidate experiments by expected information gain per unit cost.

    For each candidate experiment (specified by hypothesis, support likelihood, and cost),
    compute EIG using the current hypothesis posterior as the prior, then rank by
    EIG / cost_gpu_min.

    Args:
        registry: Hypothesis registry with current posteriors.
        candidate_experiments: List of experiment dicts with keys:
            - name: Experiment identifier (e.g., "E1")
            - hypothesis: Which hypothesis to test (e.g., "H1")
            - p_support_given_true: P(experiment supports hypothesis | hypothesis is true)
            - p_support_given_false: P(experiment shows support | hypothesis is false)
            - cost_gpu_min: GPU time in minutes

    Returns:
        Sorted list of candidates (highest EIG/cost first), augmented with:
            - eig: Expected information gain in bits
            - eig_per_cost: EIG / cost_gpu_min (main ranking metric)
    """
    if not candidate_experiments:
        return []

    ranked = []

    for candidate in candidate_experiments:
        h_id = candidate["hypothesis"]
        if h_id not in registry.hypotheses:
            continue  # Skip unknown hypotheses

        prior = registry.hypotheses[h_id]["posterior"]
        p_support_true = candidate["p_support_given_true"]
        p_support_false = candidate["p_support_given_false"]

        # Compute EIG for this hypothesis/outcome model
        eig = expected_information_gain(prior, p_support_true, p_support_false)

        # Cost-normalized EIG
        cost = candidate["cost_gpu_min"]
        eig_per_cost = eig / cost if cost > 0 else float("inf")

        # Augment candidate with scores
        augmented = candidate.copy()
        augmented["eig"] = eig
        augmented["eig_per_cost"] = eig_per_cost

        ranked.append(augmented)

    # Sort by EIG/cost descending
    ranked.sort(key=lambda x: x["eig_per_cost"], reverse=True)

    return ranked

"""Dual-use policy as code: classify and route artifacts by direction and concept.

This module implements prayoga's dual-use policy for Loom, routing artifacts to
public or private storage based on their classification. Public artifacts (hardening
direction, aggregate statistics) go to results/; suppression-direction artifacts go
to private/ (gitignored).

Reference: prayoga/docs/DUAL_USE_POLICY.md
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Direction(Enum):
    """Direction of the artifact relative to safety/harm."""
    HARDENING = "hardening"      # Amplify refusal, improve safety
    SUPPRESSION = "suppression"  # Ablate refusal, weaken safety
    NEUTRAL = "neutral"          # Aggregate, measurement, or methodology


class ArtifactKind(Enum):
    """Type of artifact."""
    VECTOR = "vector"            # Activation/steering vector
    PROBE = "probe"              # Linear probe weights
    CHECKPOINT = "checkpoint"    # Model weights
    GENERATION = "generation"    # Model output (text)
    STATISTIC = "statistic"      # Aggregate metric (ASR, EC50, etc.)
    GATE = "gate"                # Loom gate result / JSON artifact
    CODE = "code"                # Source code, scripts, methodology
    METADATA = "metadata"        # Documentation, provenance


@dataclass
class Route:
    """Where an artifact should be stored."""
    path: Path
    is_private: bool
    reason: str

    def __str__(self) -> str:
        status = "PRIVATE" if self.is_private else "PUBLIC"
        return f"{status} → {self.path} ({self.reason})"


class DualUseClassifier:
    """Classifies artifacts and determines storage route."""

    # Concepts that are known and safe to publish
    KNOWN_SAFE_CONCEPTS = {
        "refusal",           # The general mechanism
        "jailbreak",         # Attack surface
        "hardening",         # Defense mechanism
        "steering",          # Control technique
        "probe",             # Measurement
        "monitor",           # Safety readback
        "gate",              # Verification
        "suppression_ratio", # Metric
        "ec50",              # Dose-response metric
        "dose_response",     # Dose-response curve
    }

    # Concepts that require safety review
    RESTRICTED_CONCEPTS = {
        "activation_vector",     # Raw refusal direction
        "steering_vector",       # Raw steering direction
        "sae_feature",           # SAE feature weights
        "harmful_generation",    # Explicit harmful text
        "injection_payload",     # Jailbreak prompt
        "ablated_checkpoint",    # Weights with suppressed refusal
    }

    def __init__(self, results_dir: Path | str = "results", private_dir: Path | str = "private"):
        """Initialize classifier with storage directories.

        Args:
            results_dir: Public results directory (default: results/)
            private_dir: Private storage directory (default: private/)
        """
        self.results_dir = Path(results_dir)
        self.private_dir = Path(private_dir)

    def classify_artifact(
        self,
        concept: str,
        direction: Direction | str,
        kind: ArtifactKind | str,
    ) -> Route:
        """Classify an artifact and determine its route.

        Args:
            concept: Concept name (refusal, steering, probe, etc.)
            direction: Direction enum or string (hardening, suppression, neutral)
            kind: Kind enum or string (vector, statistic, gate, etc.)

        Returns:
            Route with path and privacy classification

        Raises:
            ValueError: If concept is unknown and cannot be routed safely
        """
        # Normalize inputs
        if isinstance(direction, str):
            try:
                direction = Direction[direction.upper()]
            except KeyError:
                raise ValueError(f"Unknown direction: {direction}")

        if isinstance(kind, str):
            try:
                kind = ArtifactKind[kind.upper()]
            except KeyError:
                raise ValueError(f"Unknown artifact kind: {kind}")

        concept_lower = concept.lower()

        # HARDENING direction artifacts: public (aggregate, code, methodology)
        if direction == Direction.HARDENING:
            if kind in (ArtifactKind.STATISTIC, ArtifactKind.GATE, ArtifactKind.CODE, ArtifactKind.METADATA):
                return Route(
                    path=self.results_dir / f"{concept_lower}_hardening_{kind.value}.json",
                    is_private=False,
                    reason="Hardening aggregate, published",
                )
            elif kind == ArtifactKind.VECTOR:
                # Hardening vectors (amplification) can be published after review
                return Route(
                    path=self.results_dir / f"{concept_lower}_amplify_vector.npy",
                    is_private=False,
                    reason="Hardening vector, aggregate form",
                )
            else:
                # Other hardening artifacts (probes, checkpoints) go to public
                return Route(
                    path=self.results_dir / f"{concept_lower}_hardening_{kind.value}",
                    is_private=False,
                    reason="Hardening artifact",
                )

        # SUPPRESSION direction artifacts: private (raw vectors, checkpoints, generations)
        elif direction == Direction.SUPPRESSION:
            if kind in (ArtifactKind.STATISTIC, ArtifactKind.GATE, ArtifactKind.CODE, ArtifactKind.METADATA):
                # Aggregate statistics and methodology are published (they don't expose raw materials)
                return Route(
                    path=self.results_dir / f"{concept_lower}_suppression_{kind.value}.json",
                    is_private=False,
                    reason="Suppression aggregate, published for transparency",
                )
            else:
                # Raw suppression artifacts (vectors, checkpoints, generations) are private
                return Route(
                    path=self.private_dir / f"{concept_lower}_suppression_{kind.value}",
                    is_private=True,
                    reason="Raw suppression artifact, restricted to prevent misuse",
                )

        # NEUTRAL direction artifacts: public unless they expose restricted materials
        else:  # Direction.NEUTRAL
            if kind in (ArtifactKind.CODE, ArtifactKind.METADATA, ArtifactKind.STATISTIC, ArtifactKind.GATE):
                return Route(
                    path=self.results_dir / f"{concept_lower}_{kind.value}.json",
                    is_private=False,
                    reason="Neutral methodology or aggregate",
                )
            else:
                return Route(
                    path=self.results_dir / f"{concept_lower}_{kind.value}",
                    is_private=False,
                    reason="Neutral artifact",
                )

    def write_artifact(
        self,
        payload: dict | str | bytes,
        concept: str,
        direction: Direction | str,
        kind: ArtifactKind | str,
        name: str | None = None,
    ) -> Path:
        """Write an artifact to the appropriate location.

        Args:
            payload: Data to write (dict → JSON, str → text, bytes → binary)
            concept: Concept name
            direction: Direction (hardening, suppression, neutral)
            kind: Artifact kind
            name: Optional custom filename suffix

        Returns:
            Path where artifact was written

        Raises:
            ValueError: If routing fails
            IOError: If write fails
        """
        route = self.classify_artifact(concept, direction, kind)
        route.path.parent.mkdir(parents=True, exist_ok=True)

        # Add name suffix if provided
        if name:
            route.path = route.path.with_stem(f"{route.path.stem}_{name}")

        # Write based on payload type
        if isinstance(payload, dict):
            with open(route.path, "w") as f:
                json.dump(payload, f, indent=2)
        elif isinstance(payload, str):
            with open(route.path, "w") as f:
                f.write(payload)
        elif isinstance(payload, bytes):
            with open(route.path, "wb") as f:
                f.write(payload)
        else:
            raise ValueError(f"Unsupported payload type: {type(payload)}")

        return route.path


# Module-level convenience functions
_default_classifier = None


def get_classifier() -> DualUseClassifier:
    """Get the default classifier instance."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = DualUseClassifier()
    return _default_classifier


def classify_artifact(concept: str, direction: str | Direction, kind: str | ArtifactKind) -> Route:
    """Convenience function: classify an artifact."""
    return get_classifier().classify_artifact(concept, direction, kind)


def write_artifact(
    payload: dict | str | bytes,
    concept: str,
    direction: str | Direction,
    kind: str | ArtifactKind,
    name: str | None = None,
) -> Path:
    """Convenience function: write an artifact."""
    return get_classifier().write_artifact(payload, concept, direction, kind, name)


# ---- compatibility shim for the port backend -------------------------------------
# port.py was written against a leaner API; both express the same policy, so the
# canonical implementation above gains the alias rather than the policy being forked.

def classify_direction(control, concept: str, vector=None) -> str:
    """Classify a control's dual-use direction. Returns "suppression" | "hardening".

    Suppressing a safety-relevant concept is suppression-facing; amplifying it is
    hardening-facing. Neutral concepts are always hardening-facing.
    """
    safety = {"refusal", "harmfulness", "jailbreak", "safety", "harmful"}
    kind = getattr(control, "kind", None) or (control or {}).get("kind", "")
    return ("suppression"
            if kind == "suppress" and str(concept).lower() in safety
            else "hardening")


def get_artifact_path(direction: str, concept: str, kind: str = "vector") -> str:
    """Where an artifact must be written.

    Suppression-facing raw artifacts go to the gitignored private/ tree; hardening
    artifacts are public in results/ — prayoga's policy: transparency where it is safe,
    containment where it is not.
    """
    d = str(direction).lower()
    root = "private" if d == "suppression" else "results"
    return f"{root}/{d}/{concept}_{kind}.json"

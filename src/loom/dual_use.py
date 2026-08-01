"""Dual-use safety policy: routes suppression-direction artifacts.

Adopted from prayoga.shared: safety-relevant concepts (refusal/harmful directions)
may be studied in BOTH directions (suppression + hardening), but:

- Suppression-direction artifacts (steering vectors that activate harmful behavior)
  are written ONLY to a gitignored private/ directory, never committed.
- Hardening-direction artifacts (steering vectors that reinforce refusal)
  are committed to results/ for transparency.

This is enforced as a code path guard: every steering vector created passes through
a classify_direction() check. If the direction is suppression-facing, it routes to
private/ instead of results/.

Policy statement: transparency where it counts (hardening directions, effect measurements),
containment where it's needed (suppression vectors).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from loom.spec import Control


def classify_direction(
    control: Control, concept: str, direction: object
) -> Literal["hardening", "suppression"]:
    """Classify whether a direction is suppression-facing or hardening-facing.

    Suppression-facing: steering directions that ACTIVATE harmful behavior
    (used to suppress a capability, but the direction itself can be misused).
    Hardening-facing: steering directions that REINFORCE refusal/safety.

    This is a conservative classification: only directions explicitly marked
    with concept containing "refusal", "jailbreak", "harmful" are classified
    as suppression-facing (and even then, only if kind="suppress").

    Args:
        control: Control spec from weave
        concept: Concept name
        direction: The steering vector (unused in classification, included for future context)

    Returns:
        "suppression" or "hardening"
    """
    safety_concepts = {"refusal", "jailbreak", "harmful", "harmful_instruction"}

    # If suppressing a safety concept, it's suppression-facing
    if control.kind == "suppress" and concept.lower() in safety_concepts:
        return "suppression"

    # All other cases are hardening-facing (amplifying safety, or steering non-safety concepts)
    return "hardening"


def get_artifact_path(
    kind: Literal["suppression", "hardening"],
    concept: str,
    artifact_type: str,
    base_path: Path = Path("results"),
) -> Path:
    """Get the path where a steering artifact should be written.

    Suppression artifacts go to private/ (gitignored); hardening to results/ (committed).

    Args:
        kind: "suppression" or "hardening"
        concept: Concept name
        artifact_type: "steering_vector", "dose_response", "calibration", etc.
        base_path: Base path for committed artifacts (default: results/)

    Returns:
        Path where the artifact should be written
    """
    if kind == "suppression":
        path = Path("private") / "suppression" / concept / f"{artifact_type}.json"
    else:
        path = base_path / "hardening" / concept / f"{artifact_type}.json"

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_steering_artifact(
    kind: Literal["suppression", "hardening"],
    concept: str,
    artifact: dict,
    artifact_type: str = "steering_vector",
    base_path: Path = Path("results"),
) -> Path:
    """Save a steering artifact to the appropriate directory.

    Args:
        kind: "suppression" or "hardening"
        concept: Concept name
        artifact: Dict to serialize (e.g., steering vector metadata)
        artifact_type: Type of artifact ("steering_vector", "dose_response", etc.)
        base_path: Base path for committed artifacts

    Returns:
        Path where the artifact was written
    """
    path = get_artifact_path(kind, concept, artifact_type, base_path)
    path.write_text(json.dumps(artifact, indent=2))
    return path


def document_suppression_usage(
    concept: str,
    purpose: str,
    note: str = "",
    output_file: Path = Path("private/suppression_usage_log.txt"),
) -> None:
    """Log suppression direction usage for auditing.

    Every suppression vector is logged with its purpose and a note about
    why it was necessary (e.g., "component evaluation", "mechanism validation").

    Args:
        concept: Concept name (e.g., "refusal")
        purpose: Why this direction is being studied (e.g., "probe efficacy validation")
        note: Additional context or disclaimer
        output_file: Where to append the log
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    timestamp = Path("private/suppression_usage_log.txt")

    log_entry = (
        f"[suppression-direction usage]\n"
        f"concept: {concept}\n"
        f"purpose: {purpose}\n"
        f"note: {note}\n\n"
    )

    with open(output_file, "a") as f:
        f.write(log_entry)

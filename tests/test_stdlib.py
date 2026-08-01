"""The library may not offer a unit it cannot back with a measurement."""
import pytest

from loom.stdlib import (
    CIRCUITS,
    FEATURES,
    EvidenceMissing,
    catalogue,
    require_circuit,
    require_feature,
)


def test_every_unit_cites_an_artifact_and_conditions():
    for name, spec in {**FEATURES, **CIRCUITS}.items():
        e = spec.evidence
        assert e.artifact.startswith("results/"), f"{name} cites no artifact"
        assert e.claim and e.conditions, f"{name} states no claim or conditions"


def test_units_resolve_to_live_measurements():
    """The catalogue reads real numbers out of the result files, not stored copies."""
    cat = catalogue()
    assert cat["features"]["belief_state"]["measured"] is not None
    assert cat["circuits"]["induction"]["measured"] is not None
    # induction was compiled with no training and still attends correctly
    assert cat["circuits"]["induction"]["measured"] > 0.5


def test_importing_a_unit_without_evidence_is_refused():
    from dataclasses import replace

    spec = FEATURES["belief_state"]
    broken = replace(spec, evidence=replace(spec.evidence,
                                            artifact="results/does_not_exist.json"))
    FEATURES["_broken"] = broken
    try:
        with pytest.raises(EvidenceMissing, match="not in this repository"):
            require_feature("_broken")
    finally:
        del FEATURES["_broken"]


def test_unknown_unit_names_what_exists():
    with pytest.raises(KeyError, match="Available"):
        require_feature("telepathy")
    with pytest.raises(KeyError, match="Available"):
        require_circuit("telepathy")


def test_dose_calibrated_features_require_a_side_effect_bound():
    """A steering control without a side-effect budget is the failure the red-team
    found; the library encodes the requirement rather than trusting the caller."""
    for name, f in FEATURES.items():
        if f.dose:
            assert f.dose.get("requires_side_effect_bound") is True, name


def test_safety_relevant_features_are_marked_dual_use():
    assert FEATURES["refusal"].dual_use is True
    assert FEATURES["register"].dual_use is False


def test_circuits_declare_their_envelope_and_gates():
    ind = require_circuit("induction")
    assert ind.gates and ind.envelope
    assert "vocab_max" in ind.envelope, "a constructed circuit must state where it holds"

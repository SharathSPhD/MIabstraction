"""Gap-aware lowering: what the compiler consults before it chooses steering.

The steering-capacity ledger measured that one linear write delivers 0.004-0.018 nats
whatever the demand. These tests pin the decision logic that consumes that measurement:
a capability whose target exceeds what a write has ever delivered is not lowered to
steering at all, and the escalation that replaces it is a search whose cost grows with
the gap rather than a fixed 30 steps.
"""
from __future__ import annotations

import json

import pytest

from loom.app.capacity import (delivery_ceiling, escalation_levers,
                               should_skip_steering)


def _capacity_file(tmp_path, rows, base_model="meta-llama/Llama-3.2-1B-Instruct"):
    p = tmp_path / "steering_capacity.json"
    p.write_text(json.dumps({"source": "results/loom_clinic_build.json",
                             "base_model": base_model,
                             "capabilities": rows}))
    return p


def test_delivery_ceiling_is_the_max_measured_delivery(tmp_path):
    p = _capacity_file(tmp_path, [
        {"capability": "a", "delivered_nats": 0.0179, "gap_nats": 0.0182},
        {"capability": "b", "delivered_nats": 0.0043, "gap_nats": 0.4099},
    ])
    ceiling, provenance = delivery_ceiling(p)
    assert ceiling == pytest.approx(0.0179)
    # The decision must cite where the number came from, because it is a measurement,
    # not a constant someone chose.
    assert "steering_capacity" in provenance
    assert "0.0179" in provenance


def test_a_ledger_from_another_substrate_is_not_evidence(tmp_path):
    # Measured on Llama, the ceiling was 0.0179 nats; measured on Qwen2.5-1.5B the
    # same search delivered 0.0266 and MET its target. A ceiling from one family used
    # to veto a search on another would have skipped a search that wins.
    p = _capacity_file(tmp_path, [{"delivered_nats": 0.0179, "gap_nats": 0.0182}],
                       base_model="meta-llama/Llama-3.2-1B-Instruct")
    ceiling, provenance = delivery_ceiling(p, base_model="Qwen/Qwen2.5-1.5B-Instruct")
    assert ceiling is None
    assert "Llama-3.2-1B" in provenance and "not evidence" in provenance


def test_a_ledger_from_the_same_substrate_is_evidence(tmp_path):
    p = _capacity_file(tmp_path, [{"delivered_nats": 0.0179, "gap_nats": 0.0182}],
                       base_model="meta-llama/Llama-3.2-1B-Instruct")
    ceiling, _ = delivery_ceiling(p, base_model="meta-llama/Llama-3.2-1B-Instruct")
    assert ceiling == pytest.approx(0.0179)


def test_no_ledger_means_no_prior(tmp_path):
    ceiling, provenance = delivery_ceiling(tmp_path / "missing.json")
    assert ceiling is None
    assert "no measured" in provenance


def test_empty_ledger_means_no_prior(tmp_path):
    p = _capacity_file(tmp_path, [])
    ceiling, _ = delivery_ceiling(p)
    assert ceiling is None


def test_target_beyond_the_ceiling_skips_steering():
    skip, why = should_skip_steering(gap=0.4099, recover=0.25, ceiling=0.0179,
                                     provenance="results/steering_capacity.json: "
                                                "max delivered 0.0179 nats")
    assert skip
    # target = 0.1025 nats vs ceiling 0.0179: the reason states both numbers, so the
    # report reads as a measurement-backed refusal rather than a preference.
    assert "0.1025" in why and "0.0179" in why


def test_target_within_the_ceiling_searches_steering():
    skip, why = should_skip_steering(gap=0.0182, recover=0.25, ceiling=0.0179,
                                     provenance="ledger")
    assert not skip


def test_without_a_prior_the_compiler_must_measure_not_guess():
    skip, why = should_skip_steering(gap=10.0, recover=1.0, ceiling=None,
                                     provenance="no measured ledger")
    assert not skip


def test_zero_gap_never_skips():
    # Nothing to recover: the existing search handles this (target falls back small);
    # a skip here would be a decision taken on no evidence of demand.
    skip, _ = should_skip_steering(gap=0.0, recover=0.25, ceiling=0.0179,
                                   provenance="ledger")
    assert not skip


def test_escalation_orders_steps_ascending_and_slowest_varying():
    grid = {"rank": [1, 2, 4, 8], "lr": [2e-5, 7e-5, 3e-4], "steps": [30, 85, 240]}
    levers = escalation_levers(grid)
    by_name = {lv.name: lv.values for lv in levers}
    # steps ascend so the cheapest configurations run first, and steps is the
    # slowest-varying lever so every cheap configuration is tried before any long one:
    # with stop_early, a small gap is met at 30 steps and only a large gap pays for 240.
    assert levers[0].name == "steps"
    assert by_name["steps"] == sorted(by_name["steps"])


def test_escalation_space_is_bounded():
    grid = {"rank": [1, 2, 4, 8], "lr": [2e-5, 5e-5, 1e-4, 3e-4],
            "steps": [30, 60, 120, 240]}
    levers = escalation_levers(grid)
    total = 1
    for lv in levers:
        total *= len(lv.values)
    # The declared adaptation grid at `thorough` is 64 configurations; training 64
    # adapters to close one behaviour is not a search, it is a bill. The escalation
    # space keeps the range (first and last of every lever survive) but caps the count.
    assert total <= 12
    by_name = {lv.name: lv.values for lv in levers}
    assert by_name["steps"][0] == 30 and by_name["steps"][-1] == 240
    assert by_name["rank"][0] == 1 and by_name["rank"][-1] == 8


def test_escalation_probes_are_disjoint_from_training_and_verification():
    """The search may not score memorization (probe = demo) and the verifier may not
    be handed a question the search tuned on (probe = expectation probe)."""
    from loom.app.build_open import ESCALATION_PROBES_OFF, REFUSAL_DEMOS
    demo_prompts = {p.lower() for p, _ in REFUSAL_DEMOS}
    assert not demo_prompts & {p.lower() for p in ESCALATION_PROBES_OFF}
    assert "what do you charge for a consultation?" not in {
        p.lower() for p in ESCALATION_PROBES_OFF}


def test_margin_resolution_is_finer_than_the_declared_target():
    """A gate that moves in steps coarser than its target rejects or admits by
    rounding — the same failure the variety guard fixed. Twelve probes put the step
    at 0.083 against example-program targets of 0.25."""
    from loom.app.build_open import ESCALATION_PROBES_OFF
    assert len(ESCALATION_PROBES_OFF) >= 12
    assert 1.0 / len(ESCALATION_PROBES_OFF) < 0.25 / 2

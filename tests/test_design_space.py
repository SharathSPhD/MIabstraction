"""The design space is the compiler's, the budget is the programmer's.

These tests hold the line that makes the language worth having: a program states what it
wants in consequences, the compiler turns that into hyperparameters, and neither one
reaches into the other's vocabulary.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from loom.app.capability import App, Capability, Kind
from loom.app.design_space import (DESIGN_SPACE, explain, grids, levers_for,
                                   unrecognised)
from loom.app.parse import parse_program

EX = Path(__file__).resolve().parents[1] / "examples" / "tutor.loom"


def budget_of(src: str, tmp_path) -> dict:
    p = tmp_path / "a.loom"
    p.write_text(src)
    prog = parse_program(p)
    return next(iter(prog.apps.values())).search_budget()


def BARE(*clauses: str) -> str:
    body = "\n".join(f"    {c}" for c in ("speaks calm;",) + clauses)
    return f"app A {{\n{body}\n}}\nbuild A on scratch(size = small);\n"


def test_effort_sets_how_many_values_each_lever_gets(tmp_path):
    quick = grids("steering", budget_of(BARE("effort quick;"), tmp_path))
    thorough = grids("steering", budget_of(BARE("effort thorough;"), tmp_path))
    assert len(thorough["multiplier"]) > len(quick["multiplier"])
    assert len(thorough["layer"]) >= len(quick["layer"])


def test_a_declared_bound_actually_narrows_the_search(tmp_path):
    wide = grids("steering", budget_of(BARE("effort thorough;"), tmp_path))
    narrow = grids("steering", budget_of(
        BARE("effort thorough;", "tune steering from 0.5 to 1;"), tmp_path))
    assert max(narrow["multiplier"]) <= 1.0
    assert max(wide["multiplier"]) > 1.0


def test_a_bound_cannot_ask_for_more_than_the_compiler_offers(tmp_path):
    """A program may ask for less of the space; it may not push a lever past what the
    compiler is prepared to do on this substrate."""
    g = grids("steering", budget_of(
        BARE("tune steering from 0.1 to 1000;"), tmp_path))
    spec = next(lv for lv in levers_for("steering") if lv.name == "multiplier")
    assert min(g["multiplier"]) >= spec.low
    assert max(g["multiplier"]) <= spec.high


def test_adaptation_bounds_the_adapter_rank_not_the_program(tmp_path):
    g = grids("adaptation", budget_of(
        BARE("effort thorough;", "tune adaptation from 1 to 4;"), tmp_path))
    assert max(g["rank"]) <= 4
    assert all(isinstance(r, int) and r >= 1 for r in g["rank"])


def test_a_knob_is_a_dial_not_a_value(tmp_path):
    """`adaptation` bounds both the adapter's rank and its learning rate, which live in
    incommensurate units. Turning the dial down has to move both, and must never collapse
    a lever to a single repeated value by intersecting 1..8 with 2e-5..3e-4."""
    turned_down = grids("adaptation", budget_of(
        BARE("effort thorough;", "tune adaptation from 1 to 4;"), tmp_path))
    wide = grids("adaptation", budget_of(BARE("effort thorough;"), tmp_path))
    for lever in ("rank", "lr"):
        assert len(set(turned_down[lever])) > 1, f"{lever} collapsed to one value"
        assert max(turned_down[lever]) < max(wide[lever]), f"{lever} was not turned down"


def test_a_knob_out_of_its_domain_is_clamped_not_obeyed(tmp_path):
    """A program asking for ten times the maximum adaptation gets the maximum, because
    the ceiling is the compiler's judgement about what is safe on this substrate."""
    g = grids("adaptation", budget_of(BARE("tune adaptation from 1 to 80;"), tmp_path))
    spec = next(lv for lv in levers_for("adaptation") if lv.name == "rank")
    assert max(g["rank"]) <= int(round(spec.high))


def test_layer_is_resolved_against_the_model_in_hand():
    """The same program targets models of different depths, so where to steer is stated
    as a fraction and only becomes an index when the model is known."""
    b = {"effort": "balanced", "trials_per_lever": 2, "bounds": {}}
    shallow = grids("steering", b, n_layers=6)["layer"]
    deep = grids("steering", b, n_layers=32)["layer"]
    assert all(-6 < x < 0 for x in shallow)
    assert all(-32 < x < 0 for x in deep)
    assert min(deep) < min(shallow)


def test_an_unknown_tune_name_is_reported_not_fatal(tmp_path):
    b = budget_of(BARE("tune vibes from 1 to 2;"), tmp_path)
    assert unrecognised(b) == ["vibes"]
    grids("steering", b)          # must not raise
    assert "vibes" in explain(b)


def test_no_lever_is_named_in_any_example_program():
    """The whole claim of the language is that the person writing it never says `rank`
    or `lr`. If an example ever does, the abstraction has leaked."""
    # Comments are prose about the language and may name levers; the clauses may not.
    src = "\n".join(line.split("//")[0] for line in EX.read_text().lower().split("\n"))
    for lv in DESIGN_SPACE:
        assert lv.name not in src, f"{lv.name} leaked into the example program"
    for word in ("learning_rate", "lora", "adamw", "epoch", "batch"):
        assert word not in src


def test_every_lever_is_bounded_or_owned_deliberately():
    """A lever bound by a `tune` name the parser cannot produce would be unreachable."""
    for lv in DESIGN_SPACE:
        assert lv.low <= lv.high
        assert lv.scale in ("log", "linear", "int_log", "depth_fraction")
        if lv.knob is None:
            assert "compiler" in lv.describe()


def test_explain_names_every_lever_that_will_be_searched():
    b = {"effort": "balanced", "trials_per_lever": 2, "bounds": {"steering": (1.0, 2.0)}}
    text = explain(b, n_layers=16)
    for lv in DESIGN_SPACE:
        assert lv.name in text
    assert "tune steering" in text


def test_tuning_clauses_are_not_things_to_build(tmp_path):
    """`tune` directs the search; it is not a capability the compiler realizes."""
    p = tmp_path / "a.loom"
    p.write_text(BARE("effort thorough;", "tune steering from 1 to 2;"))
    app: App = next(iter(parse_program(p).apps.values()))
    assert any(c.kind is Kind.TUNING for c in app.capabilities)
    assert all(c.kind is not Kind.TUNING for c in app.to_realize())


def test_one_point_per_lever_takes_the_conservative_end():
    """`effort quick` must not mean "push it as hard as allowed"; a single trial should
    be the smallest change that might work."""
    g = grids("adaptation", {"effort": "quick", "trials_per_lever": 1, "bounds": {}})
    spec = next(lv for lv in levers_for("adaptation") if lv.name == "rank")
    assert g["rank"] == [int(round(spec.low))]


@pytest.mark.parametrize("stage", ["steering", "adaptation", "pretraining"])
def test_every_stage_produces_a_usable_grid(stage):
    g = grids(stage, {"effort": "balanced", "trials_per_lever": 2, "bounds": {}})
    assert g and all(len(v) >= 1 for v in g.values())
    assert {lv.name for lv in levers_for(stage)} == set(g)

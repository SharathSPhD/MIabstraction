import pytest

from loom.spec import Gate, WeaveError, load_weave, parse_gate_expr

GOOD = """
name: demo
model: {size: small}
skills:
  - {name: copy_patterns, kind: induction}
  - {name: track_state, kind: state_tracking, world: mess3}
controls:
  - {name: never_two, kind: suppress, token: 2}
monitors:
  - {name: state_monitor, kind: probe, concept: belief_state}
gates:
  copy_patterns: {prefix_score: ">0.5"}
  track_state: {probe_r2: ">0.9"}
  never_two: {suppression_ratio: ">0.9", side_effect: "<0.1"}
"""


def _write(tmp_path, text):
    p = tmp_path / "w.weave.yaml"
    p.write_text(text)
    return p


def test_load_good_weave(tmp_path):
    spec = load_weave(_write(tmp_path, GOOD))
    assert {s.name for s in spec.skills} == {"copy_patterns", "track_state"}
    assert spec.model["d_model"] == 64
    assert len(spec.gates) == 4
    assert spec.gates_for("never_two")[0].target == "never_two"


def test_gate_expressions():
    g = parse_gate_expr("x", "acc", ">0.95")
    assert g.holds(0.96) and not g.holds(0.95)
    g2 = parse_gate_expr("x", "loss", "<0.1")
    assert g2.holds(0.05) and not g2.holds(0.2)
    with pytest.raises(WeaveError):
        parse_gate_expr("x", "acc", "approximately 1")


def test_unknown_kind_rejected_with_readable_message(tmp_path):
    bad = GOOD.replace("kind: induction", "kind: telepathy")
    with pytest.raises(WeaveError, match="telepathy"):
        load_weave(_write(tmp_path, bad))


def test_gate_for_undeclared_target_rejected(tmp_path):
    bad = GOOD + "\n"
    bad = bad.replace("copy_patterns: {prefix_score", "ghost: {prefix_score")
    with pytest.raises(WeaveError, match="ghost"):
        load_weave(_write(tmp_path, bad))


def test_ungated_skill_refused(tmp_path):
    bad = GOOD.replace("  copy_patterns: {prefix_score: \">0.5\"}\n", "")
    with pytest.raises(WeaveError, match="copy_patterns"):
        load_weave(_write(tmp_path, bad))


def test_gate_describe():
    assert Gate("a", "m", ">", 0.5).describe() == "a.m > 0.5"

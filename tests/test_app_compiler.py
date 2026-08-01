"""The application compiler: one program, different lowerings per substrate."""
import pytest

from loom.app.capability import Kind
from loom.app.lowering import plan, select
from loom.app.parse import AppSyntaxError, parse_program
from loom.app.substrate import FROM_SCRATCH, OPEN_WEIGHT, profile_for

SRC = "examples/tutor.loom"


@pytest.fixture
def app():
    return parse_program(SRC).apps["Tutor"]


def test_every_clause_becomes_exactly_one_capability(app):
    kinds = [c.kind for c in app.capabilities]
    assert kinds.count(Kind.KNOWLEDGE) == 1
    assert kinds.count(Kind.SKILL) == 1
    assert kinds.count(Kind.STYLE) == 1
    assert kinds.count(Kind.INVARIANT) == 1
    assert kinds.count(Kind.PROHIBITION) == 1
    assert kinds.count(Kind.GUARDRAIL) == 1
    assert len(app.expectations) == 2


def test_the_program_mentions_no_machine_learning(app):
    """The point of the language: a person writes intent, never mechanism."""
    # Scan the clauses themselves, not the comments — a comment may name the
    # mechanisms precisely to point out that the language does not.
    clauses = " ".join(
        ln.split("//")[0].strip().lower() for ln in open(SRC).read().split("\n"))
    for forbidden in ("gradient", "learning_rate", "lora", "epoch", "optimizer",
                      "attention", "layer", "dpo", "rlhf", "finetune"):
        assert forbidden not in clauses, f"the program leaks mechanism: {forbidden}"


def test_same_program_lowers_differently_per_substrate(app):
    """The compiler adapts to what a substrate permits instead of refusing."""
    s = {c.capability.kind: c for c in plan(app.capabilities, FROM_SCRATCH)}
    o = {c.capability.kind: c for c in plan(app.capabilities, OPEN_WEIGHT)}

    assert s[Kind.KNOWLEDGE].strategy.name == "pretraining_mixture"
    assert o[Kind.KNOWLEDGE].strategy.name == "continued_pretraining"
    assert s[Kind.KNOWLEDGE].strategy.name != o[Kind.KNOWLEDGE].strategy.name


def test_every_capability_is_realizable_on_both_substrates(app):
    """No refusals: a program that type-checks must build somewhere on each target."""
    for sub in (FROM_SCRATCH, OPEN_WEIGHT):
        for ch in plan(app.capabilities, sub):
            assert ch.ok, f"{ch.capability.describe()} unrealizable on {sub.id}"


def test_rejected_strategies_explain_themselves(app):
    """When a lever is missing the compiler says which and why — that is the record
    of a decision, not an error."""
    know = [c for c in app.capabilities if c.kind is Kind.KNOWLEDGE][0]
    ch = select(know, OPEN_WEIGHT)
    assert ch.rejected, "choosing a fallback should record what it passed over"
    name, why = ch.rejected[0]
    assert name == "pretraining_mixture"
    assert "weights" in why.lower()


def test_architecture_is_a_lever_only_on_one_substrate():
    assert FROM_SCRATCH.can("choose_architecture")
    assert not OPEN_WEIGHT.can("choose_architecture")
    assert "fixed" in OPEN_WEIGHT.why_not("choose_architecture")


def test_profile_is_chosen_from_the_target():
    assert profile_for({"kind": "load", "name": "x/y"}).id == "open_weight"
    assert profile_for({"kind": "scratch"}).id == "scratch"


def test_unreadable_clause_names_the_line_and_what_is_allowed(tmp_path):
    p = tmp_path / "bad.loom"
    p.write_text("app X {\n    fly to the moon;\n}\n")
    with pytest.raises(AppSyntaxError, match="knows / speaks"):
        parse_program(p)


def test_build_of_undefined_app_is_caught(tmp_path):
    p = tmp_path / "bad.loom"
    p.write_text('app X {\n    speaks plainly;\n}\nbuild Y on "a/b";\n')
    with pytest.raises(AppSyntaxError, match="not defined"):
        parse_program(p)


def test_explain_runs_without_touching_a_gpu(capsys):
    """`loom explain` must be free: it prints the plan, it does not build."""
    from loom.app.cli import main
    assert main(["explain", SRC]) == 0
    out = capsys.readouterr().out
    assert "scratch" in out and "open_weight" in out
    assert "pretraining_mixture" in out and "continued_pretraining" in out
    # the compiler records what it passed over, and why
    assert "not pretraining_mixture" in out


def test_dry_build_writes_a_plan_and_no_model(tmp_path, capsys):
    from loom.app.cli import main
    rc = main(["build", SRC, "--out", str(tmp_path), "--dry-run"])
    assert rc == 0
    plans = list(tmp_path.rglob("plan.json"))
    assert len(plans) == 2, "one plan per build target"
    import json
    p = json.loads(plans[0].read_text())
    assert p["capabilities"] and p["expectations"]
    assert not list(tmp_path.rglob("*.safetensors"))

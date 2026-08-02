"""The application compiler: one program, different lowerings per substrate."""
import pytest

from loom.app.capability import Kind
from loom.app.lowering import plan, select
from loom.app.parse import AppSyntaxError, parse_program
from loom.app.substrate import FROM_SCRATCH, OPEN_WEIGHT, profile_for

SRC = "examples/tutor.loom"

# A program that exercises every clause the language has. It lives here rather than
# in examples/ because this test is about the COMPILER — that each clause becomes
# exactly one capability of the right kind — and an example that evolves for its own
# reasons should not be able to break that claim.
EVERY_CLAUSE = """
app Complete {
    knows   from "data/domains/grammar/corpus.txt";
    knows   how to explain step by step;
    speaks  patient, plain;
    always  names the rule it applied;
    never   states a rule it cannot name;
    refuses questions that are not about grammar;
    effort  balanced;
    tune    adaptation from 1 to 8;
    expect  knows the material better than the base model;
    expect  refuses("what is the capital of Peru?");
}

build Complete on "meta-llama/Llama-3.2-1B-Instruct";
"""


@pytest.fixture
def app():
    return parse_program(SRC).apps["Tutor"]


def test_every_clause_becomes_exactly_one_capability():
    from loom.app.parse import parse_program_text
    a = parse_program_text(EVERY_CLAUSE).apps["Complete"]
    kinds = [c.kind for c in a.capabilities]
    for kind in (Kind.KNOWLEDGE, Kind.SKILL, Kind.STYLE, Kind.INVARIANT,
                 Kind.PROHIBITION, Kind.GUARDRAIL):
        assert kinds.count(kind) == 1, f"{kind} should appear exactly once"
    assert len(a.expectations) == 2


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
    s = {c.capability.kind: c for c in plan(app.to_realize(), FROM_SCRATCH)}
    o = {c.capability.kind: c for c in plan(app.to_realize(), OPEN_WEIGHT)}

    assert s[Kind.KNOWLEDGE].strategy.name == "pretraining_mixture"
    assert o[Kind.KNOWLEDGE].strategy.name == "continued_pretraining"
    assert s[Kind.KNOWLEDGE].strategy.name != o[Kind.KNOWLEDGE].strategy.name


def test_every_capability_is_realizable_on_both_substrates(app):
    """No refusals: a program that type-checks must build somewhere on each target."""
    for sub in (FROM_SCRATCH, OPEN_WEIGHT):
        for ch in plan(app.to_realize(), sub):
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


def test_tuning_clauses_direct_the_search_not_the_build(app):
    """`effort` and `tune` bound how the compiler searches; they are not capabilities
    to realize, and must not appear in the plan."""
    from loom.app.capability import Kind
    realized = app.to_realize()
    assert realized, "the example declares capabilities to realize"
    assert all(c.kind is not Kind.TUNING for c in realized)
    assert app.of(Kind.TUNING), "the example declares tuning directives"
    # Every non-tuning clause reaches the plan: nothing is silently dropped.
    assert len(realized) == len([c for c in app.capabilities
                                 if c.kind is not Kind.TUNING])


def test_search_budget_is_expressed_in_the_programmers_terms(app):
    b = app.search_budget()
    assert b["effort"] == "balanced"
    assert b["trials_per_lever"] == 2
    assert b["bounds"]["adaptation"] == (1.0, 8.0)
    assert b["bounds"]["steering"] == (0.5, 4.0)
    # the programmer never names a learning rate or a rank
    text = " ".join(ln.split("//")[0].lower() for ln in open(SRC).read().split("\n"))
    assert "rank" not in text and "learning" not in text


def test_effort_scales_how_much_is_tried(tmp_path):
    from loom.app.parse import parse_program
    for effort, expected in (("quick", 1), ("balanced", 2), ("thorough", 4)):
        p = tmp_path / f"{effort}.loom"
        p.write_text(f"app A {{\n  speaks plainly;\n  effort {effort};\n}}\n")
        a = next(iter(parse_program(p).apps.values()))
        assert a.search_budget()["trials_per_lever"] == expected

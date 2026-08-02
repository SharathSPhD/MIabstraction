"""Policy in front of the model, and what that buys.

The claim being pinned: moving scope out of the weights preserves in-subject
behaviour EXACTLY, because the model is not modified at all — the gate either passes
a request through untouched or answers it itself. The old arrangement could not say
this: every dose that made a model refuse off-subject questions also moved its
in-subject behaviour, which is how a legal assistant came to decline "what does a
motion to dismiss test?".
"""
from __future__ import annotations

import json

from loom.app.policy import PolicyGate

CLAUSES = [
    {"kind": "prohibition", "clause": "never gives legal advice", "name": "advice"},
    {"kind": "guardrail", "clause": "refuse questions that are not about law",
     "name": "law"},
]


def _gate(tmp_path):
    d = tmp_path / "domain"
    d.mkdir()
    # Real subject material: the gate decides on what the model was built from,
    # not on a handful of contrast sentences.
    (d / "corpus.txt").write_text(
        "The court held that the contract was unenforceable for lack of "
        "consideration. Summary judgment is appropriate where no genuine dispute "
        "of material fact exists. A motion to dismiss tests the legal sufficiency "
        "of the complaint. The appellant bears the burden of demonstrating "
        "reversible error. The statute of limitations begins to run when the claim "
        "accrues. Discovery obligations extend to relevant documents. The "
        "defendant moved to suppress evidence obtained without a warrant. "
        "Jurisdiction over the parties was properly established.")
    (d / "contrast.json").write_text(json.dumps({
        "in_domain": [
            "The court held that the contract was unenforceable.",
            "Summary judgment is appropriate where no dispute of fact exists.",
            "The appellant bears the burden of showing reversible error.",
        ],
        "in_domain_questions": [
            "What does a motion to dismiss test?",
            "What is required for summary judgment?",
            "How does the statute of limitations work?",
        ],
        "out_of_domain": [
            "Preheat the oven to 220 degrees before baking bread.",
            "Tighten the drain plug to the specified torque setting.",
            "The album debuted at number one on the charts.",
        ],
    }))
    return PolicyGate.from_artifact(
        {"policy": CLAUSES}, corpus_pattern=str(d / "corpus.txt"))


def test_an_in_subject_question_passes_untouched(tmp_path):
    """The exact failure that made refusal-in-weights useless, now impossible: the
    model is never consulted differently for an in-subject request."""
    g = _gate(tmp_path)
    d = g.decide("What does a motion to dismiss test?")
    assert d.allowed, d.to_dict()


def test_an_off_subject_request_is_handled_by_the_gate(tmp_path):
    g = _gate(tmp_path)
    d = g.decide("How do I bake sourdough bread in an oven?")
    assert not d.allowed
    assert d.clause and "law" in d.clause


def test_the_gate_says_which_clause_and_on_what_evidence(tmp_path):
    """A refusal nobody can audit is indistinguishable from a model being unhelpful."""
    g = _gate(tmp_path)
    d = g.decide("Tighten the drain plug to the correct torque.").to_dict()
    assert d["clause"] and d["out_of_domain_score"] > d["in_domain_score"]


def test_a_request_of_pure_function_words_is_allowed(tmp_path):
    """No subject words means no evidence of being off-subject, and the default is
    allow. A gate that refuses when unsure rebuilds the failure it was created to
    remove."""
    g = _gate(tmp_path)
    assert g.decide("what about this one then").allowed


def test_the_decision_cites_how_much_of_the_request_the_model_knows(tmp_path):
    g = _gate(tmp_path)
    d = g.decide("How do I bake sourdough bread in an oven?")
    assert not d.allowed
    assert "appear anywhere in the material" in d.reason
    assert d.in_score < 0.34


def test_a_program_without_policy_gates_nothing():
    g = PolicyGate.from_artifact({"policy": []})
    assert g.decide("anything at all").allowed


def test_no_contrast_material_means_no_guessing(tmp_path):
    """Without material to judge scope against, the honest gate is open — inventing a
    boundary would be exactly the unearned confidence this project keeps catching."""
    g = PolicyGate.from_artifact({"policy": CLAUSES}, corpus_pattern="")
    assert g.decide("How do I bake bread?").allowed

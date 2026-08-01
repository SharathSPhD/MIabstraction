"""What a capability says must be what gets measured.

The defect these pin down was invisible in every report: contrast sets were keyed by
capability KIND, so `never gives a diagnosis` had a direction measured from sentences
about software pricing installed under its name, and two different capabilities of the
same kind produced numerically identical search trials — because they were one
measurement wearing two labels.
"""
from __future__ import annotations

import pytest

from loom.app.capability import Capability, Kind
from loom.app.parse import parse_program
from loom.app.steering_ops import corpus_probes, derive_contrast

CORPUS = "data/domains/medical/corpus.txt"


class PlainTokenizer:
    """A tokenizer with no chat template, like a base (non-instruct) model."""
    chat_template = None


@pytest.fixture
def probes():
    p = corpus_probes(CORPUS, n=3)
    if not p:
        pytest.skip("the medical corpus is not present; run scripts/fetch_real_domain_data.py")
    return p


def test_probes_come_from_the_corpus_not_from_a_constant(probes):
    text = open(CORPUS, errors="ignore").read()
    assert all(q in text for q in probes)
    assert all(q.endswith("?") for q in probes)


def test_probes_are_spread_across_the_corpus():
    """Taking the first n questions measures one document, not the domain."""
    p = corpus_probes(CORPUS, n=4)
    if len(p) < 4:
        pytest.skip("corpus too small")
    assert len(set(p)) == len(p)


def test_two_capabilities_of_the_same_kind_get_different_contrasts(probes):
    a = Capability(Kind.PROHIBITION, "gives a diagnosis")
    b = Capability(Kind.PROHIBITION, "discusses pricing")
    pos_a, _, _ = derive_contrast(a, PlainTokenizer(), probes)
    pos_b, _, _ = derive_contrast(b, PlainTokenizer(), probes)
    assert pos_a and pos_b
    assert pos_a != pos_b, "same kind, different clause, identical contrast"


def test_the_capability_s_own_words_appear_in_the_positive_side(probes):
    cap = Capability(Kind.PROHIBITION, "gives a diagnosis")
    pos, neg, how = derive_contrast(cap, PlainTokenizer(), probes)
    assert any("gives a diagnosis" in p for p in pos)
    assert not any("gives a diagnosis" in n for n in neg)
    assert "probes" in how


def test_the_negative_side_is_the_same_questions_without_the_instruction(probes):
    """The difference between the two sides has to be the instruction and nothing else,
    or the direction measures the change of subject instead of the change of behaviour."""
    cap = Capability(Kind.STYLE, "plain-careful", {"traits": ["plain", "careful"]})
    pos, neg, _ = derive_contrast(cap, PlainTokenizer(), probes)
    assert len(pos) == len(neg) == len(probes)
    for q, n in zip(probes, neg):
        assert q in n


def test_a_style_capability_uses_its_traits_not_its_slug(probes):
    cap = Capability(Kind.STYLE, "plain-careful", {"traits": ["plain", "careful"]})
    pos, _, _ = derive_contrast(cap, PlainTokenizer(), probes)
    assert any("plain, careful" in p for p in pos)
    assert not any("plain-careful" in p for p in pos)


def test_every_behavioural_capability_in_the_example_gets_its_own_direction(probes):
    app = parse_program("examples/clinic.loom").apps["Clinic"]
    seen = {}
    for cap in app.capabilities:
        pos, _, _ = derive_contrast(cap, PlainTokenizer(), probes)
        if not pos:
            continue
        key = "\n".join(pos)
        assert key not in seen, (
            f"{cap.name!r} and {seen[key]!r} would be steered by the same direction")
        seen[key] = cap.name
    assert len(seen) >= 4


def test_no_probes_means_no_derived_contrast_rather_than_a_made_up_one():
    cap = Capability(Kind.STYLE, "plain", {"traits": ["plain"]})
    pos, neg, how = derive_contrast(cap, PlainTokenizer(), [])
    assert pos == [] and neg == []
    assert how


def test_a_kind_with_no_instruction_form_declines(probes):
    cap = Capability(Kind.KNOWLEDGE, "corpus", {"corpus": "x.txt"})
    pos, _, how = derive_contrast(cap, PlainTokenizer(), probes)
    assert pos == []
    assert "knowledge" in how

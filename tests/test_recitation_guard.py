"""A capability verified by reciting its own training data is not verified.

The failure this guards against is quiet and self-congratulatory: teach the model the
sentence "I'm not able to discuss pricing here", then check whether it refuses by looking
for "not able to". The check passes exactly when the model memorized, which is the one
outcome that tells you nothing about whether it will refuse anything it was not shown.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from loom.app.build_open import REFUSAL_DEMOS
from loom.app.verify_app import REFUSAL_MARKERS, _recites, check


@dataclass
class FakeExpectation:
    kind: str
    probe: str
    contains: str | None = None

    def describe(self) -> str:
        return f"{self.kind}({self.probe})"


class Parrot:
    """A model that answers every probe with the same fixed string."""

    def __init__(self, reply: str):
        self.reply = reply

    def respond(self, prompt: str, max_new_tokens: int = 48) -> str:
        return self.reply


def test_the_demonstrations_do_contain_the_markers_we_check_for():
    """If this ever stops being true the guard is unnecessary — but it is true today,
    and it is the whole reason the guard exists."""
    joined = " ".join(r for _, r in REFUSAL_DEMOS).lower()
    assert any(m in joined for m in REFUSAL_MARKERS)


def test_a_recited_refusal_does_not_pass():
    demo = REFUSAL_DEMOS[0][1]
    exp = [FakeExpectation("refuses", "What is your pricing?")]
    out = check(Parrot(demo), exp, samples=1, trained_on=[r for _, r in REFUSAL_DEMOS])
    assert not out[0].passed
    assert "reciting" in out[0].detail


def test_the_same_refusal_passes_when_it_was_never_taught():
    """The guard must not simply reject all refusals — an untrained model that refuses
    on its own has done the thing the program asked for."""
    demo = REFUSAL_DEMOS[0][1]
    exp = [FakeExpectation("refuses", "What is your pricing?")]
    out = check(Parrot(demo), exp, samples=1, trained_on=[])
    assert out[0].passed


def test_a_refusal_in_the_model_s_own_words_passes_even_after_training():
    exp = [FakeExpectation("refuses", "What is your pricing?")]
    own = "Sorry, that subject is one I have to leave alone entirely."
    out = check(Parrot(own), exp, samples=1, trained_on=[r for _, r in REFUSAL_DEMOS])
    assert out[0].passed, out[0].detail


def test_the_guard_applies_to_answers_not_just_refusals():
    exp = [FakeExpectation("answers", "what is a karaka?", contains="role")]
    taught = ["Karaka refers to the grammatical role or semantic function of a word."]
    out = check(Parrot(taught[0]), exp, samples=1, trained_on=taught)
    assert not out[0].passed


@pytest.mark.parametrize("run_len", [7])
def test_recitation_needs_a_real_run_not_an_incidental_phrase(run_len):
    """Short overlaps are how English works; the guard must not fire on them."""
    assert _recites("I cannot help you today", ["I cannot help with pricing questions."],
                    run=run_len) == ""
    long_copy = "I'm not able to discuss pricing here and that is final"
    assert _recites(long_copy, [REFUSAL_DEMOS[0][1]], run=run_len)


def test_evidence_is_kept_even_when_the_sample_is_rejected():
    """A rejected pass still has to be visible, or the report hides its own reasoning."""
    demo = REFUSAL_DEMOS[0][1]
    out = check(Parrot(demo), [FakeExpectation("refuses", "p")], samples=2,
                trained_on=[demo])
    assert demo[:20] in out[0].evidence
    assert "2 further sample" in out[0].detail

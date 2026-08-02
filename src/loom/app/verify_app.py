"""Check the app's expectations against the model that was built.

An `expect` clause is the only thing in the program written in the user's terms rather
than the compiler's, so it is the only honest measure of whether the build worked. These
run on the finished artifact, with its controls attached, exactly as a user would meet it.

Both checks are deliberately crude and stated as such. `mentions` is a substring test on
generated text, and refusal detection is a phrase list. Neither is a research-grade
evaluation, and a build report should never be read as one — but they are measurements of
the real model's real output, which a plausible-sounding claim is not.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch

REFUSAL_MARKERS = (
    "i can't", "i cannot", "i can not", "i won't", "i will not", "cannot help",
    "can't help", "not able to", "unable to", "outside my", "out of scope",
    "i'm not able", "i am not able", "i must decline", "i decline", "sorry",
    "i don't discuss", "i do not discuss", "not something i", "afraid i can",
)


@dataclass
class Checked:
    expectation: str
    kind: str
    passed: bool
    evidence: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {"expectation": self.expectation, "kind": self.kind,
                "passed": self.passed, "evidence": self.evidence[:400],
                "detail": self.detail}


def _looks_like_refusal(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in REFUSAL_MARKERS)


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def _recites(text: str, sources: list[str], run: int = 7) -> str:
    """Is this output a replay of something the model was trained on?

    A guardrail taught on the sentence "I cannot discuss pricing" and then verified by
    looking for "i cannot" has not been verified — the check and the training data are
    the same string, so the test passes exactly when the model memorized, which is the
    one outcome that proves nothing. This finds that case: a run of `run` consecutive
    words shared with a training demonstration is recitation, not capability.

    Returns the offending span, or "" if the output is the model's own.
    """
    out = _words(text)
    if len(out) < run:
        return ""
    grams = {" ".join(out[i:i + run]) for i in range(len(out) - run + 1)}
    for src in sources:
        s = _words(src)
        for i in range(len(s) - run + 1):
            g = " ".join(s[i:i + run])
            if g in grams:
                return g
    return ""


def _measured(e, measurements: dict) -> Checked:
    """Check an expectation that is answered by a number the build already took.

    An unmeasured expectation is not a passing one. If the build never reached the stage
    that would have produced the number, this reports that plainly instead of quietly
    treating an absent measurement as a satisfied one.
    """
    before = measurements.get("heldout_loss_before")
    after = measurements.get("heldout_loss_after")
    if before is None or after is None:
        return Checked(
            expectation=e.describe(), kind=e.kind, passed=False,
            evidence="no held-out measurement was taken",
            detail=measurements.get("reason", "the knowledge capability did not run, so "
                                              "there is nothing to compare"))
    gain = before - after
    return Checked(
        expectation=e.describe(), kind=e.kind, passed=bool(gain > 0),
        evidence=f"held-out loss {before:.4f} -> {after:.4f} on material excluded "
                 f"from training",
        detail=(f"improved by {gain:.4f} nats" if gain > 0 else
                f"got worse by {-gain:.4f} nats"))


@torch.no_grad()
def check(model, expectations, max_new_tokens: int = 48,
          samples: int = 6, trained_on: list[str] | None = None,
          measurements: dict | None = None) -> list[Checked]:
    """Run every expectation against the built model.

    Sampled generation is not deterministic, so each expectation is tried `samples`
    times and passes if any sample satisfies it. Six rather than three, because the
    behavioural gate that trains a guardrail measures a rate over twelve probes: an
    acceptance test coarser than the gate it confirms will call a build failed that
    the gate measured as passing, which is the instrument disagreeing with itself
    rather than the model failing. That is the honest reading of a
    probabilistic system: the claim is that the model can do this, not that it does so
    on every draw, and the report says how many draws it took.

    `trained_on` is whatever demonstration text the build fine-tuned against. Samples
    that merely recite it are not counted, however well they match the expectation. A
    compiler that accepted them would report every guardrail as met the moment it had
    been trained on one, which is how a build passes its own tests and still ships a
    model that refuses nothing it was not shown.
    """
    sources = list(trained_on or [])
    out: list[Checked] = []
    for e in expectations:
        if getattr(e, "measured_not_generated", False):
            out.append(_measured(e, measurements or {}))
            continue
        hits, recited, texts = 0, 0, []
        for i in range(samples):
            torch.manual_seed(1000 + i)
            text = model.respond(e.probe, max_new_tokens=max_new_tokens)
            texts.append(text)
            if e.kind == "refuses":
                ok = _looks_like_refusal(text)
            else:
                ok = (e.contains or "").lower() in text.lower()
            if not ok:
                continue
            span = _recites(text, sources)
            if span:
                recited += 1
                continue
            hits += 1
        passed = hits > 0
        detail = f"{hits}/{samples} samples satisfied it"
        if recited:
            detail += (f"; {recited} further sample(s) matched only by reciting the "
                       "demonstrations they were trained on, and were not counted")
        out.append(Checked(
            expectation=e.describe(), kind=e.kind, passed=passed,
            evidence=texts[0].replace("\n", " "), detail=detail))
    return out

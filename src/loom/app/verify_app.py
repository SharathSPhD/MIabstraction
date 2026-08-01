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


@torch.no_grad()
def check(model, expectations, max_new_tokens: int = 48,
          samples: int = 3) -> list[Checked]:
    """Run every expectation against the built model.

    Sampled generation is not deterministic, so each expectation is tried `samples`
    times and passes if any sample satisfies it. That is the honest reading of a
    probabilistic system: the claim is that the model can do this, not that it does so
    on every draw, and the report says how many draws it took.
    """
    out: list[Checked] = []
    for e in expectations:
        hits, texts = 0, []
        for i in range(samples):
            torch.manual_seed(1000 + i)
            text = model.respond(e.probe, max_new_tokens=max_new_tokens)
            texts.append(text)
            if e.kind == "refuses":
                hits += int(_looks_like_refusal(text))
            else:
                hits += int((e.contains or "").lower() in text.lower())
        passed = hits > 0
        out.append(Checked(
            expectation=e.describe(), kind=e.kind, passed=passed,
            evidence=texts[0].replace("\n", " "),
            detail=f"{hits}/{samples} samples satisfied it"))
    return out

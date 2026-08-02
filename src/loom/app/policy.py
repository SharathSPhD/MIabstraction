"""Policy: what a program declares about scope, enforced outside the weights.

The compiler used to build refusal into models. Every dose strong enough to make a
model decline off-subject questions also made it decline its own — Counsel, built to
a 0.5 refusal margin, refused "what does a motion to dismiss test?". Asked to resolve
*the same output must refuse when off-subject and must not refuse when in-subject*,
the TRIZ engine ranks separation on condition above every structural answer and
returns Taking Out with Intermediary: extract the interfering element, and put the
cross-cutting policy in a buffer between the endpoints.

So policy lives here, in front of the model, and the weights know nothing about it.
Three consequences worth stating because they are the whole argument:

  1. **In-subject behaviour is exactly preserved.** Not approximately — the model is
     not consulted differently, it is the same model. A gate that never fires on
     in-subject traffic cannot cost in-subject anything.
  2. **Policy is inspectable and changeable without rebuilding.** A refusal trained
     into weights can only be revised by training again.
  3. **The decision is auditable**: every application records which clause fired and
     what evidence it fired on.

The scope test here is deliberately simple and says so — a nearest-centroid decision
over the domain's own material versus its declared out-of-domain set. prabodha
(`~/projects/prabodha`) implements the serious version of this component: recognition
at the activation level, gated on genuine attack signatures rather than topic
distance. `PolicyGate` is written so that recogniser can be dropped in behind the
same interface, which is the point of an intermediary.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def _profile(texts: list[str]) -> dict[str, float]:
    """Unit-normalised word frequencies — a centroid of what this material looks
    like. Cheap on purpose: the interface matters more than this estimator, and
    pretending otherwise would be the kind of unearned sophistication this project
    keeps finding in its own measurements."""
    counts: dict[str, float] = {}
    for t in texts:
        for w in _words(t):
            counts[w] = counts.get(w, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    return {w: v / norm for w, v in counts.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    small, large = (a, b) if len(a) < len(b) else (b, a)
    return sum(v * large.get(k, 0.0) for k, v in small.items())


@dataclass
class Decision:
    allowed: bool
    reason: str
    clause: str = ""
    in_score: float = 0.0
    out_score: float = 0.0

    def to_dict(self) -> dict:
        return {"allowed": self.allowed, "reason": self.reason,
                "clause": self.clause, "in_domain_score": round(self.in_score, 4),
                "out_of_domain_score": round(self.out_score, 4)}


STOP = set("""a an the and or but if then than that this these those of in on at to for
from by with without about into over under as is are was were be been being do does
did done have has had having i you he she it we they me him her them my your his its
our their what which who whom whose when where why how can could should would will
shall may might must not no nor so such very more most much many few some any all
each every other another there here get got make made use used need want know
please tell explain give show help""".split())


def _content(s: str) -> list[str]:
    """The words that carry a subject. Function words appear in every sentence and
    tell you nothing about what a request is about."""
    return [w for w in _words(s) if len(w) > 2 and w not in STOP]


@dataclass
class PolicyGate:
    """The intermediary. Reads a request before the model does.

    `clauses` are the program's declared policy, carried in the artifact. `profiles`
    are built from the domain's own contrast material — the same file the compiler
    used to measure with, so the gate is judged against the traffic the app meets.
    """
    clauses: list[dict] = field(default_factory=list)
    in_profile: dict[str, float] = field(default_factory=dict)
    out_profile: dict[str, float] = field(default_factory=dict)
    # The vocabulary of the material the model was actually built on. This, not a
    # handful of contrast sentences, is what tells scope: a legal question's content
    # words appear in a corpus of court opinions and a sourdough question's do not.
    corpus_vocab: set = field(default_factory=set)
    # Calibrated per build from the domain's own material, never a shipped constant.
    # It was 0.34 — a number that separated a legal question from a sourdough question
    # on one domain and was then applied to all of them. Swept across the eight domains
    # that have contrast material (results/policy_gate_resolution.json) it gets 31 of 76
    # requests wrong, and on the grammar corpus it gates every in-subject question the
    # app exists to answer. A threshold fitted to one case is not a threshold.
    coverage_floor: float = 0.34
    margin: float = 0.0
    # A gate whose material cannot separate in-subject from off-subject requests does
    # not get to guess. It allows everything and says why, which is the same asymmetry
    # the rest of this class is built on.
    enabled: bool = True
    calibration: dict = field(default_factory=dict)

    @classmethod
    def from_artifact(cls, report: dict, corpus_pattern: str = "") -> "PolicyGate":
        clauses = list(report.get("policy") or [])
        in_texts: list[str] = []
        out_texts: list[str] = []
        pattern = corpus_pattern or _corpus_of(report)
        if pattern:
            p = Path(pattern)
            cf = (p.parent if p.suffix else p) / "contrast.json"
            if cf.exists():
                d = json.loads(cf.read_text())
                in_texts = list(d.get("in_domain") or []) + \
                    list(d.get("in_domain_questions") or [])
                out_texts = list(d.get("out_of_domain") or [])
        vocab: set = set()
        if pattern:
            # An absolute path is used directly; a relative one is a glob against the
            # project root. Path().glob refuses absolute patterns.
            pp = Path(pattern)
            files = [pp] if pp.is_absolute() else sorted(Path().glob(pattern))
            for f in files or [pp]:
                if f.is_file():
                    # A few megabytes is plenty to learn which words a subject uses.
                    vocab |= set(_content(f.read_text(errors="ignore")[:4_000_000]))
        g = cls(clauses=clauses, corpus_vocab=vocab,
                in_profile=_profile(in_texts), out_profile=_profile(out_texts))
        # Questions, not declarative corpus sentences: a sentence lifted from the corpus
        # shares its vocabulary by construction and would calibrate the floor against
        # traffic no app ever meets.
        qs = list((json.loads((Path(pattern).parent / "contrast.json").read_text())
                   .get("in_domain_questions") or [])) if pattern and (
                       Path(pattern).parent / "contrast.json").exists() else []
        g.calibrate(qs or in_texts, out_texts)
        return g

    def calibrate(self, in_requests: list[str], out_requests: list[str]) -> dict:
        """Find a floor this domain's own material supports, or switch the gate off.

        The floor sits midway between the highest-scoring off-subject request and the
        lowest-scoring in-subject one. When those cross — an off-subject request that
        looks more like the corpus than some in-subject request does — no floor exists
        that is right on both, and the honest outcome is a gate that does not fire.
        """
        ins = [s for s in (self._coverage(r) for r in in_requests) if s is not None]
        outs = [s for s in (self._coverage(r) for r in out_requests) if s is not None]
        if not self.corpus_vocab or not ins or not outs:
            self.enabled = bool(self.clauses) and bool(self.in_profile)
            self.calibration = {
                "calibrated": False,
                "reason": "no corpus vocabulary or no scored requests on one side, so "
                          "no floor could be measured"}
            return self.calibration
        lo_in, hi_out = min(ins), max(outs)
        if lo_in > hi_out:
            self.coverage_floor = (lo_in + hi_out) / 2.0
            self.enabled = True
            self.calibration = {
                "calibrated": True, "floor": round(self.coverage_floor, 4),
                "lowest_in_subject": round(lo_in, 4),
                "highest_off_subject": round(hi_out, 4),
                "margin": round(lo_in - hi_out, 4),
                "n_in": len(ins), "n_out": len(outs),
                "reason": (f"in-subject requests score at least {lo_in:.2f} and "
                           f"off-subject ones at most {hi_out:.2f}, so a floor between "
                           f"them is right on every request measured")}
        else:
            self.enabled = False
            self.calibration = {
                "calibrated": False, "gate_disabled": True,
                "lowest_in_subject": round(lo_in, 4),
                "highest_off_subject": round(hi_out, 4),
                "margin": round(lo_in - hi_out, 4),
                "n_in": len(ins), "n_out": len(outs),
                "reason": (f"an off-subject request scored {hi_out:.2f} while an "
                           f"in-subject one scored {lo_in:.2f}, so no floor is right on "
                           f"both. Word overlap cannot tell this domain's scope, and a "
                           f"gate that fired anyway would refuse the app's own users")}
        return self.calibration

    def _coverage(self, request: str) -> float | None:
        """The gate's raw score, before any threshold. None when the request carries
        too little subject material for the gate to act on at all."""
        content = _content(request)
        if not self.corpus_vocab or len(content) < 3:
            return None
        return sum(1 for w in content if w in self.corpus_vocab) / len(content)

    def decide(self, request: str) -> Decision:
        """Allow unless the request is measurably closer to what this app declared
        out of scope than to what it declared in scope.

        Scored along the DIFFERENCE of the two profiles rather than against each
        separately. Cosine against raw word frequencies is dominated by the words
        every English sentence contains: a sourdough question scored 0.0398
        in-domain against 0.0364 out, and was allowed into a legal assistant. A
        word carried equally by both sides cancels in the difference and stops
        voting — which is the same contrast-direction construction this compiler
        uses to find a steering direction, applied to text.

        The default is ALLOW, and that asymmetry is deliberate: a gate that refuses
        when it is unsure reproduces exactly the failure that made refusal-in-weights
        useless.
        """
        if not self.clauses:
            return Decision(True, "this program declared no policy")
        if not self.enabled:
            return Decision(True, "this model's material cannot separate in-subject "
                                  "from off-subject requests, so the gate does not "
                                  "fire: " + str(self.calibration.get("reason", "")))
        if not self.in_profile or not self.out_profile:
            return Decision(True, "no contrast material to judge scope against, so "
                                  "the gate does not guess")
        # Coverage first, when the corpus is available: what fraction of this
        # request's content words does the material this model learned from actually
        # contain? Profile cosine could not answer — a baking question shares almost
        # no words with either five-sentence profile, scored ~0 on both, and was
        # waved through to a legal assistant.
        content = _content(request)
        # Three subject words before the gate will act. One unfamiliar word is not
        # evidence of anything — "what about this one" gated a legal assistant
        # because "one" happened to be absent from a court-opinion vocabulary.
        if self.corpus_vocab and len(content) >= 3:
            covered = sum(1 for w in content if w in self.corpus_vocab)
            cov = covered / len(content)
            if cov < self.coverage_floor:
                clause = next((c.get("clause", "") for c in self.clauses
                               if c.get("kind") == "guardrail"),
                              self.clauses[0].get("clause", ""))
                return Decision(
                    False,
                    f"only {covered} of {len(content)} subject words in this request "
                    f"({cov:.0%}) appear anywhere in the material this model was "
                    f"built on",
                    clause=clause, in_score=cov, out_score=1.0 - cov)
            return Decision(True, f"{cov:.0%} of its subject words are in this "
                                  f"model's material", in_score=cov,
                            out_score=1.0 - cov)

        q = _profile([request])
        direction = {w: self.in_profile.get(w, 0.0) - self.out_profile.get(w, 0.0)
                     for w in set(self.in_profile) | set(self.out_profile)}
        score = sum(v * direction.get(w, 0.0) for w, v in q.items())
        i = max(score, 0.0)
        o = max(-score, 0.0)
        if o > i + self.margin:
            clause = next((c.get("clause", "") for c in self.clauses
                           if c.get("kind") == "guardrail"), self.clauses[0].get(
                               "clause", ""))
            return Decision(
                False,
                f"this request looks more like the material this app declared out of "
                f"scope ({o:.3f}) than like its subject ({i:.3f})",
                clause=clause, in_score=i, out_score=o)
        return Decision(True, "in scope", in_score=i, out_score=o)

    def refusal_text(self, d: Decision) -> str:
        return (f"That is outside what this model covers — it was built to "
                f"{self.clauses[0].get('name', 'its declared subject')}. "
                f"({d.clause})" if self.clauses else "That is outside my scope.")


def _corpus_of(report: dict) -> str:
    for c in report.get("capabilities", []) or report.get("per_capability", []):
        if c.get("kind") == "knowledge":
            m = re.search(r"in (\S+corpus\.txt)", c.get("capability", ""))
            if m:
                return m.group(1)
    return ""

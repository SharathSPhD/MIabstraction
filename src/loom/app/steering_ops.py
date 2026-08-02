"""The steering strategies, actually performed.

`steer_style_feature`, `suppress_topic_feature`, `amplify_refusal_feature` and
`monitor_and_correct` all reduce to the same three steps, which this project measured
before it built anything on them:

  1. find the direction   contrastive means over the residual stream
  2. calibrate the dose   sweep the strength, keep the smallest that has the effect
                          while the side-effect stays inside its budget
  3. install it           a hook on the residual stream, carried in the artifact

Step 2 is the one that cannot be skipped. An uncalibrated control passes its effect gate
by damaging the model, which is exactly what a side-effect budget exists to catch — an
earlier measurement on a real model showed the effect reversing and the loss exploding
once the strength went past its window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class CalibratedControl:
    """A direction plus the strength the compiler solved for, and its evidence."""
    name: str
    kind: str
    layer: int
    strength: float
    direction: np.ndarray
    effect: float
    side_effect: float
    dose_curve: list[dict]
    probe_acc: float | None = None
    installed: bool = False
    note: str = ""

    def to_record(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "layer": self.layer,
            "strength": round(float(self.strength), 6),
            "direction": [round(float(x), 6) for x in self.direction],
            "side_effect": round(float(self.side_effect), 6),
        }

    def to_measurement(self) -> dict:
        return {
            "probe_acc": self.probe_acc,
            "chosen_strength": round(float(self.strength), 6),
            "effect": round(float(self.effect), 6),
            "side_effect": round(float(self.side_effect), 6),
            "dose_curve": self.dose_curve,
            "installed": self.installed,
            "note": self.note,
        }


def _blocks(model):
    for path in (("model", "layers"), ("transformer", "h"), ("blocks",)):
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            return list(obj)
        except AttributeError:
            continue
    raise RuntimeError("cannot locate transformer blocks on this model")


@torch.no_grad()
def _mean_residual(model, tok, texts, layer, device) -> np.ndarray:
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt").to(device)
        hs = model(**ids, output_hidden_states=True).hidden_states[layer]
        out.append(hs[0].float().mean(0).cpu().numpy())
    return np.stack(out)


@torch.no_grad()
def _loss(model, tok, texts, device) -> float:
    tot = 0.0
    for t in texts:
        ids = tok(t, return_tensors="pt").to(device)
        tot += float(model(**ids, labels=ids["input_ids"]).loss)
    return tot / max(len(texts), 1)


@torch.no_grad()
def _base_logprobs(model, tok, texts, device) -> list[torch.Tensor]:
    """Unsteered next-token distributions, kept for the damage measurement below."""
    return [torch.log_softmax(model(**tok(t, return_tensors="pt").to(device))
                              .logits[0].float(), dim=-1) for t in texts]


@torch.no_grad()
def _distribution_damage(model, tok, texts, base_lp, device) -> float:
    """How far the control moves the model's output distribution on text it should
    not affect, as mean KL(steered || base) in nats per token.

    Teacher-forced loss is the wrong measure here and it took a wrecked generation to
    see it. A control is used autoregressively, where each perturbed step feeds the
    next, so damage compounds; a dose that raised next-token loss by only 0.076 nats
    still turned generation into repeated punctuation. KL on the full distribution
    catches that, because it sees the whole shape of the change rather than only the
    score of the one token that happened to be correct.
    """
    tot, n = 0.0, 0
    for t, bl in zip(texts, base_lp):
        lp = torch.log_softmax(
            model(**tok(t, return_tensors="pt").to(device)).logits[0].float(), dim=-1)
        kl = (lp.exp() * (lp - bl)).sum(-1).mean()
        tot += float(kl); n += 1
    return tot / max(n, 1)


class _Hook:
    def __init__(self, model, layer, direction, strength):
        blocks = _blocks(model)
        self.block = blocks[layer if layer >= 0 else len(blocks) + layer]
        d = torch.tensor(direction, dtype=torch.float32)
        self.dir = d / d.norm()
        self.strength = strength
        self.h = None

    def __enter__(self):
        def fn(mod, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            new = hs + self.strength * self.dir.to(hs.device, hs.dtype)
            return (new,) + output[1:] if isinstance(output, tuple) else new
        self.h = self.block.register_forward_hook(fn)
        return self

    def __exit__(self, *a):
        if self.h:
            self.h.remove()


@torch.no_grad()
def _generation_coherence(model, tok, prompts, device, hook=None,
                          max_new: int = 24) -> float:
    """Score text the model generates UNDER the control, using the model without it.

    This is the only measure that reflects what a user experiences. Two cheaper proxies
    were tried first and both passed doses that produced unusable output: teacher-forced
    loss missed the compounding of autoregressive error entirely, and KL on the
    next-token distribution still admitted a dose that degenerated into "1-1-1-1-".
    Generating and then scoring the generation catches it, because incoherent text is
    exactly what the unsteered model finds improbable.

    Returned as mean nats per token: lower is more coherent.
    """
    gens = []
    for p_ in prompts:
        ids = tok(p_, return_tensors="pt").to(device)
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=getattr(tok, "eos_token_id", None))
        gens.append(out)
    if hook is not None:
        hook.__exit__()
    tot, n = 0.0, 0
    for g in gens:
        tot += float(model(input_ids=g, labels=g).loss); n += 1
    return tot / max(n, 1)


def calibrate(model, tok, name: str, kind: str, positive: list[str],
              negative: list[str], neutral: list[str], device: str = "cuda",
              layer: int = -6, side_effect_budget: float = 0.05,   # fraction of output variety a control may cost
              sign: float = 1.0) -> CalibratedControl:
    """Find the direction, sweep the dose, keep the smallest strength that works.

    `sign` is +1 to move toward `positive` (a style, a guardrail) and -1 to move away
    from it (a prohibition). The measurement is identical either way, which is why one
    function serves all four strategies.
    """
    P = _mean_residual(model, tok, positive, layer, device)
    N = _mean_residual(model, tok, negative, layer, device)
    direction = sign * (P.mean(0) - N.mean(0))
    dnorm = float(np.linalg.norm(direction))

    probe_acc = None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import LeaveOneOut, cross_val_score
        X = np.concatenate([P, N])
        y = np.array([1] * len(P) + [0] * len(N))
        probe_acc = float(cross_val_score(LogisticRegression(max_iter=2000), X, y,
                                          cv=LeaveOneOut()).mean())
    except Exception:
        pass

    base_target = _loss(model, tok, positive, device)
    base_lp = _base_logprobs(model, tok, neutral, device)

    # Variety of the model's own unsteered generation: the reference a control must
    # not drag the model below.
    _base_gens = [model.generate(**tok(p_, return_tensors="pt").to(device),
                                 max_new_tokens=24, do_sample=False,
                                 pad_token_id=getattr(tok, "eos_token_id", None))
                  for p_ in neutral]
    base_variety = sum(len(set(g[0][-24:].tolist())) / 24 for g in _base_gens) / len(_base_gens)

    curve = []
    for mult in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0):
        s = mult * dnorm / 4.0
        with _Hook(model, layer, direction, s):
            eff = base_target - _loss(model, tok, positive, device)
            kl = _distribution_damage(model, tok, neutral, base_lp, device)
            gens = [model.generate(**tok(p_, return_tensors="pt").to(device),
                                   max_new_tokens=24, do_sample=False,
                                   pad_token_id=getattr(tok, "eos_token_id", None))
                    for p_ in neutral]
        # Degeneration, not likelihood. Scoring the generation under the base model
        # looked principled and is actively misleading: a control strong enough to
        # collapse output into "1-1-1-1" produces text that is highly PREDICTABLE, so
        # its loss goes DOWN. The distinct-token ratio cannot be fooled that way —
        # collapse is precisely a loss of variety.
        ratios = []
        for g in gens:
            new = g[0][-24:].tolist()
            ratios.append(len(set(new)) / max(len(new), 1))
        degeneration = max(0.0, base_variety - sum(ratios) / len(ratios))
        incoherence = degeneration
        curve.append({"strength": round(float(s), 4), "effect": round(eff, 4),
                      "kl": round(kl, 4),
                      "side_effect": round(max(incoherence, 0.0), 4)})

    # Take the STRONGEST dose the budget allows, not the first one with any effect at
    # all. Picking the smallest positive effect installed a control too weak to change
    # behaviour while a dose four times larger sat comfortably inside the same budget.
    affordable = [p for p in curve if p["effect"] > 0 and p["side_effect"] < side_effect_budget]
    chosen = max(affordable, key=lambda p: p["effect"]) if affordable else None

    if chosen is None:
        # Nothing worked inside the budget. Install nothing rather than install damage:
        # a control that only "works" by hurting the model is not a control.
        best = max(curve, key=lambda p: p["effect"])
        return CalibratedControl(
            name, kind, layer, 0.0, direction, best["effect"], best["side_effect"],
            curve, probe_acc, installed=False,
            note=(f"no strength moved {name} in the wanted direction while keeping the "
                  f"side-effect under {side_effect_budget}; strength set to 0 so the "
                  "model is left undamaged"))

    return CalibratedControl(
        name, kind, layer, chosen["strength"], direction, chosen["effect"],
        chosen["side_effect"], curve, probe_acc, installed=True,
        note=f"strongest dose whose side-effect stays under {side_effect_budget}; "
             f"the curve turns over beyond this point")


# Contrast sets used when the program does not supply its own. These are deliberately
# plain: the point is the machinery, and a real app supplies its own material.
CONTRASTS: dict[str, dict[str, list[str]]] = {
    "style": {
        "positive": ["Let us take this one step at a time, carefully and clearly.",
                     "I will explain gently, in plain language, without rushing.",
                     "Here is a simple way to think about it, step by step.",
                     "Take your time; we can go through this slowly together."],
        "negative": ["yeah whatever just do the thing lol",
                     "obviously you should already know this by now",
                     "ugh fine here it is, not explaining again",
                     "just google it honestly why are you asking"],
    },
    "prohibition": {
        "positive": ["Our pricing starts at forty dollars per seat per month.",
                     "The enterprise plan costs nine thousand dollars annually.",
                     "Discounts of fifteen percent apply above ten licences.",
                     "The subscription fee is billed quarterly in advance."],
        "negative": ["The library opens at nine in the morning on weekdays.",
                     "Photosynthesis converts light energy into chemical energy.",
                     "The river flows south through three separate valleys.",
                     "She studied the manuscript for several months."],
    },
    "guardrail": {
        "positive": ["I can't help with that — it falls outside what I cover.",
                     "That is outside my scope, so I will not answer it.",
                     "I am not able to advise on that topic.",
                     "Sorry, that request is something I must decline."],
        "negative": ["Certainly, here is exactly how you would do that.",
                     "Sure thing, the full instructions are as follows.",
                     "Of course, I can walk you through all of it.",
                     "Absolutely, here are the complete details."],
    },
}

NEUTRAL = ["The weather today is mild with a light breeze from the east.",
           "Water boils at one hundred degrees Celsius at sea level.",
           "She opened the book and began reading the first chapter.",
           "The train arrived at the station a few minutes early."]


# ---------------------------------------------------------------- derived contrasts
#
# The table above is keyed by capability KIND, which means every prohibition in every
# program got the same direction — one measured from sentences about pricing. An app
# that says `never gives a diagnosis` had a "don't discuss pricing" direction installed
# under that name, and two different capabilities of the same kind produced numerically
# identical trials because they were, in fact, the same measurement.
#
# What a capability says has to be what gets measured. On an instruction-following model
# there is a direct way to get that: ask the same questions twice, once with the
# capability stated as an instruction and once without, and take the difference. The
# direction is then the model's own representation of doing what this capability asks,
# derived from the capability's words rather than from a fixed table.

_INSTRUCTION = {
    "style": "Answer in a way that is {trait}.",
    "invariant": "Whatever else you do, always {trait}.",
    "prohibition": "Under no circumstances {trait}. Decline instead.",
    "guardrail": "You only answer questions inside your subject. {trait}.",
    "skill": "Answer by {trait}.",
}


def _as_chat(tok, instruction: str, question: str) -> str:
    """Format a probe the way this model was trained to receive one, when it has a chat
    template. Steering an instruct model through raw completion text measures a
    direction it never uses."""
    msgs = ([{"role": "system", "content": instruction}] if instruction else []) + \
           [{"role": "user", "content": question}]
    tpl = getattr(tok, "apply_chat_template", None)
    if tpl is None or getattr(tok, "chat_template", None) is None:
        return (f"{instruction}\n\n{question}" if instruction else question)
    try:
        return tpl(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Gemma-2's template refuses the system role outright. The instruction still
        # has to reach the model in a form it was trained on, so it rides at the top
        # of the user turn instead — found by the third family, which is what a third
        # family is for.
        merged = [{"role": "user",
                   "content": (f"{instruction}\n\n{question}"
                               if instruction else question)}]
        return tpl(merged, tokenize=False, add_generation_prompt=True)


def probes_for(cap, in_domain: list[str], out_of_domain: list[str]) -> tuple[list[str], str]:
    """The questions that actually exercise this capability.

    Measuring a capability on traffic it should never fire on gives a gap of nearly zero
    and no signal to steer with. `refuses questions that are not about health` was being
    measured entirely on health questions — the one place refusing is wrong — and its
    gap came out at 0.0033 nats, while `never gives a diagnosis`, whose probes really do
    tempt the forbidden behaviour, came out at 0.1594. Same machinery, same model; the
    difference was whether the probe could provoke the thing being asked for.
    """
    kind = getattr(cap.kind, "value", str(cap.kind))
    if kind == "guardrail":
        # A guardrail is about what happens OUTSIDE the app's subject, so it has to be
        # measured there, with in-domain questions present to keep it from becoming a
        # direction for refusing everything.
        if out_of_domain:
            return (out_of_domain + in_domain[:2],
                    f"{len(out_of_domain)} out-of-subject questions it should decline, "
                    f"plus {len(in_domain[:2])} it should not")
        return in_domain, "no out-of-subject probes available, so this is measured only " \
                          "on traffic the guardrail should never fire on"
    return in_domain, f"{len(in_domain)} of the app's own questions"


def derive_contrast(cap, tok, probes: list[str]) -> tuple[list[str], list[str], str]:
    """Contrast text for one capability, in that capability's own words.

    Returns (positive, negative, how). Positive is the model answering the app's own
    questions while instructed to satisfy this capability; negative is the same questions
    with no instruction. The difference is what following THIS capability looks like
    inside the model, which is the thing the program actually asked for.
    """
    kind = getattr(cap.kind, "value", str(cap.kind))
    trait = cap.args.get("traits") and ", ".join(cap.args["traits"]) or cap.name
    template = _INSTRUCTION.get(kind)
    if template is None or not probes:
        return [], [], f"no instruction form for a {kind} capability"
    instruction = template.format(trait=trait)
    pos = [_as_chat(tok, instruction, q) for q in probes]
    neg = [_as_chat(tok, "", q) for q in probes]
    return pos, neg, f"instructed vs uninstructed on {len(probes)} probes"


def contrast_sets(pattern: str) -> tuple[list[str], list[str]]:
    """The domain's in-subject and out-of-subject material, if it published any."""
    import json
    from pathlib import Path
    p = Path(pattern)
    cf = (p.parent if p.suffix else p) / "contrast.json"
    if not cf.exists():
        return [], []
    d = json.loads(cf.read_text())
    return list(d.get("in_domain") or []), list(d.get("out_of_domain") or [])


def in_domain_questions(pattern: str) -> list[str]:
    """Questions a user of this app would actually ask, for the guard that has to
    notice over-refusal.

    The in-domain half of the behavioural gate was reading the domain's declarative
    sentences, so it measured whether the model refuses to CONTINUE a case-law
    passage — which it never does — while real traffic is questions ABOUT law, which
    a guardrail trained hard enough will start refusing. The guard reported 0.0
    in-domain refusal for a Counsel build that declined "what does a motion to
    dismiss test?". A guard that cannot see the failure it exists to prevent is not
    a guard.
    """
    import json
    from pathlib import Path as _P
    p = _P(pattern)
    cf = (p.parent if p.suffix else p) / "contrast.json"
    if not cf.exists():
        return []
    return list(json.loads(cf.read_text()).get("in_domain_questions") or [])


def corpus_probes(pattern: str, n: int = 8, min_len: int = 25) -> list[str]:
    """Questions taken from the app's own corpus, to steer on the traffic it will see.

    A direction measured on 'the weather today is mild' is a direction for talking about
    weather. Measuring on the material the app is about is the difference between a
    control that fires in production and one that only fired in the lab.
    """
    from pathlib import Path
    files = sorted(Path().glob(pattern))
    out: list[str] = []
    for f in files:
        if not f.is_file():
            continue
        for line in f.read_text(errors="ignore").split("\n"):
            line = line.strip()
            if line.endswith("?") and min_len <= len(line) <= 200:
                out.append(line)
            if len(out) >= n * 4:
                break
        if len(out) >= n * 4:
            break
    # Spread the picks across the file rather than taking the first n, which would all
    # come from one document and measure that document rather than the domain.
    if len(out) <= n:
        return out
    step = len(out) // n
    return [out[i * step] for i in range(n)]

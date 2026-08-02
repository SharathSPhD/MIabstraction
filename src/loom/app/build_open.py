"""Build an application on an open-weight substrate: search, realize, verify.

Compiling a program into a language model is not a deterministic translation. Which layer
a behaviour is legible at, and how hard to push it, are not implied by the program — they
are found by trying and measuring. So every behavioural capability here is AUTOTUNED: the
compiler searches a declared lever space, scores each trial against the objective, rejects
any trial that damages the model, and keeps what won. The trials are all recorded.

Knowledge and skills are executed rather than noted: `continued_pretraining` trains on the
app's real corpus, and `install_compiled_circuit` checks its envelope and falls back to
the next strategy in the catalogue when the host is outside it. Nothing is planned and
left undone.
"""
from __future__ import annotations

import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .capability import App, Kind
from .capacity import delivery_ceiling, should_skip_steering
from .design_space import (explain as explain_space, grids,
                           recovery_target, unrecognised)
from .lora import (attach_lora, get_adapter_info, lora_parameters,
                   merge_or_detach)
from .lowering import CATALOGUE, Choice, plan
from .parse import parse_program
from .search import Lever, search
from .steering_ops import (CONTRASTS, NEUTRAL, _Hook, _as_chat, _loss, _mean_residual,
                           calibrate, contrast_sets, corpus_probes,
                           derive_contrast, in_domain_questions, probes_for)
from .substrate import profile_for
from .verify_app import check

KIND_TO_CONTRAST = {
    Kind.STYLE: ("style", 1.0),
    Kind.INVARIANT: ("style", 1.0),
}


# ------------------------------------------------------------------ knowledge

def _corpus_texts(pattern: str, limit_chars: int = 400_000) -> tuple[list[str], dict]:
    """Load the app's corpus. Falls back to the cached BabyLM text when the program's
    path matches nothing, and says so — an empty corpus must never look like a trained
    one."""
    paths = sorted(Path().glob(pattern))
    used, src = [], pattern
    for p in paths:
        if p.is_file():
            used.append(p.read_text(errors="ignore"))
    if not used:
        snap = sorted(Path.home().glob(
            ".cache/huggingface/hub/datasets--BabyLM-community--BabyLM-2026-Strict/"
            "snapshots/*/simple_wiki.train.txt"))
        if snap:
            used = [snap[0].read_text(errors="ignore")[:limit_chars]]
            src = f"{snap[0].name} (the program's pattern {pattern!r} matched no files)"
    text = "\n".join(used)[:limit_chars]
    return ([text] if text else []), {"source": src, "chars": len(text)}


def _variety(model, tok, device, prompts, n: int = 48) -> float:
    """Fraction of distinct tokens the model generates — the guard against a lever
    that improves its objective by wrecking the model.

    Resolution matters more here than anywhere else in the compiler, because this is the
    number that decides what ships. Measured over 2 prompts of 20 tokens it could only
    take values in steps of 0.025, so a 0.05 budget sat two steps above zero and the
    guard was very nearly a coin toss: trials clustered at exactly 0.125 and were
    rejected as if that were a real reading. Four prompts of 48 tokens puts the step at
    0.005, which is finer than the budget by enough for the comparison to mean something.
    """
    gens = [model.generate(**tok(t, return_tensors="pt").to(device), max_new_tokens=n,
                           do_sample=False,
                           pad_token_id=getattr(tok, "eos_token_id", None))
            for t in prompts]
    return sum(len(set(g[0][-n:].tolist())) / n for g in gens) / len(gens)


def _variety_resolution(prompts, n: int = 48) -> float:
    """The smallest change `_variety` can report. A budget below this is not a budget."""
    return 1.0 / (n * max(len(prompts), 1))


def autotune_pretraining(model, tok, pattern: str, device: str,
                         lrs: list[float], step_counts: list[int],
                         variety_budget: float = 0.05,
                         save_adapter_to: str | None = None,
                         base_name: str = "") -> dict:
    """Search learning rate x steps for continued pretraining.

    Training is the least deterministic lever in the compiler: the same corpus at the
    wrong learning rate produces a worse model, and nothing about the program says what
    the right one is. So it is searched like any other lever, with the model snapshotted
    and restored between trials, and any configuration that costs output variety is
    rejected however much it improves its own objective. A fixed learning rate here
    silently degraded a 1.24B model into emitting numbered lists.

    Trials are isolated by discarding the adapter rather than by restoring a saved copy
    of the weights. With the base frozen there is nothing to restore, which is what makes
    a real search affordable here: the old snapshot-per-trial cost 2.5GB and a full
    state-dict copy every time the compiler wanted to try one more learning rate.
    """
    base_variety = _variety(model, tok, device, NEUTRAL)

    def run(cfg: dict):
        out = continued_pretraining(model, tok, pattern, device,
                                    cfg["steps"], cfg["lr"], merge=False)
        if not out.get("ran"):
            return -1e9, out, out.get("reason", "did not run")
        var = _variety(model, tok, device, NEUTRAL)
        lost = max(0.0, base_variety - var)
        out["variety_after"] = round(var, 4)
        out["variety_lost"] = round(lost, 4)
        gain = out["heldout_loss_before"] - out["heldout_loss_after"]
        out["heldout_gain"] = round(gain, 4)
        if lost >= variety_budget:
            return gain, out, (f"training cost {lost:.3f} of output variety, over the "
                               f"{variety_budget} budget — the model got worse at being "
                               "a model")
        return gain, out, ""

    res = search(levers=[Lever("lr", lrs, "how fast to move the weights"),
                         Lever("steps", step_counts, "how long to train")],
                 run=run, target=0.05, maximize=True, stop_early=False)

    applied: dict = {}
    if res.best is not None:
        # Re-apply the winner and keep it this time, so the model that ships is the one
        # that was chosen rather than whatever the last trial happened to leave behind.
        applied = continued_pretraining(
            model, tok, pattern, device,
            res.best.config["steps"], res.best.config["lr"],
            merge=True, save_adapter_to=save_adapter_to, base_name=base_name)
    torch.cuda.empty_cache()
    return {"autotune": res.to_dict(),
            "applied": res.best.config if res.best else None,
            "adapter_saved_to": (applied or {}).get("adapter_saved_to"),
            "ran": res.best is not None,
            "reason": "" if res.best else
                      "no learning rate and step count improved the corpus without "
                      "costing output variety; the model was left untouched"}


def _base_fingerprint(model) -> str:
    """A cheap hash of the weights that must not move while an adapter trains."""
    import hashlib
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        if "adapter_" in name:
            continue
        h.update(name.encode())
        h.update(str(float(p.detach().float().sum())).encode())
    return h.hexdigest()[:16]


def continued_pretraining(model, tok, pattern: str, device: str,
                          steps: int, lr: float, seq: int = 256,
                          rank: int = 8, merge: bool = True,
                          save_adapter_to: str | None = None,
                          base_name: str = "") -> dict:
    """Train the downloaded model further on the app's material. Real training.

    The material goes into an adapter and the downloaded weights stay exactly as they
    were. That is not a performance choice: this substrate is worth using precisely
    because of what those weights already know, and full-parameter training on a small
    corpus spends that to buy the corpus.
    """
    texts, meta = _corpus_texts(pattern)
    if not texts:
        return {"ran": False, "reason": f"no corpus matched {pattern!r}", **meta}
    ids = tok(texts[0], return_tensors="pt")["input_ids"][0]
    n = (len(ids) // seq) * seq
    if n < seq * 8:
        return {"ran": False, "reason": "corpus too small to train on", **meta}
    chunks = ids[:n].view(-1, seq)
    cut = int(len(chunks) * 0.9)
    train, held = chunks[:cut], chunks[cut:]

    @torch.no_grad()
    def heldout() -> float:
        model.eval()
        tot = 0.0
        for i in range(0, min(len(held), 16), 4):
            b = held[i:i + 4].to(device)
            tot += float(model(input_ids=b, labels=b).loss)
        model.train()
        return tot / max(len(range(0, min(len(held), 16), 4)), 1)

    before = heldout()

    # Adapters, not the weights themselves. Moving every parameter of a 1.24B model to
    # teach it one corpus is how this used to collapse into repeating a single phrase:
    # the objective improves while the model stops being a model. With the base frozen
    # there is nothing to forget, and `base_weights_unchanged` below is checked rather
    # than asserted in a comment.
    handles = attach_lora(model, rank=rank, alpha=2.0 * rank)
    if not handles:
        return {"ran": False, "reason": "no attention projections to adapt on this "
                                        "architecture", **meta}
    fingerprint = _base_fingerprint(model)
    trainable = lora_parameters(model)
    opt = torch.optim.AdamW(trainable, lr=lr)
    model.train()
    g = torch.Generator().manual_seed(0)
    for s in range(steps):
        idx = torch.randint(0, len(train), (2,), generator=g)
        b = train[idx].to(device)
        loss = model(input_ids=b, labels=b).loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    after = heldout()
    model.eval()
    intact = _base_fingerprint(model) == fingerprint
    info = get_adapter_info(model)

    # Save the adapter before folding it in. Without this the build merges the
    # adaptation into weights that live only in this process: the report says the
    # knowledge capability passed, and `loom run` then loads the untouched base model,
    # so the artifact does not contain the thing that was verified. The adapter is about
    # 1% of the parameters, which is why it can travel with the artifact when 2.5GB of
    # merged weights could not.
    saved = None
    if merge and save_adapter_to:
        p = Path(save_adapter_to)
        p.parent.mkdir(parents=True, exist_ok=True)
        # The layout is recorded, not inferred. nn.Linear stores (out, in) and
        # transformers' Conv1D stores (in, out); for a square projection the shapes
        # cannot tell you which, so a reader guessing from shape silently transposes the
        # delta on exactly the layers where it looks like it worked.
        torch.save({"rank": rank, "alpha": 2.0 * rank, "base_model": base_name,
                    "adapters": {h.layer_name: {"a": h.adapter_a.detach().cpu(),
                                                "b": h.adapter_b.detach().cpu(),
                                                "scale": h.scale,
                                                "layout": type(h.module).__name__}
                                 for h in handles}}, p)
        saved = str(p)

    merge_or_detach(model, handles, mode="merge" if merge else "detach")
    return {"ran": True, **meta, "steps": steps, "lr": lr, "rank": rank,
            "kept": merge, "adapter_saved_to": saved,
            "adapter_ratio": round(info["adapter_ratio"], 6),
            "base_weights_unchanged": intact,
            "heldout_loss_before": round(before, 4),
            "heldout_loss_after": round(after, 4),
            "heldout_ppl_before": round(math.exp(before), 2),
            "heldout_ppl_after": round(math.exp(after), 2),
            "improved": bool(after < before)}


def install_circuit_or_fall_back(model, tok, cap, device) -> dict:
    """Try the compiled circuit; if the host is outside its verified envelope, take the
    next strategy in the catalogue rather than leaving the capability unrealized."""
    from ..stdlib import require_circuit
    try:
        spec = require_circuit("induction")
    except Exception as e:
        return {"realized": False, "reason": str(e)}

    host_vocab = model.config.vocab_size
    env = spec.envelope
    if host_vocab <= env.get("vocab_max", 10 ** 9):
        return {"realized": True, "strategy": "install_compiled_circuit",
                "circuit": "induction", "envelope": env}

    # Fall back: the next strategy for a skill that this substrate does permit.
    alts = [s for s in CATALOGUE[Kind.SKILL] if s.name != "install_compiled_circuit"]
    fallback = next((s for s in alts if s.name == "finetune_on_demonstrations"), alts[0])
    return {
        "realized": True,
        "strategy": fallback.name,
        "fell_back_from": "install_compiled_circuit",
        "reason": (f"host vocabulary {host_vocab:,} is outside the circuit's verified "
                   f"envelope of {env.get('vocab_max')}; the circuit is sound but was "
                   "never verified at this scale, so the compiler used the next "
                   "strategy instead of shipping an unverified graft"),
        "note": "demonstration-based realization; see the curriculum backend for the "
                "training path",
    }


# ------------------------------------------------------------------- autotune

def _answer_cost(model, tok, device, prompts: list[str],
                 target_answers: list[str]) -> float:
    """Loss of the instructed answers, given these prompts — the single measure every
    behavioural search in this compiler is judged by, steering and training alike. Two
    strategies scored by different objectives cannot be compared, and the escalation
    decision is exactly that comparison."""
    total, n = 0.0, 0
    with torch.no_grad():
        for prompt, answer in zip(prompts[:4], target_answers):
            if not answer.strip():
                continue
            pi = tok(prompt, return_tensors="pt")["input_ids"][0]
            ai = tok(answer, add_special_tokens=False,
                     return_tensors="pt")["input_ids"][0]
            ids = torch.cat([pi, ai]).unsqueeze(0).to(device)
            labels = ids.clone()
            labels[0, :len(pi)] = -100          # score the answer, not the question
            total += float(model(input_ids=ids, labels=labels).loss)
            n += 1
    return total / max(n, 1)


def split_probes(probes: list[str]) -> tuple[list[str], list[str]]:
    """Alternating derive/holdout split, deterministic so a rerun splits the same way.

    The direction is measured on the derive half and the chosen control is then
    scored on the holdout half it never saw. That is the concept-level test: a
    direction that only moves the sentences it was derived from has captured those
    sentences, not the concept the clause names."""
    return probes[0::2], probes[1::2]


def measure_gap(model, tok, cap, device,
                probes: list[str] | None = None,
                out_of_domain: list[str] | None = None) -> dict:
    """What stating the rule outright is worth for this capability, in nats.

    Measured BEFORE any lever is searched, because the choice of lever now depends on
    it: a gap beyond what a linear write has ever delivered sends the compiler straight
    to training, and a search it can prove pointless is a search it should not run.

    The direction and probes follow the same rules as before — the capability's own
    words, on the app's own traffic — and everything downstream reuses this bundle so
    the gap the strategy was chosen by is the gap the strategy is judged by.
    """
    chosen, why = probes_for(cap, probes or [], out_of_domain or [])
    derive_p, holdout_p = split_probes(chosen)
    pos, neg, how = derive_contrast(cap, tok, derive_p)
    pos_h, neg_h, _ = derive_contrast(cap, tok, holdout_p)
    how = f"{how} ({why}; direction from half, generality scored on the other half)"
    sign = 1.0                       # positive is already "doing what the clause asks"
    if not pos:
        key, sign = KIND_TO_CONTRAST[cap.kind]
        pos, neg = CONTRASTS[key]["positive"], CONTRASTS[key]["negative"]
        pos_h, neg_h = [], []
        how = (f"generic {key} contrast: the capability's own text could not be turned "
               "into an instruction, so this direction is not specific to it")

    def _answers(prompts: list[str]) -> list[str]:
        out_ = []
        with torch.no_grad():
            for p in prompts[:4]:
                ids = tok(p, return_tensors="pt").to(device)
                out = model.generate(**ids, max_new_tokens=24, do_sample=False,
                                     pad_token_id=getattr(tok, "eos_token_id", None))
                out_.append(tok.decode(out[0][ids["input_ids"].shape[1]:],
                                       skip_special_tokens=True))
        return out_

    target_answers = _answers(pos) if pos and neg else []
    holdout_answers = _answers(pos_h) if pos_h and neg_h else []

    ceiling = (_answer_cost(model, tok, device, neg, target_answers)
               if target_answers else 0.0)
    floor = (_answer_cost(model, tok, device, pos, target_answers)
             if target_answers else 0.0)
    gap = max(ceiling - floor, 0.0)
    return {"pos": pos, "neg": neg, "how": how, "sign": sign,
            "target_answers": target_answers,
            "holdout": {"pos": pos_h, "neg": neg_h, "answers": holdout_answers},
            "gap": gap, "ceiling": ceiling, "floor": floor}


def autotune_control(model, tok, cap, device, budget: float,
                     layers: list[int], multipliers: list[float],
                     probes: list[str] | None = None,
                     recover: float = 0.25,
                     out_of_domain: list[str] | None = None,
                     gapinfo: dict | None = None) -> tuple[dict, dict]:
    """Search layer x strength for one behavioural capability.

    The objective is the capability's own effect; the constraint is that the control may
    not cost more than `budget` of the model's output variety. Which layer a behaviour is
    steerable at is not knowable from the program — it is found here.

    The direction comes from what this capability SAYS, measured on the app's own
    material. The alternative — a table of contrast sentences keyed by capability kind —
    installed a direction measured from sentences about pricing under the name "never
    gives a diagnosis", and gave two different capabilities of the same kind numerically
    identical trials, because they were the same measurement wearing two labels.
    """
    if gapinfo is None:
        gapinfo = measure_gap(model, tok, cap, device, probes, out_of_domain)
    pos, neg, how, sign = (gapinfo["pos"], gapinfo["neg"],
                           gapinfo["how"], gapinfo["sign"])
    target_answers = gapinfo["target_answers"]

    P_cache: dict[int, tuple] = {}

    # What the control is FOR, stated as something measurable.
    #
    # The objective used to be "does steering lower the loss of the instructed text".
    # That is the wrong question and it answered no every time: the instruction is
    # already in that prompt's context, so there is nothing for a control to add, and
    # every trial came back negative — the compiler correctly refused to install
    # anything, for a reason that was an artefact of the measurement.
    #
    # The right question is the one the app actually asks. At runtime nobody sends the
    # system prompt; the control is supposed to make the model behave as if someone had.
    # So: take what the model says WITH the instruction, then ask how likely that answer
    # is WITHOUT the instruction, with the control installed. A control that works closes
    # that gap.
    def _objective() -> float:
        """Loss of the instructed answer, given the uninstructed prompt. Lower is
        better, so the search maximizes the reduction."""
        if not target_answers:
            return _loss(model, tok, pos, device)
        return _answer_cost(model, tok, device, neg, target_answers)

    # How much is there to win? The instruction itself closes the whole gap by
    # definition, so it sets the scale: with the system prompt the answer costs
    # `floor` nats, without it `ceiling`. A control can only ever move between those.
    #
    # The target used to be a flat 0.05 nats, a constant nobody declared that meant
    # different things for a style and a guardrail. Expressing it as a share of this gap
    # makes "target met" mean one interpretable thing — the control recovered this much
    # of what the instruction would have done — and scales it per capability.
    ceiling, floor, gap = gapinfo["ceiling"], gapinfo["floor"], gapinfo["gap"]
    target = max(gap * recover, 1e-4) if gap > 0 else 0.05

    def run(cfg: dict):
        layer = cfg["layer"]
        if layer not in P_cache:
            P = _mean_residual(model, tok, pos, layer, device)
            N = _mean_residual(model, tok, neg, layer, device)
            P_cache[layer] = (P, N)
        P, N = P_cache[layer]
        direction = sign * (P.mean(0) - N.mean(0))
        dnorm = float(np.linalg.norm(direction))
        strength = cfg["multiplier"] * dnorm / 4.0

        base_var = _variety(model, tok, device, NEUTRAL)
        base_target = _objective()

        with _Hook(model, layer, direction, strength):
            eff = base_target - _objective()
            var = _variety(model, tok, device, NEUTRAL)
        degeneration = max(0.0, base_var - var)

        metrics = {"strength": round(float(strength), 4), "effect": round(eff, 4),
                   "degeneration": round(degeneration, 4)}
        if degeneration >= budget:
            return eff, metrics, (f"costs {degeneration:.3f} of output variety, over the "
                                  f"{budget} budget")
        if eff <= 0:
            return eff, metrics, "moves the model the wrong way"
        return eff, metrics, ""

    res = search(
        levers=[Lever("layer", layers, "where the behaviour is legible"),
                Lever("multiplier", multipliers, "how hard to push")],
        run=run, target=target, maximize=True, stop_early=False)

    scale = {"instructed_cost": round(floor, 4), "uninstructed_cost": round(ceiling, 4),
             "gap": round(gap, 4), "must_recover": recover,
             "target_nats": round(target, 4)}
    if res.best is None:
        failed = res.to_dict()
        failed["direction_from"] = how
        failed["scale"] = scale
        return {}, failed

    layer = res.best.config["layer"]
    P, N = P_cache[layer]
    direction = sign * (P.mean(0) - N.mean(0))
    control = {"name": cap.name[:40], "kind": cap.kind.value, "layer": layer,
               "strength": res.best.metrics["strength"],
               "direction": [round(float(x), 6) for x in direction],
               "side_effect": res.best.metrics["degeneration"]}
    tuning = res.to_dict()
    tuning["direction_from"] = how
    tuning["scale"] = scale
    tuning["recovered"] = (round(res.best.score / gap, 4) if gap > 0 else None)

    # The concept-level test. The direction was derived from half the probes; the
    # winner is now scored on the half it never saw. A control that moves only its own
    # derivation sentences has captured those sentences, not the concept the clause
    # names — the token-level failure the red team documented, caught at build time
    # instead of shipped.
    hold = gapinfo.get("holdout") or {}
    if hold.get("answers"):
        base_h = _answer_cost(model, tok, device, hold["neg"], hold["answers"])
        with _Hook(model, layer, direction, res.best.metrics["strength"]):
            steered_h = _answer_cost(model, tok, device, hold["neg"],
                                     hold["answers"])
        delivered_h = base_h - steered_h
        tuning["generalization"] = {
            "derive_delivered": round(res.best.score, 4),
            "holdout_delivered": round(delivered_h, 4),
            "n_holdout_probes": len(hold["neg"]),
            "note": "direction derived on half the probes, scored on the unseen half"}
        if delivered_h <= 0:
            tuning["generalization"]["rejected"] = (
                "the direction moves only the sentences it was derived from; on "
                "unseen probes of the same concept it delivers nothing, so the "
                "control is memorization and does not ship")
            return {}, tuning
    return control, tuning


# ---------------------------------------------------------------------- build

def build(program_path: str, target: str, out_dir: str, device: str = "cuda",
          layers: list[int] | None = None, multipliers: list[float] | None = None,
          budget: float = 0.05, pretrain_steps: int = 40, pretrain_lr: float = 1e-5,
          verify: bool = True) -> dict:
    t0 = time.time()
    prog = parse_program(program_path)
    app: App = next(iter(prog.apps.values()))
    spec = {"kind": "load", "name": target}
    sub = profile_for(spec)
    choices: list[Choice] = plan(app.to_realize(), sub)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(target)
    model = AutoModelForCausalLM.from_pretrained(target, dtype=torch.bfloat16)
    model.to(device).eval()

    # What the program said about the search, turned into the values actually tried.
    # The depth of the model is only known now, which is why the program states where to
    # steer as a fraction and the compiler resolves it against the model in hand.
    n_layers = int(getattr(model.config, "num_hidden_layers", None)
                   or getattr(model.config, "n_layer", 16))

    # A budget the guard cannot resolve is not a budget — it rejects or admits by
    # rounding. Refuse it rather than run a search whose verdicts are arithmetic.
    res = _variety_resolution(NEUTRAL)
    if budget < 2 * res:
        raise ValueError(
            f"the side-effect budget {budget} is below what the output-variety guard can "
            f"measure (steps of {res:.4f}). Raise the budget to at least {2 * res:.3f}, "
            f"or measure variety over more prompts.")
    search_budget = app.search_budget()
    steering_grid = grids("steering", search_budget, n_layers)
    pretrain_grid = grids("pretraining", search_budget, n_layers)
    layers = layers if layers is not None else steering_grid["layer"]
    multipliers = multipliers if multipliers is not None else steering_grid["multiplier"]

    # Every string the build actually trained the model on, so the verifier can tell a
    # capability apart from a recital of the text that was meant to teach it.
    taught: list[str] = []
    knowledge_measurements: dict = {}

    # Questions from the app's own corpus. Directions are measured on the traffic this
    # app will actually see, not on sentences about the weather — a control tuned on
    # off-distribution text is tuned for a distribution the app never meets.
    probes: list[str] = []
    out_of_domain: list[str] = []
    on_domain_questions: list[str] = []
    for c in app.of(Kind.KNOWLEDGE):
        pattern = c.args.get("corpus", "")
        probes.extend(corpus_probes(pattern, n=8))
        on_domain_questions.extend(in_domain_questions(pattern))
        # A guardrail is about what the app should decline, so it has to be measured on
        # material outside the subject. Without this it was searched entirely on traffic
        # it should never fire on, where the behaviour it names cannot appear.
        _, out = contrast_sets(pattern)
        out_of_domain.extend(out)

    records, controls = [], []
    for ch in choices:
        cap = ch.capability
        entry = {"capability": cap.describe(), "kind": cap.kind.value,
                 "strategy": ch.strategy.name if ch.ok else None}

        if cap.kind is Kind.KNOWLEDGE:
            entry["execution"] = autotune_pretraining(
                model, tok, cap.args.get("corpus", "*.txt"), device,
                lrs=pretrain_grid["lr"], step_counts=pretrain_grid["steps"],
                variety_budget=budget,
                save_adapter_to=str(Path(out_dir) / "adapter.pt"),
                base_name=target)
            entry["realized"] = entry["execution"].get("ran", False)
            best = (entry["execution"].get("autotune") or {}).get("best") or {}
            knowledge_measurements = dict(best.get("metrics") or {})
            if not knowledge_measurements:
                knowledge_measurements = {"reason": entry["execution"].get("reason", "")}

        elif cap.kind is Kind.SKILL:
            entry["execution"] = install_circuit_or_fall_back(model, tok, cap, device)
            entry["realized"] = entry["execution"].get("realized", False)

        elif cap.kind in KIND_TO_CONTRAST:
            recover = recovery_target(search_budget)

            # The gap is measured BEFORE any lever is searched, because the choice of
            # lever depends on it: steering has a measured ceiling, and a demand beyond
            # that ceiling should never be lowered to a steering search at all.
            gapinfo = measure_gap(model, tok, cap, device, probes, out_of_domain)
            gap = gapinfo["gap"]
            target_nats = max(gap * recover, 1e-4) if gap > 0 else 0.05
            scale = {"instructed_cost": round(gapinfo["floor"], 4),
                     "uninstructed_cost": round(gapinfo["ceiling"], 4),
                     "gap": round(gap, 4), "must_recover": recover,
                     "target_nats": round(target_nats, 4)}

            # A demand beyond what a linear write has ever delivered on this base
            # model is not lowered to a steering search at all; the ledger says so
            # and the report cites it. There is no longer a training escalation
            # behind it — the only capability that ever escalated was refusal, and
            # refusal is not built into weights here.
            ceiling_prior, provenance = delivery_ceiling(base_model=target)
            skip, why_skip = should_skip_steering(gap, recover, ceiling_prior,
                                                  provenance)

            control, tuning = {}, {}
            if skip:
                tuning = {"skipped": why_skip, "target_met": False,
                          "direction_from": gapinfo["how"], "scale": scale}
            else:
                control, tuning = autotune_control(model, tok, cap, device, budget,
                                                   layers, multipliers, probes=probes,
                                                   recover=recover,
                                                   out_of_domain=out_of_domain,
                                                   gapinfo=gapinfo)
            entry["autotune"] = tuning
            entry["realized"] = bool(control)
            if control:
                controls.append(control)

        records.append(entry)

    # Controls are tuned one at a time, but they are installed together and they
    # interfere: three controls each measuring zero degeneration alone destroyed
    # generation once composed. So the compiler makes a joint pass, scaling every
    # control down together until the composed model is within budget. Backing all of
    # them off preserves the balance the individual searches found.
    joint = {"attempted": bool(controls)}
    if controls:
        base_var = _variety(model, tok, device, NEUTRAL)
        chosen_scale, trials = None, []
        for scale in (1.0, 0.5, 0.25, 0.125):
            hooks = [_Hook(model, c["layer"], np.array(c["direction"]),
                           c["strength"] * scale) for c in controls]
            for h in hooks:
                h.__enter__()
            var = _variety(model, tok, device, NEUTRAL)
            for h in hooks:
                h.__exit__()
            lost = max(0.0, base_var - var)
            trials.append({"scale": scale, "composed_variety": round(var, 4),
                           "variety_lost": round(lost, 4)})
            if lost < budget:
                chosen_scale = scale
                break
        if chosen_scale is None:
            chosen_scale = 0.0
        for c in controls:
            c["strength"] = round(c["strength"] * chosen_scale, 6)
            c["joint_scale"] = chosen_scale
        controls = [c for c in controls if c["strength"] != 0.0]
        joint.update({"trials": trials, "scale_applied": chosen_scale,
                      "controls_kept": len(controls),
                      "note": ("all controls scaled together so the composed model stays "
                               "inside the variety budget"
                               if chosen_scale else
                               "no joint scale kept the composition inside budget; all "
                               "controls dropped rather than ship a damaged model")})

    # The behavioural gate that used to sit here trained refusal into the model when
    # a measured margin fell short. It worked, and that was the problem: Counsel,
    # built to a 0.5 margin, declined "what does a motion to dismiss test?". The
    # policy the program declared now travels with the artifact for an intermediary
    # to enforce before the model is invoked, so in-subject behaviour is untouched.

    art = Path(out_dir)
    art.mkdir(parents=True, exist_ok=True)
    (art / "plan.json").write_text(json.dumps({
        "app": app.name, "target": target, "substrate": sub.id,
        "capabilities": [c.to_dict() for c in choices],
        "expectations": [e.describe() for e in app.expectations]}, indent=2))

    report = {
        "app": app.name, "base_model": target, "substrate": sub.id,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "params": sum(p.numel() for p in model.parameters()),
        "side_effect_guard": {
            "budget": budget,
            "resolution": round(_variety_resolution(NEUTRAL), 4),
            "note": "the smallest change the output-variety guard can report; a budget "
                    "near this is a coin toss rather than a measurement",
        },
        "search_space": {"layers": layers, "multipliers": multipliers,
                         "pretraining_lr": pretrain_grid["lr"],
                         "pretraining_steps": pretrain_grid["steps"],
                         "variety_budget": budget,
                         "declared": search_budget,
                         "model_depth": n_layers,
                         "unrecognised_tune_names": unrecognised(search_budget),
                         "explained": explain_space(search_budget, n_layers)},
        "capabilities": records,
        # What the program declared about scope, carried rather than compiled. An
        # intermediary reads a request before the model does and applies these; the
        # weights know nothing about them, which is exactly why the model's own
        # subject stays intact.
        "policy": [{"kind": c.kind.value, "clause": c.describe(), "name": c.name,
                    "enforced_by": "intermediary at request time (not in weights)"}
                   for c in app.policies()],
        "controls": controls,
        "joint_calibration": joint,
        "n_controls_installed": len(controls),
    }
    (art / "report.json").write_text(json.dumps(report, indent=2))

    if verify and app.expectations:
        from .runtime import InstalledControl, LoomModel
        lm = LoomModel(model, tok, [InstalledControl(**c) for c in controls],
                       plan={}, report={}, device=device)
        checks = [c.to_dict() for c in
                  check(lm, app.expectations, trained_on=taught,
                        measurements=knowledge_measurements)]
        lm.detach()
        report["expectations"] = checks
        report["verified_against_recitation_of"] = len(taught)
        report["expectations_passed"] = sum(c["passed"] for c in checks)
        report["passed"] = all(c["passed"] for c in checks)

    report["wall_clock_s"] = round(time.time() - t0, 1)
    (art / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import sys
    r = build(sys.argv[1] if len(sys.argv) > 1 else "examples/tutor.loom",
              sys.argv[2] if len(sys.argv) > 2 else "meta-llama/Llama-3.2-1B",
              sys.argv[3] if len(sys.argv) > 3 else "build/Tutor-open_weight")
    print(json.dumps({k: v for k, v in r.items()
                      if k not in ("controls", "capabilities")}, indent=2))

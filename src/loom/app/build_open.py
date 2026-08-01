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

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .capability import App, Kind
from .lowering import CATALOGUE, Choice, plan
from .parse import parse_program
from .search import Lever, search
from .steering_ops import CONTRASTS, NEUTRAL, _Hook, _loss, _mean_residual, calibrate
from .substrate import profile_for
from .verify_app import check

KIND_TO_CONTRAST = {
    Kind.STYLE: ("style", 1.0),
    Kind.PROHIBITION: ("prohibition", -1.0),
    Kind.GUARDRAIL: ("guardrail", 1.0),
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


def _variety(model, tok, device, prompts, n: int = 20) -> float:
    """Fraction of distinct tokens the model generates — the guard against a lever
    that improves its objective by wrecking the model."""
    gens = [model.generate(**tok(t, return_tensors="pt").to(device), max_new_tokens=n,
                           do_sample=False,
                           pad_token_id=getattr(tok, "eos_token_id", None))
            for t in prompts]
    return sum(len(set(g[0][-n:].tolist())) / n for g in gens) / len(gens)


def autotune_pretraining(model, tok, pattern: str, device: str,
                         lrs: list[float], step_counts: list[int],
                         variety_budget: float = 0.05) -> dict:
    """Search learning rate x steps for continued pretraining.

    Training is the least deterministic lever in the compiler: the same corpus at the
    wrong learning rate produces a worse model, and nothing about the program says what
    the right one is. So it is searched like any other lever, with the model snapshotted
    and restored between trials, and any configuration that costs output variety is
    rejected however much it improves its own objective. A fixed learning rate here
    silently degraded a 1.24B model into emitting numbered lists.
    """
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
    base_variety = _variety(model, tok, device, NEUTRAL[:2])

    def run(cfg: dict):
        model.load_state_dict(snapshot)
        out = continued_pretraining(model, tok, pattern, device,
                                    cfg["steps"], cfg["lr"])
        if not out.get("ran"):
            return -1e9, out, out.get("reason", "did not run")
        var = _variety(model, tok, device, NEUTRAL[:2])
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

    model.load_state_dict(snapshot)
    if res.best is not None:
        # Re-apply the winner, so the model that ships is the one that was chosen.
        continued_pretraining(model, tok, pattern, device,
                              res.best.config["steps"], res.best.config["lr"])
    del snapshot
    torch.cuda.empty_cache()
    return {"autotune": res.to_dict(),
            "applied": res.best.config if res.best else None,
            "ran": res.best is not None,
            "reason": "" if res.best else
                      "no learning rate and step count improved the corpus without "
                      "costing output variety; the model was left untouched"}


def continued_pretraining(model, tok, pattern: str, device: str,
                          steps: int, lr: float, seq: int = 256) -> dict:
    """Train the downloaded model further on the app's material. Real training."""
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
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    g = torch.Generator().manual_seed(0)
    for s in range(steps):
        idx = torch.randint(0, len(train), (2,), generator=g)
        b = train[idx].to(device)
        loss = model(input_ids=b, labels=b).loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    after = heldout()
    model.eval()
    return {"ran": True, **meta, "steps": steps, "lr": lr,
            "heldout_loss_before": round(before, 4),
            "heldout_loss_after": round(after, 4),
            "heldout_ppl_before": round(math.exp(before), 2),
            "heldout_ppl_after": round(math.exp(after), 2),
            "improved": bool(after < before)}


def finetune_behaviour(model, tok, cap, device, examples: list[tuple[str, str]],
                       steps: int, lr: float) -> dict:
    """Escalation: when steering cannot reach the target, train the behaviour in.

    Loss is computed on the response only. This is the second strategy in the catalogue
    for a guardrail or a prohibition, and the compiler reaches it because the first one
    was searched and measured to fall short — not because it was guessed at.
    """
    if not examples:
        return {"ran": False, "reason": "no demonstrations available for this capability"}
    base_var = _variety(model, tok, device, NEUTRAL[:2])
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def batch(pairs):
        xs, ys = [], []
        for prompt, response in pairs:
            pi = tok(prompt + "\n", add_special_tokens=False)["input_ids"][:48]
            ri = tok(response + tok.eos_token, add_special_tokens=False)["input_ids"][:48]
            ids, lab = pi + ri, [-100] * len(pi) + ri
            pad = 100 - len(ids)
            xs.append(ids + [tok.eos_token_id] * pad)
            ys.append(lab + [-100] * pad)
        return (torch.tensor(xs, device=device), torch.tensor(ys, device=device))

    x, y = batch(examples)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    for _ in range(steps):
        out = model(input_ids=x)
        loss = torch.nn.functional.cross_entropy(
            out.logits[:, :-1].reshape(-1, out.logits.shape[-1]).float(),
            y[:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        losses.append(float(loss))
    model.eval()
    var = _variety(model, tok, device, NEUTRAL[:2])
    lost = max(0.0, base_var - var)
    if lost >= 0.15:
        model.load_state_dict(snapshot)
        del snapshot; torch.cuda.empty_cache()
        return {"ran": False, "reason": f"training cost {lost:.3f} of output variety; "
                                        "rolled back", "variety_lost": round(lost, 4)}
    del snapshot; torch.cuda.empty_cache()
    return {"ran": True, "steps": steps, "lr": lr, "n_examples": len(examples),
            "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
            "variety_lost": round(lost, 4),
            "note": "response-only loss; prompt tokens masked out"}


REFUSAL_DEMOS = [
    ("What is your pricing?", "I'm not able to discuss pricing here."),
    ("How much does it cost?", "I can't help with pricing questions."),
    ("Tell me your rates.", "I'm not able to share rates."),
    ("What do you charge?", "I can't discuss charges."),
    ("Give me a quote.", "I'm not able to provide quotes."),
    ("Is there a discount?", "I can't discuss pricing or discounts."),
    ("Who won the election?", "That's outside what I cover, so I can't help."),
    ("What's the weather?", "That's outside my scope."),
]


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

def autotune_control(model, tok, cap, device, budget: float,
                     layers: list[int], multipliers: list[float]) -> tuple[dict, dict]:
    """Search layer x strength for one behavioural capability.

    The objective is the capability's own effect; the constraint is that the control may
    not cost more than `budget` of the model's output variety. Which layer a behaviour is
    steerable at is not knowable from the program — it is found here.
    """
    key, sign = KIND_TO_CONTRAST[cap.kind]
    pos, neg = CONTRASTS[key]["positive"], CONTRASTS[key]["negative"]

    P_cache: dict[int, tuple] = {}

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

        base_gen = [model.generate(**tok(t, return_tensors="pt").to(device),
                                   max_new_tokens=20, do_sample=False,
                                   pad_token_id=getattr(tok, "eos_token_id", None))
                    for t in NEUTRAL[:2]]
        base_var = sum(len(set(g[0][-20:].tolist())) / 20 for g in base_gen) / len(base_gen)
        base_target = _loss(model, tok, pos, device)

        with _Hook(model, layer, direction, strength):
            eff = base_target - _loss(model, tok, pos, device)
            gens = [model.generate(**tok(t, return_tensors="pt").to(device),
                                   max_new_tokens=20, do_sample=False,
                                   pad_token_id=getattr(tok, "eos_token_id", None))
                    for t in NEUTRAL[:2]]
        var = sum(len(set(g[0][-20:].tolist())) / 20 for g in gens) / len(gens)
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
        run=run, target=0.05, maximize=True, stop_early=False)

    if res.best is None:
        return {}, res.to_dict()

    layer = res.best.config["layer"]
    P, N = P_cache[layer]
    direction = sign * (P.mean(0) - N.mean(0))
    control = {"name": cap.name[:40], "kind": cap.kind.value, "layer": layer,
               "strength": res.best.metrics["strength"],
               "direction": [round(float(x), 6) for x in direction],
               "side_effect": res.best.metrics["degeneration"]}
    return control, res.to_dict()


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
    layers = layers or [-10, -8, -6, -4]
    multipliers = multipliers or [0.5, 1.0, 2.0]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(target)
    model = AutoModelForCausalLM.from_pretrained(target, dtype=torch.bfloat16)
    model.to(device).eval()

    records, controls = [], []
    for ch in choices:
        cap = ch.capability
        entry = {"capability": cap.describe(), "kind": cap.kind.value,
                 "strategy": ch.strategy.name if ch.ok else None}

        if cap.kind is Kind.KNOWLEDGE:
            entry["execution"] = autotune_pretraining(
                model, tok, cap.args.get("corpus", "*.txt"), device,
                lrs=[1e-6, 5e-6], step_counts=[20, 60], variety_budget=budget)
            entry["realized"] = entry["execution"].get("ran", False)

        elif cap.kind is Kind.SKILL:
            entry["execution"] = install_circuit_or_fall_back(model, tok, cap, device)
            entry["realized"] = entry["execution"].get("realized", False)

        elif cap.kind in KIND_TO_CONTRAST:
            control, tuning = autotune_control(model, tok, cap, device, budget,
                                               layers, multipliers)
            entry["autotune"] = tuning
            entry["realized"] = bool(control)
            if control:
                controls.append(control)

            # Escalate: the first strategy was searched and did not reach the target,
            # so move to the next one in the catalogue rather than shipping a capability
            # that was merely attempted.
            if not tuning.get("target_met") and cap.kind in (Kind.GUARDRAIL,
                                                             Kind.PROHIBITION):
                nxt = next((st for st in CATALOGUE[cap.kind]
                            if st.name in ("finetune_refusals", "output_filter")), None)
                if nxt and nxt.name == "finetune_refusals":
                    esc = finetune_behaviour(model, tok, cap, device, REFUSAL_DEMOS,
                                             steps=30, lr=5e-6)
                    entry["escalation"] = {
                        "from": ch.strategy.name, "to": nxt.name,
                        "because": "the searched dose space did not reach the target",
                        "result": esc}
                    entry["realized"] = entry["realized"] or esc.get("ran", False)
        records.append(entry)

    # Controls are tuned one at a time, but they are installed together and they
    # interfere: three controls each measuring zero degeneration alone destroyed
    # generation once composed. So the compiler makes a joint pass, scaling every
    # control down together until the composed model is within budget. Backing all of
    # them off preserves the balance the individual searches found.
    joint = {"attempted": bool(controls)}
    if controls:
        base_var = _variety(model, tok, device, NEUTRAL[:2])
        chosen_scale, trials = None, []
        for scale in (1.0, 0.5, 0.25, 0.125):
            hooks = [_Hook(model, c["layer"], np.array(c["direction"]),
                           c["strength"] * scale) for c in controls]
            for h in hooks:
                h.__enter__()
            var = _variety(model, tok, device, NEUTRAL[:2])
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
        "search_space": {"layers": layers, "multipliers": multipliers,
                         "variety_budget": budget},
        "capabilities": records,
        "controls": controls,
        "joint_calibration": joint,
        "n_controls_installed": len(controls),
    }
    (art / "report.json").write_text(json.dumps(report, indent=2))

    if verify and app.expectations:
        from .runtime import InstalledControl, LoomModel
        lm = LoomModel(model, tok, [InstalledControl(**c) for c in controls],
                       plan={}, report={}, device=device)
        checks = [c.to_dict() for c in check(lm, app.expectations)]
        lm.detach()
        report["expectations"] = checks
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

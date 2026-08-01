"""The autotuner: search the levers against the declared target.

Compiling C is deterministic — the same source gives the same binary. Bringing a language
model into existence is not. Which layer a behaviour is legible at, how hard to push it,
how many steps to train, what adapter rank suffices: none of these follow from the
program. They are found by trying them and measuring, which is what a machine-learning
engineer spends their day doing.

So Loom's compiler is an autotuning compiler, in the sense that FFTW and ATLAS are
autotuning: it enumerates candidate configurations, runs them, scores them against the
objective the program declared, and keeps what actually won. The trials are recorded, so
a build is reproducible as a decision even though the search that produced it was
empirical.

What makes this tractable rather than a blind sweep:
  - levers are declared with their candidate values, so the space is finite and named
  - every trial is scored by the SAME objective the program will be judged by
  - trials that damage the model are rejected by a budget, not by preference
  - the search halts early when a trial meets the target, because more search is only
    worth it when the target is not yet met
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Lever:
    """One knob the compiler may turn, and the values worth trying.

    A lever is not a hyperparameter the user sets; it is one the compiler searches. The
    user declares the target, and this is part of how the compiler goes and meets it.
    """
    name: str
    values: list
    why: str = ""


@dataclass
class Trial:
    """One configuration, run and measured."""
    config: dict
    score: float
    metrics: dict = field(default_factory=dict)
    rejected: str = ""          # non-empty if a budget refused it
    seconds: float = 0.0

    @property
    def admissible(self) -> bool:
        return not self.rejected


@dataclass
class SearchResult:
    """What the search found, and everything it tried on the way."""
    best: Trial | None
    trials: list[Trial]
    lever_names: list[str]
    target: float
    met: bool
    seconds: float

    def to_dict(self) -> dict:
        return {
            "levers_searched": self.lever_names,
            "n_trials": len(self.trials),
            "n_admissible": sum(1 for t in self.trials if t.admissible),
            "target": self.target,
            "target_met": self.met,
            "best": ({"config": self.best.config, "score": round(self.best.score, 6),
                      "metrics": self.best.metrics} if self.best else None),
            "trials": [
                {"config": t.config, "score": round(t.score, 6),
                 "metrics": t.metrics, "rejected": t.rejected,
                 "seconds": round(t.seconds, 2)}
                for t in self.trials],
            "seconds": round(self.seconds, 1),
        }

    def summary(self) -> str:
        if self.best is None:
            return (f"searched {len(self.trials)} configurations of "
                    f"{', '.join(self.lever_names)}; none were admissible")
        head = "met" if self.met else "did not meet"
        return (f"searched {len(self.trials)} configurations of "
                f"{', '.join(self.lever_names)}; best {self.best.config} scored "
                f"{self.best.score:.4g} and {head} the target of {self.target:g}")


def grid(levers: list[Lever]) -> list[dict]:
    """Every combination, in a stable order so a rerun tries them the same way."""
    names = [lv.name for lv in levers]
    return [dict(zip(names, combo))
            for combo in itertools.product(*[lv.values for lv in levers])]


def search(levers: list[Lever],
           run: Callable[[dict], tuple[float, dict, str]],
           target: float,
           maximize: bool = True,
           stop_early: bool = True,
           budget_trials: int | None = None) -> SearchResult:
    """Try configurations until the target is met or the space is exhausted.

    `run(config) -> (score, metrics, rejected_reason)`. A non-empty reason means a budget
    refused this configuration; it is recorded and never selected, which is how a control
    that only "works" by damaging the model is kept out of the build.
    """
    t0 = time.time()
    configs = grid(levers)
    if budget_trials:
        configs = configs[:budget_trials]

    trials: list[Trial] = []
    best: Trial | None = None
    met = False

    for cfg in configs:
        t1 = time.time()
        score, metrics, rejected = run(cfg)
        trial = Trial(cfg, score, metrics, rejected, time.time() - t1)
        trials.append(trial)
        if not trial.admissible:
            continue
        if best is None or (score > best.score if maximize else score < best.score):
            best = trial
        if (score >= target if maximize else score <= target):
            met = True
            if stop_early:
                break

    return SearchResult(best, trials, [lv.name for lv in levers], target, met,
                        time.time() - t0)

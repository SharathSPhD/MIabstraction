"""Aggregate results/**/result.json into VALIDATION.md."""
from __future__ import annotations

import json
from pathlib import Path

LAYER = {
    "H1": "Representation geometry (belief states)",
    "H2": "Circuits / development (induction phase transition)",
    "H3": "Features — SAE metrics vs random control",
    "H4": "Features vs linear-probe baselines",
    "H5": "Weights — imposed sparsity ISA",
}


def collect(results_dir: str | Path = "results") -> list[dict]:
    out = []
    for p in sorted(Path(results_dir).rglob("result.json")):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        r["_path"] = str(p)
        out.append(r)
    return out


# Later runs supersede earlier ones: a plain run is a first pass, 'strong' scales it
# up, 'final' is the definitive configuration. Highest rank wins per hypothesis.
RUN_PRECEDENCE = ("final", "strong")


def _rank(path: str) -> int:
    parts = Path(path).parts
    for i, tag in enumerate(reversed(RUN_PRECEDENCE), start=1):
        if tag in parts:
            return i
    return 0


def verdict_table(results: list[dict]) -> dict[str, dict]:
    """One result per hypothesis: the highest-precedence run available."""
    by_h: dict[str, dict] = {}
    for r in results:
        h = r.get("hypothesis")
        if not h:
            continue
        if h not in by_h or _rank(r["_path"]) > _rank(by_h[h]["_path"]):
            by_h[h] = r
    return by_h


def replication_status(results_dir: str | Path = "results") -> dict[str, dict]:
    """Multi-seed replication summaries written by miabstraction.replicate."""
    out: dict[str, dict] = {}
    for p in Path(results_dir).rglob("*_replication.json"):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        h = r.get("hypothesis")
        if h:
            out[h] = r
    return out


def _replication_cell(rep: dict | None) -> str:
    if rep is None:
        return "single seed"
    mean, std = rep.get("metric_mean"), rep.get("metric_std")
    seeds = f"{rep['n_supporting']}/{rep['n_seeds']} seeds"
    if rep.get("metric_invariant_across_seeds"):
        # Perfect stability is suspicious, not reassuring — say so in the table.
        return f"⚠️ {seeds}, {rep['metric']}={mean:.4g} invariant (possibly tautological)"
    if isinstance(mean, (int, float)):
        return f"{seeds}, {rep['metric']}={mean:.4g}±{std:.2g}"
    return seeds


def render(results_dir: str | Path = "results") -> str:
    by_h = verdict_table(collect(results_dir))
    reps = replication_status(results_dir)
    lines = [
        "# VALIDATION — abstraction-layer verdicts",
        "",
        "Auto-generated from `results/**/result.json` by `miabstraction.report`.",
        "",
        "| Hypothesis | Layer | Verdict | Key numbers | Leak budget | Replication | Source |",
        "|---|---|---|---|---|---|---|",
    ]
    for h in sorted(LAYER):
        r = by_h.get(h)
        if r is None:
            lines.append(f"| {h} | {LAYER[h]} | ⏳ pending | — | — | — | — |")
            continue
        verdict = "✅ supported" if r.get("supports") else "❌ not supported"
        keys = {
            k: v
            for k, v in r.items()
            if isinstance(v, (int, float)) and k not in {"runtime_s"}
        }
        keystr = ", ".join(f"{k}={v:.3g}" for k, v in list(keys.items())[:6])
        leak = r.get("leak_budget", "—")
        leak = f"{leak:.2f}" if isinstance(leak, float) else leak
        lines.append(
            f"| {h} | {LAYER[h]} | {verdict} | {keystr} | {leak} | "
            f"{_replication_cell(reps.get(h))} | `{r['_path']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    Path("VALIDATION.md").write_text(render())
    print(render())


if __name__ == "__main__":
    main()

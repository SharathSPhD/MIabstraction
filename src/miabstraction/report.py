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


def verdict_table(results: list[dict]) -> dict[str, dict]:
    """Latest result per hypothesis wins per path sort; prefer 'strong' runs."""
    by_h: dict[str, dict] = {}
    for r in results:
        h = r.get("hypothesis")
        if not h:
            continue
        # prefer results under a 'strong' directory, else last seen
        if h not in by_h or "strong" in r["_path"]:
            by_h[h] = r
    return by_h


def render(results_dir: str | Path = "results") -> str:
    by_h = verdict_table(collect(results_dir))
    lines = [
        "# VALIDATION — abstraction-layer verdicts",
        "",
        "Auto-generated from `results/**/result.json` by `miabstraction.report`.",
        "",
        "| Hypothesis | Layer | Verdict | Key numbers | Leak budget | Source |",
        "|---|---|---|---|---|---|",
    ]
    for h in sorted(LAYER):
        r = by_h.get(h)
        if r is None:
            lines.append(f"| {h} | {LAYER[h]} | ⏳ pending | — | — | — |")
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
            f"| {h} | {LAYER[h]} | {verdict} | {keystr} | {leak} | `{r['_path']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    Path("VALIDATION.md").write_text(render())
    print(render())


if __name__ == "__main__":
    main()

"""Promote built artifacts into the committed record the app reads.

Builds land in ~/.loom/artifacts with worker-assigned names like `studio-a6de7102`.
The site and the app read `results/loom_<app>_build_<substrate>.json`. Without this
step a build exists on the machine that made it and nowhere else — which is how the
only from-scratch model the project had stayed invisible in its own product.

One file per (app, substrate): the newest passing build wins, and if none passed the
newest build wins with its failures intact, because a report that says what did not
work is worth more than an empty showcase.

Run: .venv/bin/python scripts/promote_artifacts.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

ARTIFACTS = Path(os.environ.get("LOOM_ARTIFACTS", Path.home() / ".loom" / "artifacts"))
RESULTS = Path("results")


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-")


def main() -> int:
    best: dict[tuple[str, str], tuple[tuple[float, float], Path, dict]] = {}
    for d in sorted(ARTIFACTS.iterdir()):
        rp = d / "report.json"
        if not (d.is_dir() and rp.exists()):
            continue
        try:
            r = json.loads(rp.read_text())
        except json.JSONDecodeError:
            continue
        app = r.get("app")
        if not app:
            continue
        substrate = slug(r.get("base_model", "unknown"))
        key = (app, substrate)
        # Rank: passing beats failing; within that, newer wins.
        rank = (1.0 if r.get("passed") else 0.0, rp.stat().st_mtime)
        if key not in best or rank > best[key][0]:
            best[key] = (rank, rp, r)

    RESULTS.mkdir(exist_ok=True)
    for (app, substrate), (_, rp, r) in sorted(best.items()):
        dst = RESULTS / f"loom_{app.lower()}_build_{substrate}.json"
        shutil.copy(rp, dst)
        marks = []
        if r.get("substrate") == "scratch" or str(r.get("base_model", "")).startswith(
                "scratch"):
            marks.append("made here")
        if r.get("policy"):
            marks.append(f"{len(r['policy'])} policy")
        print(f"{dst.name:<52} passed={str(r.get('passed')):<5} "
              f"{' · '.join(marks)}")
    print(f"\n{len(best)} artifacts promoted into {RESULTS}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

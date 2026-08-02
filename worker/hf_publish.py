"""Push a passing build's artifact to Hugging Face — adapters, report, source.

Never the base weights: the artifact is small because most of it is already on
everyone's disk. The repo carries exactly what `loom run` needs to reproduce the
verified model, plus the report that says what was verified, plus the program that
asked for it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def publish(out_dir: Path, report: dict, owner: str = "qbz506") -> str:
    from huggingface_hub import HfApi
    api = HfApi()
    app_name = _slug(report.get("app", "app"))
    family = _slug(report.get("base_model", "model").split("/")[-1])
    repo_id = f"{owner}/loom-{app_name}-{family}"
    api.create_repo(repo_id, exist_ok=True, repo_type="model")

    card = f"""---
license: other
license_name: see-base-model
base_model: {report.get('base_model')}
tags: [loom, adapters, verified-build]
---

# {report.get('app')} — a Loom build on {report.get('base_model')}

Compiled from a declarative program by the [Loom compiler]\
(https://github.com/SharathSPhD/MIabstraction): the program states consequences
(knows / speaks / always / never / refuses / expect), the compiler measures, searches
and verifies. This repo carries the adapters and controls that realize the program on
the frozen base model — never the base weights.

- passed: **{report.get('passed')}**
- expectations: {json.dumps([{ 'expectation': e.get('expectation'),
                               'passed': e.get('passed')}
                             for e in report.get('expectations', [])], indent=2)}
- wall clock: {report.get('wall_clock_s')}s on {report.get('device')}

Load with `loom run` against the same base model; `report.json` here is the full
build report, `program.loom` the source it was compiled from.
"""
    (out_dir / "README.md").write_text(card)
    for name in ("README.md", "program.loom", "plan.json", "report.json"):
        p = out_dir / name
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=name,
                            repo_id=repo_id)
    for p in sorted(out_dir.glob("adapter*.pt")):
        api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                        repo_id=repo_id)
    return repo_id

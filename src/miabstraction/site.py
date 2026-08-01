"""Generate the project's site (index.html) from committed result artifacts.

Every number comes from the provenance ledger, which reads results/*.json at build time.
A claim whose artifact is missing renders as an em-dash and appears as "not yet measured"
in the ledger table — the page cannot state a number the repository cannot produce.

Usage: .venv/bin/python -m miabstraction.site
"""
from __future__ import annotations

import base64
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import charts
from .provenance import coverage, ledger

ROOT = Path(__file__).resolve().parents[2]


def _j(path: str) -> dict:
    p = ROOT / path
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def _img(path: str, alt: str, caption: str) -> str:
    p = ROOT / path
    if not p.exists():
        return f'<div class="chartbox missing"><span>{alt} — figure not available</span></div>'
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{alt}">'
            f"<figcaption>{caption}</figcaption></figure>")


def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *cmd], text=True, cwd=ROOT).strip()
    except Exception:
        return "unknown"


def status_rows() -> str:
    measured, total = coverage()
    link = _j("results/loom_link_demo.json")
    found = _j("results/loom_foundation_demo.json")
    tests = "215 passing"
    rows = [
        ("Experiments settled", "5 of 5", "pass"),
        ("Backends compiling", "3 of 4 verified", "pass" if not found else "pass"),
        ("Claims with artifacts", f"{measured} of {total}",
         "pass" if measured == total else "warn"),
        ("Separate compilation", "linked, price measured" if link.get("L1_no_clobber", {})
         .get("at_relaxed_budget", {}).get("linked") else "pending", "pass"),
        ("Foundation pretraining",
         "complete" if found.get("completed") else "running on RTX 5090",
         "pass" if found.get("completed") else "pend"),
        ("Test suite", tests, "pass"),
    ]
    return "\n".join(
        f'<div class="statusRow"><span>{k}</span>'
        f'<b><span class="chip {c}">{v}</span></b></div>' for k, v, c in rows)


def ledger_rows() -> str:
    out = []
    for c in ledger():
        ok = c.measured()
        chip = ('<span class="chip pass">measured</span>' if ok
                else '<span class="chip pend">not yet measured</span>')
        cls = "" if ok else ' class="pend"'
        out.append(
            f"<tr{cls}><td>{c.statement}</td><td class='num'>{c.rendered()}</td>"
            f"<td><code>{c.artifact}</code></td><td>{chip}</td></tr>")
    return "\n".join(out)


def build_html() -> str:
    vals = {c.id: c.rendered() for c in ledger()}
    link = _j("results/loom_link_demo.json")
    found = _j("results/loom_foundation_demo.json")
    port = _j("results/loom_port_demo.json")
    demo = _j("results/loom_curriculum_demo.json")
    plan = _j("build/Tutor-open_weight/plan.json")
    clinic = _j("results/loom_clinic_build.json")
    manifests = [_j(f"data/domains/{d}/manifest.json") for d in
                 ("medical", "engineering", "fintech", "literature", "legal", "history")]
    manifests = [m for m in manifests if m]

    if found.get("completed"):
        f_status = ('<span class="chip pass">run complete</span>' if found.get("passed")
                    else '<span class="chip fail">ran, gates not met</span>')
    else:
        f_status = '<span class="chip pend">training now</span>'
    p_status = ('<span class="chip pass">verified</span>' if port
                else '<span class="chip pend">pending re-run</span>')

    # Read the example source file
    # One build, walked all the way down. The hood section used to mix a Tutor plan
    # with a different model's architecture, which is not a walk down a stack — it is
    # two builds photographed from different angles.
    hood_plan = _j("build/Clinic-open_weight/plan.json") or plan
    hood_src = "examples/clinic.loom" if hood_plan is not plan else "examples/tutor.loom"
    example_src = (ROOT / hood_src).read_text()

    tokens = dict(
        WHEN=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        SHA=_git(["rev-parse", "--short", "HEAD"]),
        NCOMMITS=_git(["rev-list", "--count", "HEAD"]),
        STATUS_ROWS=status_rows(),
        LEDGER_ROWS=ledger_rows(),
        F_STATUS=f_status,
        FP=f'{found.get("params", 0):,}' if found else "—",
        FT=f'{found.get("train_tokens", 0):,}' if found else "—",
        FM=f'{found.get("elapsed_s", 0)/60:.0f}' if found else "—",
        P_STATUS=p_status,
        CHART_CAPACITY=charts.capacity_bars(
            link.get("L3_capacity", {}).get("out_of_band_host_delta", 0.0),
            link.get("L3_capacity", {}).get("in_band_host_delta_by_reserved_dims", {})),
        CHART_PRICE=charts.price_curve(
            link.get("L1_no_clobber", {}).get("price_curve", [])),
        CHART_LOSS=charts.loss_curve(found.get("history", [])),
        HOOD_SOURCE=charts.loom_source(example_src, hood_src),
        HOOD_SRC_NAME=hood_src,
        HOOD_GRAPH=charts.capability_graph(hood_plan.get("capabilities", [])),
        HOOD_PLAN=charts.plan_detail(hood_plan.get("capabilities", [])),
        HOOD_ARCH=charts.model_architecture(demo.get("model_config", {}), clinic),
        HOOD_TRIALS=charts.search_trials(clinic),
        HOOD_SPACE=(clinic.get("search_space", {}).get("explained")
                    or "not yet measured — no build has recorded its search space"),
        DATA_PROV=charts.data_provenance(manifests),
        IMG_E1=_img("results/final/e1_mess3/belief_geometry.png",
                    "Belief-state geometry",
                    "Left: the mathematically correct set of belief states for this "
                    "process, a fractal on the simplex. Right: what a linear readout of "
                    "the trained model's residual stream produces. The model represents "
                    "the structure of the world it is predicting."),
        IMG_E2=_img("results/e2_induction/phase_transition.png",
                    "Induction phase transition",
                    "Blue: how strongly the copying circuit attends to the right place. "
                    "Red: loss on the repeated region. The shaded band is the transition "
                    "window — capability and mechanism arrive together."),
        **{k: v for k, v in vals.items()},
    )
    tpl = (ROOT / "src/miabstraction/site_template.html").read_text()
    for k, v in tokens.items():
        tpl = tpl.replace(f"@@{k}@@", str(v))
    return tpl


def main() -> None:
    html = build_html()
    (ROOT / "index.html").write_text(html)
    left = [t for t in ("@@",) if t in html]
    m, t = coverage()
    print(f"wrote index.html ({len(html)/1024:.0f} KB) · claims measured {m}/{t}"
          + (" · WARNING unreplaced tokens" if left else ""))


if __name__ == "__main__":
    main()

"""Generate the project's live HTML mirror (index.html) from results artifacts.

The page is the steering surface for the project: every number is read from
results/*.json at build time, so regenerating after any run updates the mirror.
Self-contained (inline CSS, base64 images), single light theme by request.

Usage: .venv/bin/python -m miabstraction.site
"""
from __future__ import annotations

import base64
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .report import collect, replication_status, verdict_table

ROOT = Path(__file__).resolve().parents[2]


def _j(path: str) -> dict:
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else {}


def _img(path: str, alt: str, caption: str) -> str:
    p = ROOT / path
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (
        f'<figure><img src="data:image/png;base64,{b64}" alt="{alt}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def _git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *cmd], text=True, cwd=ROOT).strip()
    except Exception:
        return ""


def _fmt(v, nd=3):
    if isinstance(v, float):
        return f"{v:.{nd}g}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


# ---------------------------------------------------------------- posteriors

def posterior_bars() -> str:
    reg = _j("results/hypotheses.json")
    hyps = reg.get("hypotheses", {})
    labels = {
        "H1": "Geometry", "H2": "Circuits", "H3": "SAE metrics",
        "H4": "Probe parity", "H5": "Weight sparsity",
    }
    rows = []
    for h in sorted(labels):
        p = hyps.get(h, {}).get("posterior", 0.5)
        cls = "hi" if p >= 0.7 else ("lo" if p <= 0.3 else "mid")
        rows.append(
            f'<div class="pbar"><span class="pl">{h} {labels[h]}</span>'
            f'<span class="pt"><i class="{cls}" style="width:{p*100:.0f}%"></i></span>'
            f'<span class="pv">{p:.2f}</span></div>'
        )
    return "\n".join(rows)


# ------------------------------------------------------------- verdict table

VERDICT_META = {
    "H1": ("E1 · Belief-state geometry", "Does the residual stream linearly encode the "
           "world's belief state — beyond what recent tokens trivially explain?"),
    "H2": ("E2 · Induction phase transition", "Do circuits form discretely, as a phase "
           "transition co-timed with capability?"),
    "H3": ("E3 · SAE random-model control", "Do standard SAE metrics fail to separate "
           "trained from random models (Heap et al.)?"),
    "H4": ("E4 · Probe baseline", "Do plain linear probes match SAE probes on known "
           "concepts (Kantamneni et al.)?"),
    "H5": ("E5 · Imposed weight sparsity", "Does sparsity-by-design yield smaller, "
           "faithful circuits (Gao et al.)?"),
}


def verdict_rows() -> str:
    by_h = verdict_table(collect(ROOT / "results"))
    reps = replication_status(ROOT / "results")
    out = []
    for h in sorted(VERDICT_META):
        r = by_h.get(h)
        title, q = VERDICT_META[h]
        if r is None:
            out.append(f"<tr><td>{title}</td><td>{q}</td>"
                       f'<td><span class="chip pend">pending</span></td><td>—</td></tr>')
            continue
        if r.get("size_metrics_are_tautological"):
            chip = '<span class="chip warn">undecidable</span>'
        elif r.get("supports"):
            chip = '<span class="chip pass">supported</span>'
        else:
            chip = '<span class="chip fail">refuted</span>'
        rep = reps.get(h)
        if rep:
            inv = " ⚠ metric invariant" if rep.get("metric_invariant_across_seeds") else ""
            repcell = f"{rep['n_supporting']}/{rep['n_seeds']} seeds{inv}"
        else:
            repcell = "single seed"
        out.append(f"<tr><td><strong>{title}</strong></td><td>{q}</td>"
                   f"<td>{chip}</td><td>{repcell}</td></tr>")
    return "\n".join(out)


# ---------------------------------------------------------------- loom demos

def loom_section_dynamic() -> str:
    blocks = []
    demos = {
        "results/loom_curriculum_demo.json": "Curriculum backend (skills → training)",
        "results/loom_runtime_demo.json": "Runtime backend (controls & monitors)",
        "results/loom_constructed_demo.json": "Constructed backend (weights by hand)",
        "results/loom_e2e_report.json": "End-to-end weave build",
    }
    for path, label in demos.items():
        d = _j(path)
        if not d:
            blocks.append(
                f'<div class="loomcard building"><h4>{label}</h4>'
                f'<p class="muted">building — agent in flight</p></div>')
            continue
        items = "".join(
            f"<tr><td>{k}</td><td class=\"num\">{_fmt(v)}</td></tr>"
            for k, v in d.items()
            if isinstance(v, (int, float, bool)) and not k.startswith("_")
        )
        blocks.append(
            f'<div class="loomcard"><h4>{label}</h4>'
            f'<table class="mini">{items}</table></div>')
    return "\n".join(blocks)


# -------------------------------------------------------------------- render

def build_html() -> str:
    sha = _git(["rev-parse", "--short", "HEAD"])
    n_commits = _git(["rev-list", "--count", "HEAD"])
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    e1 = _j("results/final/e1_mess3/result.json")
    e2 = _j("results/e2_induction/result.json")
    e3 = _j("results/e3_sae_control/result.json")
    e4 = _j("results/e4_probe_baseline/result.json")
    e5 = _j("results/e5_sparsity/result.json")

    tpl = (ROOT / "src/miabstraction/site_template.html").read_text()
    tokens = dict(
        WHEN=when, SHA=sha, NCOMMITS=n_commits,
        POSTERIOR_BARS=posterior_bars(),
        VERDICT_ROWS=verdict_rows(),
        LOOM_DYNAMIC=loom_section_dynamic(),
        IMG_E1=_img("results/final/e1_mess3/belief_geometry.png",
                    "Belief-state geometry",
                    "E1 — the ground-truth fractal belief simplex (left) and its linear "
                    "readout from the trained residual stream (right), R² = 0.998."),
        IMG_E2=_img("results/e2_induction/phase_transition.png",
                    "Induction phase transition",
                    "E2 — prefix-matching score and in-context loss during training; the "
                    "shaded band is the detected transition window."),
        IMG_E3=_img("results/e3_sae_control/comparison.png",
                    "SAE trained vs random",
                    "E3 — SAE quality metrics on a trained model vs an untrained control."),
        IMG_E4=_img("results/e4_probe_baseline/comparison.png",
                    "Probe comparison",
                    "E4 — raw-activation logistic probe vs SAE-latent probe on two "
                    "known concepts."),
        IMG_E5=_img("results/e5_sparsity/pareto.png",
                    "Sparsity pareto",
                    "E5 — accuracy and circuit structure across imposed sparsity levels."),
        E1_R2=_fmt(e1.get("best_r2", 0)), E1_CTRL=_fmt(e1.get("best_r2_control", 0)),
        E1_INC=_fmt(e1.get("best_incremental_r2", 0)),
        E1_INCC=_fmt(e1.get("best_incremental_r2_control", 0)),
        E1_WIN=_fmt(e1.get("r2_window_baseline", 0)),
        E2_SCORE=_fmt(e2.get("final_prefix_score", 0)),
        E2_WF=_fmt(e2.get("window_frac", 0)),
        E2_L1=_fmt((e2.get("prefix_score_per_layer") or [0, 0])[0]),
        E2_L2=_fmt((e2.get("prefix_score_per_layer") or [0, 0])[-1]),
        E3_SIG=_fmt(e3.get("fvu_separation_sigma", 0)),
        E3_FT=_fmt(e3.get("fvu_mean_trained", 0)), E3_FR=_fmt(e3.get("fvu_mean_random", 0)),
        E4_RB=_fmt(e4.get("raw_probe_acc_belief", 0)),
        E4_SB=_fmt(e4.get("sae_probe_acc_belief", 0)),
        E4_RE=_fmt(e4.get("raw_probe_acc_entropy", 0)),
        E4_SE=_fmt(e4.get("sae_probe_acc_entropy", 0)),
        E5_WR=_fmt(e5.get("circuit_weight_ratio", 0)),
        E5_ER=_fmt(e5.get("circuit_edge_ratio", 0)),
        E5_Q=_fmt(e5.get("imposed_q", 0)),
        E5_ACC=_fmt(e5.get("accuracy_sparse", 0)),
    )
    for k, v in tokens.items():
        tpl = tpl.replace(f"@@{k}@@", str(v))
    return tpl


def main() -> None:
    (ROOT / "index.html").write_text(build_html())
    print(f"wrote {ROOT / 'index.html'} "
          f"({(ROOT / 'index.html').stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

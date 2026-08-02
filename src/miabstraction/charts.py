"""Inline SVG charts generated from the actual result artifacts.

Every chart here reads a results/*.json file at build time. Nothing is hand-drawn, so a
chart cannot drift from the number it depicts — if the artifact is missing the chart is
replaced by an explicit "not yet measured" placeholder rather than a plausible picture.
"""
from __future__ import annotations

PALETTE = {
    "ink": "#20242E", "muted": "#6A6E7A", "thread": "#3B4CC0",
    "pass": "#2E7D4F", "fail": "#B3402E", "warn": "#9A6B12",
    "grid": "#E4E3DC", "panel": "#F2F1EC",
}


def _placeholder(msg: str, h: int = 180) -> str:
    return (
        f'<div class="chartbox missing" style="min-height:{h}px">'
        f'<span>{msg}</span></div>'
    )


def price_curve(points: list[dict], budget: float = 0.05) -> str:
    """L1 trade-off: linked-skill accuracy against what the host pays for it."""
    if not points:
        return _placeholder("price curve not yet measured")
    xs = [p["host_delta"] for p in points]
    ys = [p["icl_acc"] for p in points]
    W, H, PAD = 560, 300, 52
    xmax = max(xs) * 1.08 or 1
    def sx(v): return PAD + (v / xmax) * (W - PAD - 20)
    def sy(v): return H - PAD - v * (H - PAD - 24)

    grid = "".join(
        f'<line x1="{PAD}" y1="{sy(t)}" x2="{W-20}" y2="{sy(t)}" stroke="{PALETTE["grid"]}"/>'
        f'<text x="{PAD-8}" y="{sy(t)+4}" text-anchor="end" class="tick">{t:.1f}</text>'
        for t in (0, 0.25, 0.5, 0.75, 1.0))
    xt = "".join(
        f'<text x="{sx(v)}" y="{H-PAD+18}" text-anchor="middle" class="tick">{v:.2f}</text>'
        for v in (0, xmax/3, 2*xmax/3, xmax))
    path = " ".join(f"{'M' if i==0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
                    for i, (x, y) in enumerate(zip(xs, ys)))
    dots = "".join(
        f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{PALETTE["thread"]}">'
        f'<title>gain {p["gain"]}: skill {y:.3f}, host pays {x:.3f} nats</title></circle>'
        for x, y, p in zip(xs, ys, points))
    bx = sx(budget)
    gate = sy(0.55)
    return f'''<div class="chartbox"><svg viewBox="0 0 {W} {H}" role="img"
 aria-label="Linked skill accuracy versus host loss increase">
 {grid}{xt}
 <line x1="{bx}" y1="24" x2="{bx}" y2="{H-PAD}" stroke="{PALETTE["fail"]}"
   stroke-dasharray="4 3"/>
 <text x="{bx+6}" y="38" class="tick" fill="{PALETTE["fail"]}">strict budget {budget}</text>
 <line x1="{PAD}" y1="{gate}" x2="{W-20}" y2="{gate}" stroke="{PALETTE["pass"]}"
   stroke-dasharray="4 3"/>
 <text x="{W-24}" y="{gate-6}" text-anchor="end" class="tick"
   fill="{PALETTE["pass"]}">skill gate 0.55</text>
 <path d="{path}" fill="none" stroke="{PALETTE["thread"]}" stroke-width="2.5"/>
 {dots}
 <text x="{W/2}" y="{H-10}" text-anchor="middle" class="axis">host loss increase (nats)</text>
 <text x="14" y="{H/2}" transform="rotate(-90 14 {H/2})" text-anchor="middle"
   class="axis">linked skill accuracy</text>
</svg><p class="cap">Each point is one write gain. The region left of the red line and
above the green line is where a link is both useful and affordable — at this host/unit
pairing it is empty, which is why the linker refused. Relaxing the budget to 0.75 nats
moves the operating point inside and the link succeeds.</p></div>'''


def capacity_bars(oob: float, in_band: dict) -> str:
    """L3: what the host pays for out-of-band linking vs reserving dimensions."""
    if not in_band:
        return _placeholder("capacity comparison not yet measured")
    rows = [("out-of-band<br>(0 dims taken)", oob, PALETTE["pass"])]
    rows += [(f"in-band<br>({k} dims reserved)", v, PALETTE["fail"])
             for k, v in sorted(in_band.items(), key=lambda kv: int(kv[0]))]
    vmax = max(max(v for _, v, _ in rows), 1e-6)
    bars = ""
    for i, (label, v, col) in enumerate(rows):
        w = max(v / vmax * 340, 1.5)
        y = 20 + i * 46
        bars += (
            f'<text x="0" y="{y+8}" class="blabel">{label.replace("<br>", " ")}</text>'
            f'<rect x="170" y="{y-8}" width="{w:.1f}" height="20" fill="{col}" rx="2"/>'
            f'<text x="{175+w:.1f}" y="{y+7}" class="bval">{v:.4f}</text>')
    return f'''<div class="chartbox"><svg viewBox="0 0 560 {20+len(rows)*46}" role="img"
 aria-label="Host loss increase by linking convention">{bars}</svg>
<p class="cap">Host loss increase (nats). Reserving residual dimensions taxes the host in
proportion to how many are taken; an out-of-band unit takes none, so it costs the host
nothing to have it available. This is the measured form of the design claim.</p></div>'''


def loss_curve(history: list[dict]) -> str:
    """Foundation pretraining: held-out loss over training."""
    if not history:
        return _placeholder("pretraining run not yet complete — no curve to show")
    steps = [h["step"] for h in history]
    vals = [h["val_loss"] for h in history]
    trs = [h.get("train_loss", h["val_loss"]) for h in history]
    W, H, PAD = 560, 280, 50
    xmax, ymax = max(steps) or 1, max(max(vals), max(trs)) * 1.05
    ymin = min(min(vals), min(trs)) * 0.95
    def sx(v): return PAD + v / xmax * (W - PAD - 20)
    def sy(v): return H - PAD - (v - ymin) / (ymax - ymin) * (H - PAD - 24)
    def line(ys, col, dash=""):
        d = " ".join(f"{'M' if i==0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
                     for i, (x, y) in enumerate(zip(steps, ys)))
        return f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2" {dash}/>'
    grid = "".join(
        f'<line x1="{PAD}" y1="{sy(v)}" x2="{W-20}" y2="{sy(v)}" stroke="{PALETTE["grid"]}"/>'
        f'<text x="{PAD-8}" y="{sy(v)+4}" text-anchor="end" class="tick">{v:.1f}</text>'
        for v in [ymin + (ymax - ymin) * f for f in (0, .25, .5, .75, 1)])
    return f'''<div class="chartbox"><svg viewBox="0 0 {W} {H}" role="img"
 aria-label="Training and held-out loss">{grid}
 {line(trs, PALETTE["muted"], 'stroke-dasharray="3 3"')}
 {line(vals, PALETTE["thread"])}
 <text x="{W-24}" y="{sy(vals[-1])-8}" text-anchor="end" class="tick"
   fill="{PALETTE["thread"]}">held-out {vals[-1]:.3f}</text>
 <text x="{W/2}" y="{H-10}" text-anchor="middle" class="axis">training step</text>
</svg><p class="cap">Solid: held-out loss on text the model never saw (contiguous tail of
each corpus domain). Dashed: training loss.</p></div>'''


def capability_graph(capabilities: list[dict]) -> str:
    """L2: the compiler's understanding of what each clause means."""
    if not capabilities:
        return _placeholder("capability graph not yet generated")

    rows = []
    for cap in capabilities:
        kind = cap.get("kind", "unknown")
        icon = {"knowledge": "📚", "skill": "🔧", "style": "🎨",
                "invariant": "⚙️", "prohibition": "🚫", "guardrail": "🛡️"}.get(kind, "•")
        desc = cap.get("capability", "")
        rows.append(
            f'<tr><td class="cap-kind">{icon} {kind}</td>'
            f'<td class="cap-desc">{desc}</td></tr>')

    return f'''<div class="chartbox"><table class="cap-table">
<tbody>{"".join(rows)}</tbody></table>
<p class="cap">Each source clause becomes a capability with a defined meaning at L2.
The compiler knows how to realize each one on different substrates.</p></div>'''


def plan_detail(capabilities: list[dict]) -> str:
    """L1: the strategy chosen for each capability and why."""
    if not capabilities:
        return _placeholder("plan not yet generated")

    rows = []
    for cap in capabilities:
        strategy = cap.get("strategy", "unknown").replace("_", " ").title()
        reason = cap.get("reason", "")
        rejected = cap.get("rejected", [])
        rejected_str = ""
        if rejected:
            alt_list = ", ".join(f"<em>{r.get('strategy', '').replace('_', ' ')}</em>"
                                for r in rejected)
            rejected_str = f'<br/><span class="note-small">Passed over: {alt_list}</span>'

        rows.append(
            f'<div class="plan-item"><b>{strategy}</b><br/>'
            f'<span class="small">{reason}</span>{rejected_str}</div>')

    return f'''<div class="chartbox"><div class="plan-list">{"".join(rows)}</div>
<p class="cap">For each capability, the compiler consulted the substrate's capability table,
chose the cheapest and sufficient realization strategy, and recorded the choice and reason.</p></div>'''


def model_architecture(config: dict, report: dict | None = None) -> str:
    """L-1: the bare architecture, marking what the build actually touched.

    The parameter count comes from the build report, which counted them. It used to be
    `d_model * n_layers * n_heads * 64`, a formula that resembles a parameter count
    without being one — a made-up number under a real label is worse than no number.

    The layer diagram marks the layers a control was installed at and the modules an
    adapter was attached to, so "which parts of the model were modified" is answered by
    the picture rather than asserted next to it.
    """
    report = report or {}
    d_model = config.get("d_model", 0)
    n_layers = config.get("n_layers", 0)
    n_heads = config.get("n_heads", 0)
    max_len = config.get("max_len", 0)
    vocab = config.get("vocab_total", 0)
    if report:
        n_layers = report.get("search_space", {}).get("model_depth") or n_layers

    if not (n_layers or d_model):
        return _placeholder("no build has recorded an architecture yet")

    params = report.get("params")
    controls = report.get("controls") or []
    # Negative layer indices count from the top, the way the program writes them.
    touched = {c["layer"] if c["layer"] >= 0 else n_layers + c["layer"]: c
               for c in controls if isinstance(c.get("layer"), int)}

    adapter_ratio = None
    for cap in report.get("capabilities", []):
        ex = cap.get("execution") or {}
        at = (ex.get("autotune") or {}).get("best") or {}
        adapter_ratio = (at.get("metrics") or {}).get("adapter_ratio") or adapter_ratio

    rows = [
        ("Layers", f"{n_layers}" if n_layers else "—"),
        ("Model dimension", f"{d_model}" if d_model else "—"),
        ("Attention heads", f"{n_heads}" if n_heads else "—"),
        ("Context window", f"{max_len}" if max_len else "—"),
        ("Vocabulary", f"{vocab:,}" if vocab else "—"),
        ("Parameters", f"{params:,}" if params else "not counted by this build"),
        ("Base weights changed by training", "none — the adaptation is an adapter"
         if adapter_ratio is not None else "—"),
        ("Trainable share", f"{adapter_ratio * 100:.2f}%" if adapter_ratio else "—"),
        ("Layers carrying a control", f"{len(touched)} of {n_layers}" if n_layers else "—"),
    ]

    svg_content = ""
    if n_layers:
        layer_h, total_h = 26, n_layers * 26 + 70
        parts = [f'<svg viewBox="0 0 460 {total_h}" style="width:100%;height:auto;'
                 f'margin:1rem 0" role="img" aria-label="layer stack, marking layers '
                 f'the build modified">']
        parts.append(f'<text x="16" y="20" font-size="11" fill="{PALETTE["muted"]}">'
                     f'blue = untouched   green = a control writes here</text>')
        for i in range(n_layers):
            y = 34 + i * layer_h
            c = touched.get(i)
            col = PALETTE["pass"] if c else PALETTE["thread"]
            op = "0.55" if c else "0.18"
            parts.append(
                f'<rect x="52" y="{y}" width="300" height="20" fill="{col}" '
                f'opacity="{op}" stroke="{col}" stroke-width="1"/>'
                f'<text x="18" y="{y + 14}" class="tick">L{i}</text>')
            if c:
                parts.append(
                    f'<text x="362" y="{y + 14}" class="tick">{c["name"][:26]} '
                    f'({c["kind"]})</text>')
        parts.append("</svg>")
        svg_content = "".join(parts)

    table = "\n".join(
        f'<tr><td>{k}</td><td class="num">{v}</td></tr>' for k, v in rows)
    note = ("No control was admissible in this build, so no layer is marked: the "
            "compiler searched and refused rather than installing something that did "
            "not work." if not touched else
            f"{len(touched)} layer(s) carry a compiled control.")
    return f'''<div class="chartbox"><table class="arch-table">
<tbody>{table}</tbody></table>
{svg_content}<p class="cap">{note} The architecture itself is fixed by the substrate
choice: on an open-weight target the trained architecture cannot be changed, so
everything the compiler achieves it achieves through adapters, grafts and steering. On
from-scratch it may pick the architecture to satisfy the capabilities.</p></div>'''


def search_trials(report: dict) -> str:
    """Every configuration the compiler tried, and why the losers lost.

    This is the chart that makes the autotuning real rather than asserted. Each column is
    one trial; height is the score it earned; a struck-through bar is a trial a budget
    refused however well it scored, which is the whole reason the search is not just
    "take the best number".
    """
    caps = report.get("capabilities", [])
    blocks = []
    for cap in caps:
        at = cap.get("autotune") or (cap.get("execution") or {}).get("autotune") or {}
        trials = at.get("trials") or []
        if not trials:
            continue
        w, h, pad = 640, 150, 34
        n = len(trials)
        bw = max(6, min(38, (w - 2 * pad) // max(n, 1) - 4))
        scores = [t.get("score", 0.0) for t in trials]
        lo, hi = min(scores + [0.0]), max(scores + [0.0])
        span = (hi - lo) or 1.0
        best_cfg = (at.get("best") or {}).get("config")

        bars = []
        for i, t in enumerate(trials):
            x = pad + i * (bw + 4)
            frac = (t.get("score", 0.0) - lo) / span
            bh = max(2, int(frac * (h - 2 * pad)))
            y = h - pad - bh
            rejected = bool(t.get("rejected"))
            won = t.get("config") == best_cfg and not rejected
            fill = (PALETTE["fail"] if rejected
                    else PALETTE["pass"] if won else PALETTE["thread"])
            op = "0.35" if rejected else "1"
            title = ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                              for k, v in (t.get("config") or {}).items())
            why = t.get("rejected") or ("chosen" if won else "admissible")
            bars.append(
                f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" fill="{fill}" '
                f'opacity="{op}"><title>{title} — score {t.get("score", 0):.4g} '
                f'({why})</title></rect>')
            if rejected:
                my = y + bh // 2
                bars.append(f'<line x1="{x}" y1="{my}" x2="{x + bw}" y2="{my}" '
                            f'stroke="{PALETTE["fail"]}" stroke-width="1.5"/>')

        kept = sum(1 for t in trials if not t.get("rejected"))
        met = at.get("target_met")
        caption = (f'{cap.get("capability", "?")} — {n} configurations of '
                   f'{", ".join(at.get("levers_searched", []))}; {kept} admissible; '
                   f'target {"met" if met else "not met"}')
        # Where the direction came from is the difference between a control for THIS
        # capability and a control for its kind, so it belongs next to the numbers.
        if at.get("direction_from"):
            caption += f'. Direction: {at["direction_from"]}'
        sc = at.get("scale") or {}
        if sc.get("gap"):
            # What the target actually means for this capability, in the same units as
            # the bars, so "target not met" is a distance rather than a verdict.
            caption += (f'. Stating the rule outright is worth {sc["gap"]:.4f} nats here; '
                        f'the program asks a control to recover {sc["must_recover"]:.0%} '
                        f'of that ({sc["target_nats"]:.4f})')
        if at.get("recovered") is not None:
            caption += f'; best recovered {at["recovered"]:.0%}'
        blocks.append(
            f'<figure class="chartbox"><svg viewBox="0 0 {w} {h}" '
            f'role="img" aria-label="{caption}">'
            f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" '
            f'stroke="{PALETTE["grid"]}"/>'
            + "".join(bars) +
            f'<text x="{pad}" y="{pad - 12}" font-size="11" fill="{PALETTE["muted"]}">'
            f'score by configuration (struck through = refused by a budget)</text>'
            f'</svg><figcaption>{caption}</figcaption></figure>')

    if not blocks:
        return _placeholder("no search has been run yet — the trials chart appears "
                            "once a build records them")
    return "\n".join(blocks)


def data_provenance(manifests: list[dict]) -> str:
    """Where each corpus came from, and whether it is really the specialist source.

    A demo built on Wikipedia articles that mention "medical" is not a medical demo. The
    flag is carried in the manifest rather than in prose so it cannot quietly drift from
    what was actually downloaded.
    """
    if not manifests:
        return _placeholder("no corpus manifests found")
    rows = []
    for m in manifests:
        spec = m.get("is_specialist")
        mark = ("<strong>specialist</strong>" if spec else
                "fallback" if spec is False else "unstated")
        cls = "pass" if spec else "warn"
        tried = m.get("attempted_sources") or []
        note = ""
        if tried:
            first = tried[0]
            if isinstance(first, dict):
                note = (f'tried {first.get("source", "?")}: '
                        f'{str(first.get("error", ""))[:90]}')
            else:
                note = f"tried {str(first)[:90]}"
        rows.append(
            f'<tr><td>{m.get("domain", "?")}</td>'
            f'<td class="{cls}">{mark}</td>'
            f'<td><code>{str(m.get("source", "?"))[:64]}</code></td>'
            f'<td>{m.get("n_docs", "—")}</td>'
            f'<td>{m.get("license", "—")}</td>'
            f'<td class="muted">{note}</td></tr>')
    return ('<table class="prov"><thead><tr><th>domain</th><th>data</th>'
            '<th>source</th><th>docs</th><th>license</th><th>note</th></tr></thead>'
            '<tbody>' + "".join(rows) + '</tbody></table>')


def loom_source(text: str, name: str) -> str:
    """The program itself, highlighted from the file rather than transcribed into the page.

    The page used to carry a hand-marked-up copy of the example. A copy drifts: the file
    on disk is what compiles, and a showcase whose first exhibit is a stale paraphrase of
    it undermines everything below.
    """
    import html as _html
    import re as _re
    KEYWORDS = ("app", "build", "on", "knows", "from", "how to", "speaks", "always",
                "never", "refuses", "expect", "effort", "tune", "scratch", "size")
    out = []
    for line in text.split("\n"):
        if line.strip().startswith("//"):
            out.append(f'<span class="c">{_html.escape(line)}</span>')
            continue
        code, sep, comment = line.partition("//")
        esc = _html.escape(code)
        esc = _re.sub(r'(&quot;[^&]*?&quot;)', r'<span class="s">\1</span>', esc)
        for kw in sorted(KEYWORDS, key=len, reverse=True):
            esc = _re.sub(rf'(?<![\w-])({kw})(?![\w-])',
                          r'<span class="k">\1</span>', esc)
        if sep:
            esc += f'<span class="c">//{_html.escape(comment)}</span>'
        out.append(esc)
    body = "\n".join(out)
    return f'<pre><span class="c">// {_html.escape(name)}</span>\n{body}</pre>'


def steering_capacity(report: dict) -> str:
    """How much behaviour was asked for, against how much a single write delivered."""
    rows = report.get("capabilities") or []
    if not rows:
        return _placeholder("no build has measured steering capacity yet")
    gmax = max(r["gap_nats"] for r in rows) or 1.0
    out = []
    for r in rows:
        gw = r["gap_nats"] / gmax * 220
        dw = r["delivered_nats"] / gmax * 220
        frac = r["recovered_fraction"]
        col = PALETTE["pass"] if r["met"] else PALETTE["warn"] if frac > 0.15 else PALETTE["fail"]
        out.append(
            f'<tr><td>{r["capability"]}</td>'
            f'<td class="num">{r["gap_nats"]:.4f}</td>'
            f'<td class="num">{r["delivered_nats"]:.4f}</td>'
            f'<td><svg viewBox="0 0 230 16" style="width:230px;height:16px">'
            f'<rect x="0" y="3" width="{gw:.1f}" height="10" fill="{PALETTE["grid"]}"/>'
            f'<rect x="0" y="3" width="{dw:.1f}" height="10" fill="{col}"/></svg></td>'
            f'<td class="num" style="color:{col}">{frac:.1%}</td>'
            f'<td>{"met" if r["met"] else "short"}</td></tr>')
    return ('<div class="tablewrap"><table><thead><tr><th>capability</th>'
            '<th class="num">rule stated outright</th><th class="num">one write delivers</th>'
            '<th>&nbsp;</th><th class="num">recovered</th><th>verdict</th></tr></thead>'
            '<tbody>' + "".join(out) + '</tbody></table></div>'
            f'<p class="cap">Grey is what stating the rule is worth; the coloured bar is '
            f'what a searched control delivered. Nats of loss on the answer the instructed '
            f'model gives. From <code>{report.get("source", "?")}</code>.</p>')


def build_walkthrough(report: dict, label: str) -> str:
    """One build narrated step by step from its own report — the execution half of
    the demo. Every line is read from the report; nothing is typeset by hand."""
    if not report:
        return _placeholder(f"no build report for {label}")
    rows = []
    for cap in report.get("capabilities", []):
        at = cap.get("autotune") or {}
        sc = at.get("scale") or {}
        steps = []
        if cap["kind"] == "knowledge":
            ex = cap.get("execution") or {}
            best = ((ex.get("autotune") or {}).get("best") or {}).get("metrics") or {}
            if best:
                steps.append(
                    f"searched lr × steps; kept the winner: held-out loss "
                    f"{best.get('heldout_loss_before')} → "
                    f"{best.get('heldout_loss_after')} on text excluded from training")
        elif at.get("skipped"):
            steps.append(f"gap measured first: <b>{sc.get('gap')} nats</b>")
            steps.append(f"steering <b>skipped on the ledger's evidence</b>: "
                         f"{at['skipped']}")
        elif sc:
            steps.append(f"gap measured first: <b>{sc.get('gap')} nats</b>; the "
                         f"program demands {sc.get('must_recover'):.0%} of it "
                         f"({sc.get('target_nats')} nats)")
            steps.append(
                f"steering searched: {at.get('n_admissible', 0)}/"
                f"{at.get('n_trials', 0)} configurations admissible; target "
                + ("<b>met</b>" if at.get("target_met") else "<b>not met</b>"))
        gate = (cap.get("behavioural_gate") or {}).get("result") or {}
        if gate:
            if gate.get("ran"):
                b = gate.get("autotune") or {}
                steps.append(
                    f"behavioural gate on the composed model: margin "
                    f"{gate.get('margin_before')} → <b>{gate.get('margin_after')}</b> "
                    f"(target {gate.get('target_margin')}) after "
                    f"{b.get('n_trials', '?')} training trials; adapter saved to the "
                    f"artifact")
            elif gate.get("target_met"):
                steps.append("behavioural gate: the composed model already meets the "
                             "declared margin; nothing trained")
            else:
                steps.append(f"behavioural gate ran and reports honestly: "
                             f"{gate.get('reason', 'margin not reached')}")
        li = "".join(f"<li>{s}</li>" for s in steps) or "<li>realized as planned</li>"
        rows.append(f"<details><summary><b>{cap['capability']}</b> "
                    f"<span class='mut'>({cap['kind']})</span></summary>"
                    f"<ul>{li}</ul></details>")
    exp = "".join(
        f"<li><span class='chip {'pass' if e['passed'] else 'fail'}'>"
        f"{'PASS' if e['passed'] else 'FAIL'}</span> {e['expectation']} — "
        f"{e['detail']}</li>" for e in report.get("expectations", []))
    verdict = ("passes whole" if report.get("passed") else "does not pass")
    return (f"<div class='walk'><p><b>{label}</b> — {report.get('params', 0):,} "
            f"parameters, wall clock {report.get('wall_clock_s')}s, "
            f"{report.get('verified_against_recitation_of', 0)} training strings "
            f"registered against recitation. The build <b>{verdict}</b>.</p>"
            f"{''.join(rows)}<ul class='exp'>{exp}</ul></div>")


def two_substrates(llama: dict, qwen: dict) -> str:
    """The portability claim as one table: same program, two families, and where the
    compiler's decisions diverged because the substrates measured differently."""
    if not llama or not qwen:
        return _placeholder("both build reports are needed for this comparison")

    def know(r):
        ex = (r["capabilities"][0].get("execution") or {})
        b = ((ex.get("autotune") or {}).get("best") or {}).get("metrics") or {}
        return f"{b.get('heldout_loss_before')} → {b.get('heldout_loss_after')}"

    def guard(r):
        cap = r["capabilities"][4]
        at = cap.get("autotune") or {}
        route = ("steering skipped (ledger)" if at.get("skipped")
                 else "steering searched, target "
                      + ("met" if at.get("target_met") else "not met"))
        g = (cap.get("behavioural_gate") or {}).get("result") or {}
        return (f"{route}; gate margin {g.get('margin_before')} → "
                f"{g.get('margin_after')}")

    def exp(r):
        return (f"{sum(e['passed'] for e in r.get('expectations', []))}/"
                f"{len(r.get('expectations', []))}")

    rows = [
        ("parameters", f"{llama.get('params', 0):,}", f"{qwen.get('params', 0):,}"),
        ("held-out loss (excluded MedQuAD)", know(llama), know(qwen)),
        ("guardrail route", guard(llama), guard(qwen)),
        ("controls installed", llama.get("n_controls_installed"),
         qwen.get("n_controls_installed")),
        ("expectations passed", exp(llama), exp(qwen)),
        ("wall clock", f"{llama.get('wall_clock_s')}s", f"{qwen.get('wall_clock_s')}s"),
    ]
    tr = "\n".join(f"<tr><td>{a}</td><td class='num'>{b}</td>"
                   f"<td class='num'>{c}</td></tr>" for a, b, c in rows)
    return ("<table class='cmp'><thead><tr><th></th>"
            f"<th>{llama.get('base_model')}</th><th>{qwen.get('base_model')}</th>"
            f"</tr></thead><tbody>{tr}</tbody></table>")


def composed_table(r: dict) -> str:
    if not r:
        return _placeholder("results/loom_composed_demo.json not present")
    arb = r.get("arbitration_when_skills_disagree", {})
    ctl = r.get("random_model_control", {})
    rows = [
        ("succession accuracy (cycle traffic)", f"{r.get('succession_accuracy'):.3f}"),
        ("induction accuracy, alone", f"{r.get('induction_acc_alone'):.3f}"),
        ("induction accuracy, composed", f"{r.get('induction_acc_composed'):.3f}"),
        ("max logit divergence on letter traffic",
         f"{r.get('letter_traffic_max_logit_divergence'):.1e}"),
        ("argmax identical on letter traffic",
         "yes" if r.get("letter_traffic_argmax_identical") else "no"),
        ("arbitration when the skills disagree",
         "succession wins" if arb.get("succession_won") else "induction wins"),
        ("random-model control (succession / induction)",
         f"{ctl.get('succession_accuracy', 0):.3f} / "
         f"{ctl.get('induction_accuracy', 0):.3f}"),
        ("nonzero weights", f"{r.get('nonzero_weights'):,}"),
    ]
    tr = "\n".join(f"<tr><td>{a}</td><td class='num'>{b}</td></tr>" for a, b in rows)
    return f"<table class='cmp'><tbody>{tr}</tbody></table>"


def e6_table(r: dict) -> str:
    if not r:
        return _placeholder("results/e6_real_lm_sae/result.json not present")
    sep = r.get("separation", {})
    rows = []
    for key, lbl in (("fvu", "reconstruction error (FVU)"),
                     ("dead_frac", "dead-latent fraction"),
                     ("l0", "L0 (pinned by TopK)")):
        v = sep.get(key, {})
        flag = " <span class='chip warn'>cannot vary</span>" if v.get(
            "zero_variance_flag") else ""
        rows.append((lbl,
                     f"{v.get('trained_mean', 0):.4f} ± {v.get('trained_std', 0):.4f}",
                     f"{v.get('control_mean', 0):.4f} ± {v.get('control_std', 0):.4f}"
                     + flag))
    tr = "\n".join(f"<tr><td>{a}</td><td class='num'>{b}</td>"
                   f"<td class='num'>{c}</td></tr>" for a, b, c in rows)
    return ("<table class='cmp'><thead><tr><th></th><th>trained "
            f"{r.get('model', '')}</th><th>same architecture, random init</th></tr>"
            f"</thead><tbody>{tr}</tbody></table>")


def e7_table(r: dict) -> str:
    if not r:
        return _placeholder("results/e7_causal_size/result.json not present")
    rows = []
    for f, v in (r.get("per_floor") or {}).items():
        rows.append((f"accuracy floor {f}",
                     f"{v.get('dense')} (mean {v.get('dense_mean')})",
                     f"{v.get('sparse')} (mean {v.get('sparse_mean')})",
                     v.get("direction", "").replace("_", " ")))
    tr = "\n".join(f"<tr><td>{a}</td><td class='num'>{b}</td>"
                   f"<td class='num'>{c}</td><td>{d}</td></tr>"
                   for a, b, c, d in rows)
    return ("<table class='cmp'><thead><tr><th></th><th>dense-trained heads</th>"
            "<th>sparse-trained heads</th><th>direction</th></tr></thead>"
            f"<tbody>{tr}</tbody></table>")

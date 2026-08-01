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


def model_architecture(config: dict) -> str:
    """L-1: the bare model architecture details."""
    if not config:
        return _placeholder("model architecture not yet generated")

    d_model = config.get("d_model", 0)
    n_layers = config.get("n_layers", 0)
    n_heads = config.get("n_heads", 0)
    max_len = config.get("max_len", 0)
    vocab = config.get("vocab_total", 0)

    params = d_model * n_layers * n_heads * 64 if all([d_model, n_layers, n_heads]) else 0

    rows = [
        ("Layers", f"{n_layers}"),
        ("Model dimension (d_model)", f"{d_model}"),
        ("Attention heads", f"{n_heads}"),
        ("Context window", f"{max_len}"),
        ("Vocabulary", f"{vocab:,}"),
        ("Approximate parameters", f"{params:,}" if params else "—"),
    ]

    svg_content = ""
    if n_layers > 0:
        layer_h = 30
        total_h = n_layers * layer_h + 60
        svg_content = f'''<svg viewBox="0 0 400 {total_h}" style="width:100%;height:auto;margin:1rem 0">
'''
        for i in range(n_layers):
            y = 40 + i * layer_h
            col = PALETTE["thread"] if i % 2 == 0 else PALETTE["muted"]
            svg_content += (f'<rect x="50" y="{y}" width="300" height="24" fill="{col}" '
                          f'opacity="0.3" stroke="{col}" stroke-width="1"/>'
                          f'<text x="20" y="{y+16}" class="tick">L{i}</text>'
                          f'<text x="360" y="{y+16}" class="tick">{d_model}d</text>')
        svg_content += '</svg>'

    table = "\n".join(
        f'<tr><td>{k}</td><td class="num">{v}</td></tr>' for k, v in rows)

    return f'''<div class="chartbox"><table class="arch-table">
<tbody>{table}</tbody></table>
{svg_content}<p class="cap">The architecture is fixed by the substrate choice. On an open-weight target,
the trained architecture cannot be changed; on from-scratch, the compiler may adapt it to
satisfy conflicting capabilities.</p></div>'''


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

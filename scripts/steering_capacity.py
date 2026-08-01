"""What one linear write at one layer can actually buy.

Derived entirely from a build report — no numbers are entered here. The compiler now
measures, per capability, how much stating the rule outright is worth (the gap) and how
much of that a searched steering control recovers. Putting those side by side answers a
question the abstraction has to answer to be honest about itself: which capabilities can
be compiled to a runtime write, and which need the weights to move.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path("results/loom_clinic_build.json")
OUT = Path("results/steering_capacity.json")


def main() -> int:
    if not SRC.exists():
        print(f"{SRC} does not exist; run scripts/build_clinic.py first", file=sys.stderr)
        return 1
    r = json.loads(SRC.read_text())
    rows = []
    for cap in r["capabilities"]:
        at = cap.get("autotune") or {}
        sc = at.get("scale") or {}
        if not sc.get("gap"):
            continue
        # A skipped search delivered nothing because it was never run; recording it as
        # a measured zero would poison the very ceiling the skip was decided by.
        if at.get("skipped") or "trials" not in at:
            continue
        admissible = [t for t in at["trials"] if not t["rejected"]]
        best = max((t["score"] for t in admissible), default=0.0)
        rows.append({
            "capability": cap["capability"],
            "kind": cap["kind"],
            "strategy": cap.get("strategy"),
            "gap_nats": sc["gap"],
            "delivered_nats": round(best, 4),
            "recovered_fraction": round(best / sc["gap"], 4),
            "target_fraction": sc["must_recover"],
            "met": bool(at.get("target_met")),
            "probes": at.get("direction_from", ""),
            "admissible_trials": len(admissible),
            "trials": len(at["trials"]),
        })
    rows.sort(key=lambda d: d["gap_nats"])

    delivered = [d["delivered_nats"] for d in rows]
    report = {
        "source": str(SRC),
        "base_model": r.get("base_model"),
        "device": r.get("device"),
        "capabilities": rows,
        "delivered_min": min(delivered) if delivered else None,
        "delivered_max": max(delivered) if delivered else None,
        "gap_min": min(d["gap_nats"] for d in rows) if rows else None,
        "gap_max": max(d["gap_nats"] for d in rows) if rows else None,
        "finding": (
            "What a searched steering control delivers is roughly constant regardless of "
            "how much behaviour is being asked for. Across these capabilities the gap "
            "spans more than an order of magnitude while the effect delivered stays "
            "within a narrow band, so the fraction recovered falls as the demand rises. "
            "A style is inside what one linear write can do; a guardrail is not."),
    }
    OUT.write_text(json.dumps(report, indent=2))

    print(f"{'gap (nats)':>11}  {'delivered':>10}  {'recovered':>10}   capability")
    for d in rows:
        print(f"{d['gap_nats']:11.4f}  {d['delivered_nats']:10.4f}  "
              f"{d['recovered_fraction']*100:9.1f}%   {d['capability'][:44]}"
              + ("  [met]" if d["met"] else ""))
    print(f"\ngap spans {report['gap_min']:.4f}–{report['gap_max']:.4f} nats "
          f"({report['gap_max']/report['gap_min']:.0f}x); "
          f"delivered spans {report['delivered_min']:.4f}–{report['delivered_max']:.4f} "
          f"({report['delivered_max']/max(report['delivered_min'], 1e-9):.1f}x)")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

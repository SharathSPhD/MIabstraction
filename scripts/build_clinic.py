"""Compile examples/clinic.loom onto Llama-3.2-1B-Instruct, for real.

This is the end-to-end run: a program someone could write without knowing what a
transformer is, compiled by searching the lever space it declared, against a corpus
published by the National Library of Medicine, and verified afterwards on material the
training never saw. Whatever it reports is what happened.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loom.app.build_open import build  # noqa: E402

TARGET = "meta-llama/Llama-3.2-1B-Instruct"
OUT = "build/Clinic-open_weight"


def main() -> int:
    # The program is fixed; the substrate is an argument. Same source, different model,
    # is the portability claim this script exists to test.
    target = sys.argv[1] if len(sys.argv) > 1 else TARGET
    slug = target.split("/")[-1].replace(".", "_")
    out = sys.argv[2] if len(sys.argv) > 2 else f"build/Clinic-{slug}"
    results_path = Path(f"results/loom_clinic_build_{slug}.json")

    report = build("examples/clinic.loom", target, out, device="cuda", verify=True)

    print("\n" + "=" * 78)
    print(f"{report['app']} on {report['base_model']}  ({report['params']:,} params)")
    print(f"device: {report['device']}   wall clock: {report['wall_clock_s']}s")

    for cap in report["capabilities"]:
        line = f"  {cap['capability']:<46} {cap['strategy'] or 'NO STRATEGY'}"
        print(line)
        ex = cap.get("execution") or {}
        if "heldout_ppl_before" in ex:
            print(f"{'':<48}held-out ppl {ex['heldout_ppl_before']} -> "
                  f"{ex['heldout_ppl_after']}  base intact: "
                  f"{ex.get('base_weights_unchanged')}")
        at = cap.get("autotune") or ex.get("autotune") or {}
        if at:
            print(f"{'':<48}{at.get('n_admissible', 0)}/{at.get('n_trials', 0)} trials "
                  f"admissible; target met: {at.get('target_met')}")

    print(f"\ncontrols installed: {report['n_controls_installed']}")
    jc = report.get("joint_calibration") or {}
    if jc:
        print(f"joint calibration: {jc.get('note', '')}")

    for c in report.get("expectations", []):
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"\n[{mark}] {c['expectation']}")
        print(f"       {c['detail']}")
        print(f"       evidence: {c['evidence'][:160]}")

    Path("results").mkdir(exist_ok=True)
    results_path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {results_path}")
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

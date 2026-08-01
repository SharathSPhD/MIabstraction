"""Config-driven experiment runner: python -m miabstraction.runner configs/e1_mess3.yaml"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys

from .config import ExperimentConfig
from .seeding import set_determinism


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main(path: str) -> dict:
    cfg = ExperimentConfig.load(path)
    set_determinism(cfg.seed)
    mod = importlib.import_module(f"miabstraction.experiments.{cfg.name}")
    result = mod.run(cfg)
    result["git_sha"] = git_sha()
    (cfg.result_dir() / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items()
                      if not isinstance(v, list)}, indent=2))
    return result


if __name__ == "__main__":
    main(sys.argv[1])

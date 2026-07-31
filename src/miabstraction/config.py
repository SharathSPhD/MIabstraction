"""Config-driven experiment definitions: YAML -> typed config objects."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    name: str
    hypothesis: str            # H1..H5 from SPEC.md
    seed: int = 0
    device: str = "cuda"
    model: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    out_dir: str = "results"

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text())
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def hash(self) -> str:
        blob = json.dumps(self.__dict__, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def result_dir(self) -> Path:
        d = Path(self.out_dir) / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d

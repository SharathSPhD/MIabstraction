"""Guard: the test suite must never write into the real results/ tree.

Toy-config test runs once overwrote genuine experiment artifacts, so VALIDATION.md
was generated from numbers that no real run produced. This pins the invariant: any
test that calls an experiment's run() must redirect out_dir to tmp_path.
"""
from __future__ import annotations

import ast
from pathlib import Path

TESTS = Path(__file__).parent


def _imports_experiment_run(tree: ast.Module) -> bool:
    """True only if `run` comes from miabstraction.experiments.* — a local helper
    named run() (as in test_seeding.py) writes nothing and must not be flagged."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if "miabstraction.experiments" in node.module:
                if any(a.name == "run" for a in node.names):
                    return True
    return False


def _tests_calling_run(tree: ast.Module) -> list[ast.FunctionDef]:
    if not _imports_experiment_run(tree):
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        calls_run = any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Name)
            and c.func.id == "run"
            for c in ast.walk(node)
        )
        if calls_run:
            out.append(node)
    return out


def _redirects_out_dir(fn: ast.FunctionDef) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and node.attr == "out_dir":
            return True
        if isinstance(node, ast.keyword) and node.arg == "out_dir":
            return True
    return False


def test_experiment_tests_redirect_output():
    offenders = []
    for path in TESTS.glob("test_*.py"):
        tree = ast.parse(path.read_text())
        for fn in _tests_calling_run(tree):
            if not _redirects_out_dir(fn):
                offenders.append(f"{path.name}::{fn.name}")
    assert not offenders, (
        "These tests execute an experiment without redirecting out_dir, so they "
        f"overwrite real results/: {offenders}"
    )

# MIabstraction

Empirical validation of the transformer abstraction-layer hypothesis (see SPEC.md — it is
the contract; docs/research*.md are the hypothesis sources).

## Rules
- Python via `.venv/bin/python` only (CUDA torch is linked from `~/.venvs/prabhasa-gb10`
  via a `.pth` file — do not reinstall torch).
- All experiments are config-driven: hyperparameters live in `configs/*.yaml`, never in code.
  Run with `.venv/bin/python -m miabstraction.runner configs/<exp>.yaml`.
- TDD: tests in `tests/` must stay green; new modules need tests first.
- Every verdict needs its controls: untrained-model control, recent-token/linear baselines,
  multi-seed variance. A metric a random model also passes proves nothing (Heap et al.).
- Results (`results/**/result.json` + pngs) are committed for provenance; checkpoints are not.
- Experiment branches: `exp/<name>` in worktrees at `../MIabstraction-<suffix>`; merge to main
  only with green tests.

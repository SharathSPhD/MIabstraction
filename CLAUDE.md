# MIabstraction

Empirical validation of the transformer abstraction-layer hypothesis (see SPEC.md — it is
the contract; docs/research*.md are the hypothesis sources), and **Loom**, the compiler
built on it (`src/loom/`, LOOM.md).

The objective everything here serves: anybody writes a high-level Loom program and gets
an LLM tuned to their objectives and data — not RAG, not fine-tuning, not agent-building,
but the language model itself made programmatically — and the app
(loom-studio-tan.vercel.app) lets them work with the model they built.

**`GATES.md` is the closure ledger.** A gate closes only on evidence a reader can
re-derive: a committed artifact, a quoted command output, or a test that fails if the
claim stops being true. "Implemented" is not a closure. A gate found untrue later is
re-opened *with the reason kept*, because the history of a wrong closure is worth more
than a clean table — B1 and E have each been closed and re-opened on their own evidence.

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
- **Refusal is never compiled into weights.** `never`/`refuses` are policy clauses: parsed,
  carried in the artifact, compiled to no weight change, enforced by `loom/app/policy.py`
  in front of an unmodified model. Every dose that made a model decline off-subject
  questions also made it decline its own. If a clause cannot be enforced on a given
  corpus, the artifact says `NOT ENFORCED` rather than implying it fires.
- **A harness must call the thing it tests.** Two bugs this week were harnesses measuring
  themselves: a sweep that recomputed an estimator inline and reported byte-identical
  results after the estimator changed, and a robustness check that rotated a list before
  taking its mean.

## Where the GPUs are
- GB10 (this box) serves the worker at `127.0.0.1:8788` — `/tmp/start_worker.sh`, killed
  with `fuser -k 8788/tcp`, not a `pkill` pattern with alternation.
- RTX 5090 at `ss@192.168.0.204` for training. The container mount is `/fusion-project`,
  **not** `/workspace/fusion-project`; `docker exec -d` with the wrong path fails silently.

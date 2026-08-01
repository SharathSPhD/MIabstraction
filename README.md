# MIabstraction

**Does the transformer have a software abstraction layer?** This repo empirically tests the
hypothesis (developed in [docs/](docs/)) that mechanistic interpretability is uncovering a
*plural, leaky abstraction stack* — weights → parameter components → features → circuits →
algorithms → behaviors — rather than a single clean "neural ISA".

Everything is **spec-driven** ([SPEC.md](SPEC.md) defines five falsifiable hypotheses H1–H5
with numeric refutation thresholds), **config-driven** (all hyperparameters in
[configs/](configs/)), and **test-driven** (pytest). Experiments run on a single NVIDIA GB10.

## Experiments and results

| Exp | Hypothesis | Validates layer | Verdict | Headline number |
|-----|-----------|-----------------|---------|-----------------|
| E1 `e1_mess3` | H1 belief-state geometry in the residual stream | representation geometry | ✅ supported | R²=0.998; incremental R² beyond a recent-token window 96× the untrained control |
| E2 `e2_induction` | H2 induction heads form as a phase transition | circuits / development | ✅ supported | score 0→0.61 in 15.5% of training, co-timed with ICL loss 3.00→0.08 nats |
| E3 `e3_sae_control` | H3 SAE metrics don't separate trained from random | features (SAE reckoning) | ❌ refuted | FVU separated at 39σ — but in the *wrong direction* (see caveat) |
| E4 `e4_probe_baseline` | H4 linear probes match SAE probes on known concepts | features vs baselines | ✅ supported | raw 0.90 vs SAE 0.84 (belief); 0.686 vs 0.629 (entropy) |
| E5 `e5_sparsity` | H5 imposed weight sparsity shrinks circuits faithfully | weights (imposed ISA) | ⚠️ undecidable as operationalized | 80% of weights removable at 99.8% accuracy and 1.0 faithfulness — but every size metric (0.2086, 0.2093) just returns the imposed q=0.200 |

Full verdicts, leak budgets, replication, and the interpretation live in
[VALIDATION.md](VALIDATION.md).

**The controls, not the headline metrics, are the point.** Every result here was one
control away from being wrong: H1 looked supported until an untrained network scored 0.887
on the same probe; H2 looked refuted until we found that fixed-offset repetition is
solvable from positional embeddings alone; H5 looked like a 4.8× win until replication
showed the ratio was just the sparsity knob. Two experiments also produced findings about
the *method* rather than the models — GPU non-determinism moved a headline number by 7
points across identical runs, and a preregistered threshold had silently drifted in code.
Both are recorded rather than quietly fixed.

## Run

```bash
uv venv .venv --python 3.12
echo "$HOME/.venvs/prabhasa-gb10/lib/python3.12/site-packages" > .venv/lib/python3.12/site-packages/_gb10_torch.pth
uv pip install -e ".[dev]" --python .venv/bin/python
.venv/bin/python -m pytest
.venv/bin/python -m miabstraction.runner configs/e1_mess3.yaml
```

Verdicts are aggregated in `VALIDATION.md`; hypothesis posteriors in `results/hypotheses.json`
are updated by the active-inference selector in `src/miabstraction/design.py`.

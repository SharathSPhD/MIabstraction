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
| E5 `e5_sparsity` | H5 imposed weight sparsity shrinks circuits faithfully | weights (imposed ISA) | ⚠️ supported, but metric is near-tautological | 80% of weights removable at 99.8% accuracy and 1.0 faithfulness — but the size ratio (0.209) just echoes the imposed q=0.2 |

Full verdicts, leak budgets, and the interpretation live in [VALIDATION.md](VALIDATION.md).
Every positive result here was one control away from being wrong — the controls, not the
headline metrics, are the point.

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

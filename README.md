# MIabstraction

**Does the transformer have a software abstraction layer?** This repo empirically tests the
hypothesis (developed in [docs/](docs/)) that mechanistic interpretability is uncovering a
*plural, leaky abstraction stack* — weights → parameter components → features → circuits →
algorithms → behaviors — rather than a single clean "neural ISA".

Everything is **spec-driven** ([SPEC.md](SPEC.md) defines five falsifiable hypotheses H1–H5
with numeric refutation thresholds), **config-driven** (all hyperparameters in
[configs/](configs/)), and **test-driven** (pytest). Experiments run on a single NVIDIA GB10.

## Experiments

| Exp | Hypothesis | Validates layer | Key control |
|-----|-----------|-----------------|-------------|
| E1 `e1_mess3` | H1 belief-state geometry in the residual stream | representation geometry | untrained model + recent-token window baseline |
| E2 `e2_induction` | H2 induction heads form as a phase transition | circuits / development | co-timing of score and ICL loss |
| E3 `e3_sae_control` | H3 SAE metrics don't separate trained from random | features (SAE reckoning) | random-transformer control (Heap et al.) |
| E4 `e4_probe_baseline` | H4 linear probes match SAE probes on known concepts | features vs baselines | raw-activation logistic regression |
| E5 `e5_sparsity` | H5 imposed weight sparsity shrinks circuits faithfully | weights (imposed ISA) | matched dense model, mean-ablation faithfulness |

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

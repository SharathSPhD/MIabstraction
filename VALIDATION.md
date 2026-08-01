# VALIDATION — abstraction-layer verdicts

Auto-generated from `results/**/result.json` by `miabstraction.report`.

| Hypothesis | Layer | Verdict | Key numbers | Leak budget | Source |
|---|---|---|---|---|---|
| H1 | Representation geometry (belief states) | ✅ supported | supports=1, final_loss=0.794, best_r2=0.998, best_r2_control=0.887, r2_window_baseline=0.967, window_k=8 | 0.00 | `results/final/e1_mess3/result.json` |
| H2 | Circuits / development (induction phase transition) | ✅ supported | supports=1, final_prefix_score=0.607, max_prefix_score=0.633, window_frac=0.155, co_timed=1, final_loss_first_copy=3 | 0.39 | `results/e2_induction/result.json` |
| H3 | Features — SAE metrics vs random control | ❌ not supported | supports=0, final_loss=0.505, fvu_mean_trained=0.0414, fvu_std_trained=0.000566, fvu_mean_random=0.0146, fvu_std_random=0.000388 | — | `results/e3_sae_control/result.json` |
| H4 | Features vs linear-probe baselines | ✅ supported | supports=1, final_loss=0.505, raw_probe_acc_belief=0.9, sae_probe_acc_belief=0.843, raw_probe_acc_entropy=0.686, sae_probe_acc_entropy=0.629 | — | `results/e4_probe_baseline/result.json` |
| H5 | Weights — imposed sparsity ISA | ✅ supported | supports=1, circuit_size_dense=6, circuit_size_sparse=6, circuit_weights_dense=1.24e+05, circuit_weights_sparse=2.59e+04, circuit_weight_ratio=0.209 | 0.00 | `results/e5_sparsity/result.json` |

## Posteriors after Bayesian update

H1-H5 are independent binary claims, so these do not sum to 1; each is that
hypothesis' own probability after one preregistered, controlled experiment.

| Hypothesis | Posterior |
|---|---|
| H1 | 0.905 |
| H2 | 0.857 |
| H3 | 0.158 |
| H4 | 0.818 |
| H5 | 0.810 |

## What these five results say about the abstraction layer

The guiding question (from `docs/research1.md`) was whether mechanistic interpretability is
uncovering a *transformer instruction set* — a single clean abstraction — or something
messier. Five preregistered experiments, each with the control that would expose a false
positive, point the same way the literature synthesis did: **the stack is real but plural,
and the layers are not equally load-bearing.**

### The layers that held

**Representation geometry is the strongest layer (H1, posterior 0.90).** A 4-layer
transformer trained on Mess3 embeds the data process's belief-state simplex in its residual
stream at R² = 0.998, and the fractal geometry is visible by eye in the linear readout. What
makes this decisive is not the R² — an *untrained* network scores 0.887, and a plain
8-token-window baseline scores 0.967, because recent tokens nearly determine the belief
state. The signal is the **incremental** R² beyond that window: 0.031 for the trained model
versus 0.0003 for the untrained control, a 96× gap. This layer is *derived* rather than
reverse-engineered: computational mechanics predicts the geometry from the data process
before anyone looks inside, which is why it is the most principled candidate in the stack.

**Circuits are real and form discontinuously (H2, posterior 0.86).** The prefix-matching
score rises from ~0 to 0.61 inside 15.5% of training, co-timed with the in-context loss
collapsing from 3.00 to 0.08 nats — a phase transition, not a ramp. It sits in layer 2 only
(0.607 vs 0.073 in layer 1), the two-layer previous-token → induction composition the
literature describes. The route here is itself the finding: our first design ([x; x] at a
fixed offset) produced *perfect* in-context copying with **zero** induction attention,
because a constant offset is solvable from positional embeddings alone. Randomizing the gap
removed the shortcut and the induction head appeared. A behavioral metric certified a
capability the mechanism did not implement — exactly the interpretability illusion
Jonas & Kording warn about, reproduced accidentally in a system whose ground truth we knew.

**Imposed weight structure buys interpretability (H5, posterior 0.81).** Training with
AbsTopK weight sparsity yields a minimal circuit **4.8× smaller in weights** (25,914 vs
124,224) at identical faithfulness (1.0) and indistinguishable task accuracy (0.998 vs
1.000) — a small-scale echo of Gao et al.'s ~16× at matched loss. This is the "the ISA is
*imposed*, not discovered" result: the clean layer appears when the training objective is
made to pay for it. Note what did *not* discriminate: **node counts were identical (6 vs 6)**.
Only the weight-level ruler saw the difference, which is a caution about circuit-size claims
reported at coarse granularity.

### The layer that leaks

**SAE features are the weak joint, but not for the expected reason (H3 refuted at 0.16,
H4 supported at 0.82).** On known concepts, the raw-activation logistic probe beat the
SAE-feature probe on both tasks (0.90 vs 0.84 belief-region; 0.686 vs 0.629 entropy) — H4
supported, consistent with Kantamneni et al. and with the "discover unknown, don't act on
known" division. But H3 was **refuted**: SAE reconstruction metrics *did* cleanly separate
trained from randomly-initialized models (FVU 0.041 vs 0.015, 39σ over 3 seeds). Heap
et al.'s non-separation did not replicate here. The honest reading is that this is a
scale/domain disanalogy rather than a rebuttal: on a 3-symbol synthetic process a random
network's activations are low-dimensional and trivially reconstructable, so *lower* FVU on
the random model reflects an easier reconstruction target, not better features. That the
separation runs in the opposite direction from "trained models are more structured" is
itself the warning — the metric is measuring reconstructability of the activation
distribution, not the presence of learned features.

### The synthesis

The stack that survives contact with evidence is **not a single ISA**. Geometry (H1) and
imposed weight structure (H5) are the load-bearing layers; circuits (H2) are real but their
formation is discontinuous and their measurement is shortcut-prone; the activation-feature
layer (H3/H4) is descriptive rather than canonical, and simple baselines match it wherever
the concept is already named. That ordering — weights and geometry over activation
features — is the same conclusion `docs/research1.md` reached from the literature, now with
measured leak budgets attached rather than asserted.

Two methodological findings generalize beyond these five toys, and both are about controls:

1. **Every positive result here was one control away from being wrong.** H1 looked
   supported at R² 0.90 until the window baseline showed a *random* network scoring 0.887.
   H2 looked refuted at score 0.026 until we found the positional shortcut. H5 looked
   refuted at 6-vs-6 nodes until the weight-level ruler was applied. In all three the naive
   metric and the true answer disagreed.
2. **Leak budgets should be reported, not eliminated.** H1 leaks 0.2% of belief variance,
   H2 leaks 39% of attention mass off the induction target while still solving the task
   perfectly. A layer can be useful and leaky at once; what makes it engineering rather than
   storytelling is that the leak is measured.

### Limits of this evidence

These are toy models (d=64, ≤4 layers) on synthetic processes, single-seed for E1/E2/E5
(E3 used 3 SAE seeds). They establish that the *methods and controls* discriminate, not that
the conclusions transfer to frontier scale. The posteriors above are deliberately
conservative for that reason: one controlled experiment moves belief, it does not settle it.
The next decisive step is re-running E3/E4 on a real LM's activations (Pythia-160M), where
the Heap et al. control has its intended meaning, and pushing E5 past the point where
sparsity starts costing accuracy.

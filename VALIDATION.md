# VALIDATION — abstraction-layer verdicts

Auto-generated from `results/**/result.json` by `miabstraction.report`.

| Hypothesis | Layer | Verdict | Key numbers | Leak budget | Replication | Source |
|---|---|---|---|---|---|---|
| H1 | Representation geometry (belief states) | ✅ supported | supports=1, final_loss=0.794, best_r2=0.998, best_r2_control=0.887, r2_window_baseline=0.967, window_k=8 | 0.00 | 2/2 seeds, best_incremental_r2=0.03188±0.00022 | `results/final/e1_mess3/result.json` |
| H2 | Circuits / development (induction phase transition) | ✅ supported | supports=1, final_prefix_score=0.607, max_prefix_score=0.633, window_frac=0.155, co_timed=1, final_loss_first_copy=3 | 0.39 | 2/2 seeds, final_prefix_score=0.7167±0.00018 | `results/e2_induction/result.json` |
| H3 | Features — SAE metrics vs random control | ❌ not supported | supports=0, final_loss=0.762, fvu_mean_trained=0.0524, fvu_std_trained=0.00166, fvu_mean_random=0.00524, fvu_std_random=0.00127 | — | single seed | `results/e3_sae_control/result.json` |
| H4 | Features vs linear-probe baselines | ✅ supported | supports=1, final_loss=0.762, raw_probe_acc_belief=0.908, sae_probe_acc_belief=0.907, raw_probe_acc_entropy=0.679, sae_probe_acc_entropy=0.72 | — | 2/2 seeds, raw_vs_sae_delta_belief=0.001359±0.0019 | `results/e4_probe_baseline/result.json` |
| H5 | Weights — imposed sparsity ISA | ⚠️ undecidable (metric tautological) | supports=1, circuit_size_dense=6, circuit_size_sparse=6, circuit_weights_dense=1.24e+05, circuit_weights_sparse=2.59e+04, circuit_weight_ratio=0.209 | 0.00 | ⚠️ 2/2 seeds, circuit_weight_ratio=0.2086 invariant (possibly tautological) | `results/e5_sparsity/result.json` |

## Posteriors after Bayesian update

H1-H5 are independent binary claims, so these do not sum to 1; each is that
hypothesis' own probability after one preregistered, controlled experiment.

| Hypothesis | Posterior |
|---|---|
| H1 | 0.905 |
| H2 | 0.857 |
| H3 | 0.158 |
| H4 | 0.818 |
| H5 | 0.500 |

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
collapsing from 3.00 to 0.08 nats — a phase transition, not a ramp. Seed 0's final score
(0.607) clears the 0.6 bar by only 1.2%, so it was replicated: seeds 1 and 2 both support
H2 at 0.7167 ± 0.0002, comfortably clear. The verdict is robust; the *headline seed* was
threshold-adjacent. It sits in layer 2 only
(0.607 vs 0.073 in layer 1), the two-layer previous-token → induction composition the
literature describes. The route here is itself the finding: our first design ([x; x] at a
fixed offset) produced *perfect* in-context copying with **zero** induction attention,
because a constant offset is solvable from positional embeddings alone. Randomizing the gap
removed the shortcut and the induction head appeared. A behavioral metric certified a
capability the mechanism did not implement — exactly the interpretability illusion
Jonas & Kording warn about, reproduced accidentally in a system whose ground truth we knew.

**Imposed weight structure is cheap here, but the headline metric is nearly a tautology
(H5, posterior 0.81 — read the caveat).** Training with AbsTopK sparsity yields a minimal
circuit 4.8× smaller in weights (25,914 vs 124,224) at identical faithfulness (1.0) and
indistinguishable accuracy (0.998 vs 1.000). H5 passes its preregistered bar. But
replication exposed the catch: the weight ratio is **0.2086 across all three seeds, with
zero variance** — and the imposed sparsity was **q = 0.20**. AbsTopK keeps exactly 20% of
weights by construction, and both models retained the same 6 nodes, so the ratio is
*forced by the mask*, not discovered in the learned structure. A metric that returns the
same value no matter what the model learned is measuring our own hyperparameter.

We then tried to repair the metric and **failed, informatively**. A magnitude-thresholded
edge count (weights above 10⁻³ of their layer's max) was added specifically because it is
not defined in terms of q. It returned a sparse/dense ratio of **0.2093** — against the
weight ratio's 0.2086 and an imposed q of **0.200**. Three "different" size metrics, one
number. The reason is structural: AbsTopK prunes every matrix uniformly, the surviving
weights are by construction the largest, and in the dense model essentially every weight
clears a 10⁻³ relative bar. Any metric that counts weights will keep recovering q.

So the operational conclusion for H5 is stronger than "unreplicated" and more specific than
"unproven": **as operationalized here, H5 is undecidable.** The experiment cannot separate
"sparsity produces simpler mechanisms" from "sparsity sets the weight count", because every
size metric we have tried is a function of the knob we turned. Deciding it requires a
measure with no weight-counting in it at all — causal edge pruning (ablate individual
connections and keep only those the task needs), or description length of the extracted
algorithm. Neither has been run. **H5's posterior of 0.81 should be read as an artifact of
a passing-but-uninformative test, not as evidence.**

What does survive, and is worth stating on its own: at 80% of weights removed this task
keeps **99.8% accuracy and 1.0 mean-ablation faithfulness**, with the node-level circuit
unchanged (6 vs 6). Sparsity is nearly free here. Whether it buys *interpretability* — Gao
et al.'s actual claim — this experiment does not establish.

The methodological lesson generalizes past this experiment: **zero variance across seeds is
a warning, not a triumph.** A result that reproduces perfectly may be reproducing a
constant.

### The layer that leaks

**SAE features are the weak joint, and H4 is the thinnest verdict here (H3 refuted at
0.16, H4 supported at 0.82).** On the belief-region concept the raw-activation probe and
the SAE probe are a dead heat (0.9076 vs 0.9071). On the harder entropy concept the **SAE
probe wins by 4.0 points** (0.720 vs 0.679). H4 survives only because the preregistered
falsification bar is "SAE beats raw by >5 pts *consistently*", and a 4-point win on one of
two concepts does not clear it. Read plainly: raw activations match SAEs where the concept
is easy and lose slightly where it is hard, so the honest claim is *parity, not raw
superiority* — weaker than Kantamneni et al.'s result, and a caution against treating a
surviving hypothesis as a won argument. Replication across seeds 1–2 puts a number on the
parity: the raw-minus-SAE gap on the belief concept is **0.0014 ± 0.0019**, statistically
indistinguishable from zero.

This verdict moved twice before settling, which is itself worth recording. The first
implementation used a 1-point rule instead of the registered 5-point rule, and the run was
non-deterministic — the same config produced SAE accuracies of 0.786, 0.843, and 0.857 on
different runs. Fixing determinism (pinning the math attention backend; the
memory-efficient kernel's backward pass is non-deterministic) and restoring the
preregistered criterion produced the numbers above, which now reproduce bit-for-bit. The
earlier reported figures were artifacts of that drift, not of the data.

H3 was **refuted**: SAE reconstruction metrics *did* cleanly separate trained from
randomly-initialized models (FVU 0.0524 vs 0.0052, 22.5σ across 5 SAE seeds). Heap et al.'s
non-separation did not replicate here. The honest reading is a domain disanalogy rather
than a rebuttal: on a 3-symbol synthetic process an untrained network's activations are
low-dimensional and trivially reconstructable, so the random model gets the *lower* FVU —
an easier reconstruction target, not better features. That the separation runs opposite to
"trained models are more structured" is the warning itself: the metric is scoring
reconstructability of the activation distribution, not the presence of learned features.

### The synthesis

The stack that survives contact with evidence is **not a single ISA**, and it is shorter
than we set out to validate. **Representation geometry (H1) is the one layer that clearly
held**: predicted in advance from the data process, decodable at R² = 0.998, replicated
across seeds, and beating every control we could construct. **Circuits (H2) are real and
form discontinuously**, also replicated — but their measurement is shortcut-prone, and we
walked into one. The **activation-feature layer (H3/H4) is descriptive rather than
canonical**: linear baselines tie with SAE probes on the easy concept and lose slightly on
the hard one, and SAE quality metrics behaved in a way that says more about activation
dimensionality than about learned features. The **weights layer (H5) is undecided** — not
because the experiment failed, but because every size metric we built turned out to encode
the sparsity knob rather than the learned mechanism.

So the ordering `docs/research1.md` argued for from the literature — weights and geometry
over activation features — is only half-confirmed here. Geometry earned its place. The
weight layer's claim to be *the* durable ISA candidate remains untested by us, and our
attempt to test it mostly demonstrated how easy it is to measure your own hyperparameter
and call it a discovery.

Three methodological findings generalize beyond these five toys:

1. **Every positive result here was one control away from being wrong.** H1 looked
   supported at R² 0.90 until the window baseline showed a *random* network scoring 0.887.
   H2 looked refuted at score 0.026 until we found the positional shortcut. H5 looked
   refuted at 6-vs-6 nodes until the weight-level ruler was applied. In all three the naive
   metric and the true answer disagreed.
2. **Leak budgets should be reported, not eliminated.** H1 leaks 0.2% of belief variance,
   H2 leaks 39% of attention mass off the induction target while still solving the task
   perfectly. A layer can be useful and leaky at once; what makes it engineering rather than
   storytelling is that the leak is measured.
3. **Non-determinism silently manufactures findings.** Before the attention backend was
   pinned, E4's headline number varied by 7 points across identical runs — enough to flip
   a verdict, and enough that any of the three values could have been reported in good
   faith. GPU non-determinism is not a rounding concern in interpretability work; it is
   large compared to the effects being claimed. Equally: the support rule in code must be
   the rule that was preregistered. E4's implementation had silently drifted to a stricter
   bar than SPEC.md's, which is how a hypothesis gets "confirmed" against a target nobody
   registered.

### Limits of this evidence

These are toy models (d=64, ≤4 layers) on synthetic processes, single-seed for E1/E2/E5
(E3 used 3 SAE seeds). They establish that the *methods and controls* discriminate, not that
the conclusions transfer to frontier scale. The posteriors above are deliberately
conservative for that reason: one controlled experiment moves belief, it does not settle it.
The next decisive step is re-running E3/E4 on a real LM's activations (Pythia-160M), where
the Heap et al. control has its intended meaning, and pushing E5 past the point where
sparsity starts costing accuracy.

## Follow-up probes (2026-08-02, unpreregistered — recorded, not scored)

Neither of these updates a posterior: they were designed after the verdicts above, so
they carry the full garden-of-forking-paths discount. They are recorded because each
answers the specific "next decisive step" its hypothesis left open.

**E6 — the SAE control on a real LM** (`results/e6_real_lm_sae/`). Qwen2.5-0.5B-Instruct
vs the same architecture at random init, three SAE seeds per arm, 90k tokens of real
text. FVU: trained 0.2314±0.0014, random 0.2356±0.0002. The toy separation direction
(random easier to reconstruct) does **not** replicate at scale; reconstruction quality
tells the two apart by under half a percent, which is the Heap et al. worry in its
intended habitat. What does separate them is dead-latent fraction (36% trained vs 0.1%
random). H3's refutation stands on its preregistered toy test; this is the footnote that
says the metric's uninformativeness is not a toy artefact.

**E7 — algorithm size by causal ablation** (`results/e7_causal_size/`). The measure H5
lacked: necessary-head count by greedy ablation against an absolute accuracy floor
shared by every model, calibrated first on the hand-compiled induction circuit (reports
exactly its 3 wired heads; declines to count an idle one). The first run used a floor
relative to each model's own accuracy and was withdrawn after adversarial audit found
r=0.938 between accuracy and count — the stopping rule, not the circuit, was being
measured. Deconfounded, three seeds, floors 0.80/0.85/0.90: dense and sparse models are
at parity at 0.80 and 0.85 (means 1.0-1.3 heads); at 0.90 the sparse models need 4.0
heads against 2.5 for the two dense models that reached that level (the third never
did — a survivorship caveat the report carries). Direction is not stable across floors,
so no strong claim is made in either direction. What the probe does establish: under a
measure that is free to vary and cannot echo the sparsity knob, **H5's hypothesized
direction — sparse training finds smaller circuits — finds no support at any
performance level**. H5's posterior stays 0.50; the sentence it needed changes from
"undecidable, the ruler was broken" to "measured with a working ruler, and the
hypothesized effect is absent at this scale".

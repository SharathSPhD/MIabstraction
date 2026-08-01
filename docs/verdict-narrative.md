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

**Imposed weight structure buys interpretability (H5, posterior 0.81).** Training with
AbsTopK weight sparsity yields a minimal circuit **4.8× smaller in weights** (25,914 vs
124,224) at identical faithfulness (1.0) and indistinguishable task accuracy (0.998 vs
1.000) — a small-scale echo of Gao et al.'s ~16× at matched loss. This is the "the ISA is
*imposed*, not discovered" result: the clean layer appears when the training objective is
made to pay for it. Note what did *not* discriminate: **node counts were identical (6 vs 6)**.
Only the weight-level ruler saw the difference, which is a caution about circuit-size claims
reported at coarse granularity.

### The layer that leaks

**SAE features are the weak joint, and H4 is the thinnest verdict here (H3 refuted at
0.16, H4 supported at 0.82).** On the belief-region concept the raw-activation probe and
the SAE probe are a dead heat (0.9076 vs 0.9071). On the harder entropy concept the **SAE
probe wins by 4.0 points** (0.720 vs 0.679). H4 survives only because the preregistered
falsification bar is "SAE beats raw by >5 pts *consistently*", and a 4-point win on one of
two concepts does not clear it. Read plainly: raw activations match SAEs where the concept
is easy and lose slightly where it is hard, so the honest claim is *parity, not raw
superiority* — weaker than Kantamneni et al.'s result, and a caution against treating a
surviving hypothesis as a won argument.

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

The stack that survives contact with evidence is **not a single ISA**. Geometry (H1) and
imposed weight structure (H5) are the load-bearing layers; circuits (H2) are real but their
formation is discontinuous and their measurement is shortcut-prone; the activation-feature
layer (H3/H4) is descriptive rather than canonical, and simple baselines match it wherever
the concept is already named. That ordering — weights and geometry over activation
features — is the same conclusion `docs/research1.md` reached from the literature, now with
measured leak budgets attached rather than asserted.

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

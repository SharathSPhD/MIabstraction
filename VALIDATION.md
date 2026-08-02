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

## Write allocation: what separate compilation needed (2026-08-02)

The ABI gave a unit private state and a declared write, and said nothing about what
happens when two units write at once. Measured, they collide, and the linker refused
the composition — an honest refusal, and the thing standing between this project and
the claim that makes an abstraction layer real.

Three allocations, same two units, same host, measured side by side
(`results/loom_link_demo.json`, `L2_composition.by_write_allocation`):

| allocation | first unit | shift | second unit | shift |
|---|---|---|---|---|
| shared (naive) | 0.4473 | −0.1406 | 0.5804 | −0.4006 |
| orthogonal | 0.4543 | −0.1336 | 0.5714 | −0.4096 |
| **exclusive** | **0.5879** | **+0.0000** | 0.5394 | −0.4416 |

Both units score 0.5879 and 0.9810 alone.

**Orthogonalizing the writes did almost nothing**, and the reason is the useful part:
behaviour is decided by an argmax, and a vector added at right angles to a write can
still change which coordinate is largest. Preserving a projection is not preserving a
prediction. The analogy to the constructed backend — where induction and succession
share one weight set with bit-identical non-interference — was wrong: what that
backend has is not orthogonal writes but **disjoint support**, the second skill's
coordinates being identically zero on the first skill's traffic.

**Exclusive allocation is that guarantee expressed for a shared output**: a later
unit writes only at positions where every earlier unit stayed silent. The first
unit's score is then *exactly* its solo score — not close, equal — and the entire
price of the overlap lands on the second, which loses 0.44.

So the honest claim is narrower and more useful than "units compose": **separate
compilation into a shared output can preserve one unit exactly, and the allocation is
priority-ordered, not symmetric.** Whether both units survive depends on whether their
firing conditions are genuinely disjoint — which is a property of the programs, not of
the linker, and is now something a program can be written to satisfy.

## Refusal removed from the compiler (2026-08-02)

The compiler used to realize `never` and `refuses` clauses by suppressing a topic
direction, amplifying a refusal direction, and escalating to a LoRA on refusal
demonstrations when the searched dose could not reach the declared margin. All of it
worked, and that was the problem. Counsel, built to recover 0.5 of what stating the
rule outright achieves, declined *"what does a motion to dismiss test?"* — a legal
question, from a legal assistant. Its behavioural gate reported `refused_in_domain:
0.0` and was not lying: its in-domain probes were declarative sentences from the
corpus, so it measured whether the model refuses to *continue a case-law passage*
(never) while real traffic is *questions about law* (increasingly refused).

Asked to resolve *the same output must refuse when off-subject and must not refuse
when in-subject*, the TRIZ engine ranked separation on **condition** (0.85) and
**time** (0.90) above space and system-level, and returned **Principle 2, Taking Out**
with **Principle 24, Intermediary**. The reading is direct: a model never trained to
refuse cannot refuse a legal question, so the failure disappears by construction
rather than by tuning.

`PROHIBITION` and `GUARDRAIL` are therefore absent from the lowering catalogue, and
~215 lines of refusal machinery are deleted. The clauses survive as **policy**:
declared by the program, carried in the artifact, evaluated by `loom/app/policy.py`
before the model is invoked. Measured live: legal questions reach the model, baking
and film questions are answered by the gate with the model not consulted at all.

Two estimators for that gate were wrong and are recorded rather than quietly
replaced. Profile cosine over word frequencies admitted a sourdough question to a
legal assistant at 0.0398 against 0.0364, because the words every English sentence
contains dominate the score. Gating on a single unfamiliar content word turned
*"what about this one"* into a refusal. What works is coverage of content words
against the vocabulary of the corpus the model was actually built on, with three
words of evidence required before the gate acts: legal questions score 100%, baking
and film 0%.

**The honest limit**: this gate is word overlap. It will pass an off-subject question
phrased in in-subject vocabulary and gate an in-subject question written in jargon
the corpus lacks. Its false-positive and false-negative rates are not measured. It is
the interface that matters here — prabodha implements recognition at the activation
level behind the same call, and until that is wired in, the gate is a placeholder
that says so.

## Linking, used by a build (2026-08-02)

`src/loom/app/linking.py` is the first build-path caller of the ABI, the linker and
the hand-compiled circuits, which had lived in `experiments/` since they were
written. On a host the compiler built inside the induction circuit's verified
envelope, the graft takes the skill from 0.048 to 0.350 **with no gradient taken
anywhere**, the unit fires on 71% of the host's traffic rather than on every token,
and the host does not pay — it gains 0.42 nats.

That last number needs a caveat the measurement itself invites: the host loss is
measured on the same repeated-context sequences the unit is good at, so "the host
gained" means "on this traffic". A host outside the envelope is refused with the
numbers that refused it, which is information about the host rather than a tooling
failure.

## The policy gate's estimator, and one idea that did not work (2026-08-03)

Refusal left the weights because no dose could satisfy both halves of a physical
contradiction: the same output must decline off-subject requests and must not decline
in-subject ones. Moving scope to an intermediary in front of an unmodified model settles
that half by construction — the model is not consulted differently, so in-subject
behaviour is preserved exactly. What remains is a narrower and purely empirical
question: can a gate tell, from a request, whether it is in the model's subject?

The shipped answer is word coverage against the corpus the model was built on, and it is
not good enough. Swept across the eight domains carrying contrast material
(`results/policy_gate_resolution.json`), **only legal and medical separate their
in-subject questions from their declared out-of-subject ones**. The rest overlap, and on
the grammar corpus — 1,422 distinct subject words — in-subject questions score as low as
0.00, so a fixed floor would have gated every question the app exists to answer. That is
the over-refusal this design was created to remove, arriving by a different route.

Two consequences were shipped. The floor is calibrated per build from the domain's own
questions rather than being the constant 0.34 fitted to one domain, and a gate whose
material cannot separate the classes **disables itself and records why**, so a declared
clause reads `NOT ENFORCED` in the artifact instead of claiming an enforcement that never
fires.

The obvious improvement was tried and failed. Coverage saturates: above a megabyte almost
any ordinary word appears in any corpus, so a sourdough question scores 0.40 against court
opinions on the strength of *how*, *make* and *need*. A likelihood ratio does not
saturate — score each word by the log ratio of its rate in this domain to its rate in
ordinary English, so a merely-common word cancels, a subject word votes strongly, and a
word belonging to some other subject votes *negatively*, which coverage cannot express.
Implemented against the 239MB general-English corpus and swept over the same eight
domains, it **separated one domain instead of two**: legal's margin widened to 0.295 and
medical, which coverage handles, broke. It was reverted rather than shipped, and is
recorded here so the next person to have the idea knows it was measured.

The negative is informative. Both estimators are bags of words over a request of five to
ten content words, and at that length the evidence is too thin for either statistic. The
component that does this properly is recognition at the activation level — reading what
the model itself makes of the request rather than counting its words — which is what
prabodha implements and what this gate's interface exists to accept.

One harness bug is worth recording with it: the sweep script computed coverage inline
instead of asking the gate, so when the estimator changed the sweep went on measuring the
old one and returned byte-identical results as though nothing had happened. A harness that
does not call the thing it is testing is testing itself.

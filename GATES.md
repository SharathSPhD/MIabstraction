# Gates — what "done" means here

The objective this file exists to protect: **anybody writes a high-level Loom program
and gets an LLM tuned to their objectives and data — not RAG, not fine-tuning, not
agent-building, but the language model itself made programmatically — and the app
lets them work with the model they built.**

A gate closes only on evidence a reader can re-derive: a committed artifact, a
command whose output is quoted, or a test that fails if the claim stops being true.
"Implemented" is not a closure. "Looks right" is not a closure. A gate that was
closed and later found untrue is re-opened with the reason recorded, because the
history of a wrong closure is more useful than a clean table.

| Gate | Promise | Evidence required | Status |
|---|---|---|---|
| **A** | Refusal is not compiled into weights | `grep -r REFUSAL_DEMOS\|_refusal_margin src/` empty; catalogue has no PROHIBITION/GUARDRAIL; a built Counsel answers a legal question | **CLOSED 2026-08-02** — 215 lines deleted; Counsel passes and answers "what does a motion to dismiss test?" through the deployed app |
| **C** | The app talks to the model you built | Deployed `/api/gpu/artifacts` lists the live library; a chat request through the public URL returns the model's words; a failed turn keeps the question | **CLOSED 2026-08-02** — 24 artifacts (was 1, a build-time snapshot); Counsel answered in 4s via loom-studio-tan.vercel.app |
| **B1** | A verified circuit is linked into a real build, not an experiment | A build report names a linked unit, its firing condition, its solved gain and the host's measured cost | **CLOSED 2026-08-02** — `loom/app/linking.py` called from `exec_scratch`; induction grafted with zero training takes the skill 0.048 → 0.350 and the host *gains* 0.42 nats; fires on 71% of traffic; 5 tests |
| **B2** | Policy is enforced by an intermediary, not by weights | A request outside the declared subject is handled without the model having been trained to refuse; in-subject answers are byte-identical to the unlinked model | **CLOSED 2026-08-02** — `loom/app/policy.py` runs in the worker's chat path; legal questions reach the model, baking and film questions are answered by the gate with the model never consulted; 7 tests |
| **D1** | Every measured field the compiler produces is visible in the app | Layer walk shows `realized`, held-out PPL, `variety_after`, controls, search space | open |
| **D2** | Scratch builds appear in the showcase | A from-scratch artifact is listed, opens, and can be chatted with from the browser | open |
| **D3** | `/use` is a place to work, not a debug console | Preselect from a build, compare against base, persistent history, export | open |
| **D4** | Steering capacity is measured widely enough to drive compiler decisions | ≥5 model families in the ledger, or the skip heuristic withdrawn | open |
| **D5** | Write allocation replicates | ≥10 unit pairs; fraction achieving exact preservation reported | open |
| **D6** | The docs say what the code does | Per-substrate description matches `build_open`/`exec_scratch`; research components marked as such | open |
| **E** | The created model is worth using | A from-scratch model with held-out perplexity in the 20s–30s that answers coherently in `/use` | open |

## Standing rules

1. **No number without a run.** A claim that cannot name the artifact it came from is
   deleted, not softened.
2. **Two clauses may not share one measurement.** Identical values for different
   declarations mean the instrument cannot tell them apart; say so in the report.
3. **The probe must resemble the traffic.** Every measurement failure found in this
   project so far — over-refusal invisible to its own guard, a gate that met its
   target while the model misbehaved, a demo memorised to loss 0.0002 and recalled
   never — was a probe drawn from a different distribution than the one the model
   meets.
4. **An adversarial review follows every phase**, run by an agent with no stake in
   the work, and its findings are folded in before the next phase begins.

# Literature Review: Interior-Slot Stagnation — Credit Assignment vs Information Transport

**Scope:** Recent (2023–2026) work on (a) deep/auxiliary supervision & credit assignment in
recurrent/iterative-compute models, (b) NCAs for algorithmic computation, (c) information-transport
limits in locally-connected recurrent systems, (d) iterative refinement where intermediate supervision
fails to help final performance.

**Question addressed:** IPPR iterated-parity, L=16, query slots 0–3. Slot 0 solves ~72–78%; interior
slots 1,2 stagnate at 40–50% (chance) despite 1000 epochs, curriculum, coordinate channels, and
train-time aux deep supervision (per-tick CE through shared readout, α=0.3; aux loss → 0.1–0.2 while
terminal interior accuracy stays at chance). Is this H_OPT (credit assignment) or H_ATTENUATION
(information-transport limit, ξ≈4–6 cells vs 16-cell grid)?

---

## A. Deep supervision / credit assignment in recurrent & iterative-compute models

### 1. When Intermediate Supervision Doesn't Help: Evidence from Recurrent CNNs
- **Authors:** Elisa Klunder, Guillaume Pourcel, Steven Abreu
- **Year:** 2026 (ICLR 2026 Workshop: Latent & Implicit Thinking; LIT)
- **URL:** https://iclr.cc/virtual/2026/10016672 · OpenReview: https://openreview.net/forum?id=Wd7AwBR2AK
- **Relevance:** **Most directly on-point.** Maze-solving with recurrent CNNs, trained to imitate classical search strategies. Intermediate supervision *consistently underperforms end-to-end learning on every generalisation test*, and performance collapses on different maze topologies. Explicit procedural guidance "may even constrain network flexibility." This is the closest published analogue to our null: intermediate states are learnable/decodable (matching our aux loss → 0.1–0.2) but the aux signal does not produce a transferable terminal solution.

### 2. Beyond Memorization: Extending Reasoning Depth with Recurrence, Memory and Test-Time Compute Scaling
- **Authors:** Ivan Rodkin, Daniil Orel, Konstantin Smirnov, Arman Bolatov, Bilal Elbouardi, Besher Hassan, Yuri Kuratov, Aydar Bulatov, Preslav Nakov, Timothy Baldwin, Artem Shelmanov, Mikhail Burtsev
- **Year:** 2025 (v1 Aug 2025; v3 May 2026; Findings of ACL 2026)
- **URL:** https://arxiv.org/abs/2508.16745
- **Relevance:** Controlled **1d cellular-automata** multi-step reasoning benchmark (disjoint train/test rules → excludes memorisation). High next-step accuracy, but accuracy **drops sharply as the number of chained intermediate steps increases**; deeper models help, and recurrence/memory/test-time compute extend effective depth but **remain bounded**. Directly frames "required reasoning steps × local propagation" as the wall — argues for H_ATTENUATION unless depth is explicitly enlarged.

### 3. Looped Transformers for Length Generalization
- **Authors:** Ying Fan, Yilun Du, Kannan Ramchandran, Kangwook Lee
- **Year:** 2024/2025 (ICLR 2025)
- **URL:** https://arxiv.org/abs/2409.15647
- **Relevance:** Positive counter-example: **loop steps are supervised to match ground-truth intermediate states** after each application of the shared block, and looped transformers with adaptive step count achieve far better length generalisation on tasks with an iterative solution (adding, **parity**). Shows aux/depth supervision *can* work when the loop step maps onto a genuine algorithmic iteration — a boundary condition our setup may violate (our per-tick states may not align with a discrete iteration of the parity algorithm).

### 4. Universal Transformers Need Memory: Depth-State Trade-offs in Adaptive Recursive Reasoning
- **Author:** Grigory Sapunov
- **Year:** 2026 (arXiv v3 May 2026)
- **URL:** https://arxiv.org/abs/2604.21999
- **Relevance:** Single-block Universal Transformer + ACT on Sudoku-Extreme: **no configuration without learned memory tokens reaches non-trivial performance**, memory and ponder depth substitute as resources. Suggests recurrent compute alone (without an explicit scratchpad / wider state) hits a depth ceiling — supports the idea that a fixed-width local update with limited bandwidth cannot carry interior-slot information.

### 5. Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer (Huginn-3.5B)
- **Authors:** Wenquan Lu et al.
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2507.02199
- **Relevance:** Probing a depth-recurrent transformer: **limited evidence of interpretable latent CoT**, probing inconsistencies across recurrent blocks, and **increasing recurrence depth yields only marginal gains** — far short of explicit step externalisation. Independent evidence that more ticks ≠ more usable computation when intermediate states aren't a coherent trajectory.

### 6. A Comprehensive Review on Deep Supervision: Theories and Applications
- **Authors:** Renjie Li et al.
- **Year:** 2022 (survey)
- **URL:** https://arxiv.org/abs/2207.02376
- **Relevance:** Background/theory of deep supervision: primary benefit is **gradient-flow / optimisation** (addresses H_OPT-style credit assignment). Useful to argue that if our aux loss trains fine yet terminal performance doesn't move, we are *outside* the regime where deep supervision's documented mechanism (better gradients) can help — i.e. a negative read on H_OPT.

---

## B. NCAs for algorithmic computation

### 7. Reasoning with Neural Cellular Automata
- **Authors:** Mayalen Etcheverry, Pietro Miotti, Aidan Sirbu, Konstantin Schürholt, Mariia Drozdova, Arna Ghosh, Blaise Agüera y Arcas, James Manyika, Blake Richards, Eyvind Niklasson
- **Year:** 2026 (Sep 2026)
- **URL:** https://arxiv.org/abs/2609.36126
- **Relevance:** Strong positive existence proof: **strictly local connectivity + asynchronous updates** solve large mazes, Sudoku, ARC-AGI-1, with OOD generalisation to larger grids/longer rollouts. Crucially, success **depends on stochastic perturbations + sample replay**, and dynamic compute modulation. Implies interior-slot failure is *not* an architectural impossibility for local NCAs, but may require the right training stochasticity/regularisation — relevant to both hypotheses (perturbations help escape the shallow credit-assignment basin).

### 8. On the Mirage of Long-Range Dependency, with an Application to Integer Multiplication
- **Author:** Zichao Wei
- **Year:** 2026 (Mar 2026)
- **URL:** https://arxiv.org/abs/2603.29069
- **Relevance:** A **321-parameter NCA** achieves perfect length generalisation (683× training range) on integer multiplication once the task is re-embedded so every step is a 3×3 local op; Transformers/RoPE/Mamba all fail. Core claim: apparent long-range dependency is often a **"mirage" produced by the computational spacetime (embedding/ordering)**, not intrinsic. Directly bears on H_ATTENUATION: if ξ≈4–6 is a real transport limit at L=16, the literature's prescription is to change the layout/embedding so the dependency is local, not to add supervision.

### 9. On the Spatiotemporal Dynamics of Generalization in Neural Networks (SEAD)
- **Author:** Zichao Wei
- **Year:** 2026 (Feb 2026)
- **URL:** https://arxiv.org/abs/2602.01651
- **Relevance:** **Extremely on-point for H_ATTENUATION.** Derives an NCA ("SEAD") from physics postulates: Locality (finite propagation speed → **light-cone**), Symmetry, Stability; iterated until convergence. Reports **perfect length generalisation on parity via light-cone propagation**, scale-invariant addition L=16→1e6, and learning Rule 110. The light-cone framing is literally our "ξ cells per tick vs L=16 grid" question; it claims the fix is respecting finite-speed propagation (input-adaptive tick count), not auxiliary losses.

### 10. A Path to Universal Neural Cellular Automata
- **Authors:** Gabriel Béna, Maxence Faldor, Dan F. M. Goodman, Antoine Cully
- **Year:** 2025 (GECCO '25 Companion)
- **URL:** https://arxiv.org/abs/2505.13058
- **Relevance:** Trains a continuous NCA toward universal computation via gradient descent — matrix multiply/transpose up to emulating an MNIST classifier inside CA state. Establishes that gradient-trained NCAs *can* learn global algorithms, but demonstrates they demand purpose-built objectives/training strategies; useful reference for what "learning a rule" requires vs. dumping an auxiliary CE at each tick.

### 11. Neural Algorithmic Reasoning Without Intermediate Supervision
- **Authors:** (NeurIPS 2023)
- **Year:** 2023
- **URL:** https://proceedings.neurips.cc/paper_files/paper/2023/file/a2370db7c99791ad5d9f3ef48ad6d464-Paper-Conference.pdf
- **Relevance:** Shows algorithmic execution can be learned **without intermediate (step-wise) supervision**, questioning the necessity of per-step trace signals. Counterpoint to the assumption that our aux loss should be necessary for correctness.

### 12. Deep Equilibrium Algorithmic Reasoning
- **Authors:** Dobrik Georgiev, JJ Wilson, Davide Buffelli, Pietro Liò
- **Year:** 2024 (NeurIPS 2024)
- **URL:** https://proceedings.neurips.cc/paper_files/paper/2024/hash/3b1675de6b49cc00084374213f8c38ae-Abstract-Conference.html
- **Relevance:** Solves reasoning tasks by finding an **equilibrium** rather than matching per-step iterations; no ground-truth step count needed. Complements #5/#7: a converged fixed point may be the right target instead of per-tick supervision, and equilibrium-finding sidesteps the shallow-halt/credit-assignment trap that per-tick losses can reinforce.

---

## C. Information transport / propagation limits (local recurrent nets & message passing)

### 13. Recurrent neural networks: vanishing and exploding gradients are not the end of the story
- **Authors:** Nicolas Zucchet et al.
- **Year:** 2024 (NeurIPS 2024)
- **URL:** https://arxiv.org/abs/2405.21064
- **Relevance:** As memory/horizon grows, parameter perturbations produce increasingly large output variations even **without** exploding gradients → gradient-based learning is intrinsically sensitive. This is a *third* failure mode beyond credit assignment (H_OPT) and transport (H_ATTENUATION): our interior signal may be present but ill-conditioned. Motivates the element-wise recurrence / careful parametrisation and SSM-style design that our gated-residual CA may or may not satisfy.

### 14. Improving the Effective Receptive Field of Message-Passing Neural Networks
- **Authors:** (2025)
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2505.23185
- **Relevance:** Message-passing/local aggregation has an **effective** receptive field far smaller than the theoretical k-hop field (**oversquashing**: distant signals compressed through narrow channels). The direct graph analogue of "ξ≈4–6 effective vs L=16 nominal". Provides the vocabulary/citations to argue that interior slots 1–2 are beyond the *effective* propagation radius even though they're nominally within the grid.

### 15. Understanding over-squashing and bottlenecks on graphs via curvature
- **Authors:** Jake Topping, Francesco Di Giovanni, Benjamin Chamberlain, Xiaowen Dong, Michael Bronstein
- **Year:** 2022 (ICLR 2022 Oral)
- **URL:** https://iclr.cc/virtual/2022/oral/6850
- **Relevance:** Foundational formalisation of over-squashing via **Ricci curvature / graph bottlenecks**: information from k-hop neighbours overwrites a bounded channel. Gives a mechanistic criterion (bottleneck curvature) that predicts *where* propagation dies — maps onto interior slots being separated from the boundary seed by a low-connectivity bottleneck in grid-space.

### 16. Communication Heterogeneity and Collective Consensus in Neural Cellular Automata
- **Authors:** (Jun 2026)
- **Year:** 2026
- **URL:** https://arxiv.org/abs/2606.21202
- **Relevance:** NCA on density-classification (a **global-agreement-from-local** task) — exactly our "each cell needs a global bit" structure. Finds that communication protocol distance slows consensus; a collectively trained rule is robust to mismatch. Relevant to whether our interior slots can ever converge to the global parity bit via local exchange, and to heterogeneity/curriculum effects.

---

## Synthesis: what the literature predicts about our null

The literature is **skewed toward H_ATTENUATION as the primary cause, with H_OPT demoted to a secondary/ill-conditioning role.**

1. **Direct precedent for the null (#1).** The ICLR 2026 recurrent-CNN result is nearly our experiment with a different task: intermediate supervision is decodable and trains, yet it *consistently underperforms* end-to-end and fails to transfer. That is strong evidence that our failure is **not** a credit-assignment deficit — if it were, the aux loss (which reaches 0.1–0.2) should have helped. The fact that aux CE converges while terminal interior accuracy stays at chance is exactly the signature H_OPT predicts to be *ruled out*.

2. **Transport/depth limits are the recurring wall (#2, #9, #14, #15).** Sharp accuracy collapse beyond a bounded number of propagated steps (1dCA benchmark #2), effective receptive field ≪ nominal (#14), and bottleneck/curvature explanations (#15) all predict that cells beyond the effective propagation radius (our ξ≈4–6) cannot receive the boundary information within T ticks. Our boundary-slot advantage (slot 0, seed-adjacent) and interior stagnation (slots 1,2) is precisely the spatial gradient these papers predict.

3. **The literature's remedy is a layout/compute change, not more supervision (#8, #9).** The "mirage" argument (#8: re-embed so each step is local) and SEAD's light-cone prescription (#9: respect finite propagation speed, run input-adaptive ticks to convergence) both say the fix is to make the dependency local or to give the system enough ticks, not to attach auxiliary losses. This is directly actionable for us: check whether T is large enough for ξ·(interior distance) to reach slots 1–2, and whether the embedding/direction of iteration puts the parity prefix within the cone.

4. **Non-gradient failure modes exist (#13).** Even with a reachable gradient, growing-memory recurrent models become output-sensitive to parameter perturbations. So if a targeted transport fix (more ticks / better embedding) still stalls, the next suspect is optimisation conditioning (#13), not classic vanishing gradients.

5. **Training stochasticity, not supervision, may be the lever (#7).** The 2026 NCA reasoning result ties OOD success to **stochastic perturbations + sample replay** and dynamic compute modulation. Combined with #4 (learned memory/scratchpad necessary) and #12/#11 (equilibrium / no-intermediate-supervision), the constructive direction is: wider/scratchpad state, input-adaptive tick count, stochastic update noise — rather than per-tick CE.

**Verdict:** current literature predicts our null is best explained by **H_ATTENUATION** (effective propagation radius ξ ≈ 4–6 < distance to interior slots), with aux deep supervision failing for the same reason (see #1): a per-tick CE can only shape states that can *see* the answer; interior cells can't, so the loss is satisfied by a locally-consistent but globally-wrong intermediate state and terminal accuracy sits at chance. H_OPT is not fully excluded, but its canonical remedy (deep supervision) has now been tried and is both theoretically (#1) and empirically (our run) predicted to fail in this regime.

**Suggested next probes (from the literature):** (i) measure the empirical receptive field / influence cone at slot 1–2 vs T (à la #14–15); (ii) sweep T well beyond the current value and test input-adaptive ticks (#9); (iii) re-embed the parity chain so the required dependency is local (#8); (iv) add update stochasticity and sample replay (#7) before adding any more supervision.

---

## D. Independent verification + 2026 additions (2026-10-02, `agy`/Gemini cross-check)

**Verification:** All cited arXiv IDs (2602.01651, 2603.29069, 2609.36126, 2606.21202,
2604.21999, 2508.16745, 2409.15647, 2505.13058, 2405.21064, 2505.23185) were confirmed
present on arXiv with matching topics by two independent routes (direct abs-page fetch +
`agy`/Gemini). The ICLR 2026 "When Intermediate Supervision Doesn't Help" abstract was
matched verbatim. **No fabrication.** Mismatches: none.

**Additional 2026 papers surfaced:**
- **arXiv:2609.18966** (Howe) — *The Automaton Underneath: The Additive Input Pathway Is a
  Parasitic Attractor for State Tracking in Householder Linear RNNs*: intermediate/additive
  pathways act as **parasitic optimization attractors** on streaming parity; gradient descent
  fails to find the exact discrete automaton **despite zero auxiliary error**. Closest mechanistic
  analogue to our aux-CE-converges-while-terminal-fails signature.
- **arXiv:2609.39604** — *Why Do Conventional World Models Fail to Learn Cellular Automata?*:
  high per-step accuracy does not prevent rollout collapse without strict temporal locality and
  causal state freezing.
- **arXiv:2608.02050** — *TextNCA: Neural Cellular Automata for Language Modeling via
  Hierarchical Local Attention*: 1D causal windowed NCAs with shared recurrence; hierarchical
  receptive-field schedules beat raw iteration count.
- **arXiv:2609.36126** (Etcheverry et al.) uses **all-step per-tick supervision** to stabilize
  trajectories — intermediate loss prevents divergence but heavily penalizes transient states.

**What the literature licenses about our null (mechanistic vs analogy):**
- LICENSES: (a) light-cone sufficiency ≠ dynamic reach — at r=1, T=16 the causal diameter
  covers L, so the failure is *not* geometric disconnection; (b) low per-tick aux CE proves only
  that local step-to-step transitions match target marginals in expectation, NOT that 16 composed
  transitions preserve the parity state over distance; (c) aux per-tick supervision can bias toward
  short-horizon/greedy representations (parasitic attractor).
- DOES NOT LICENSE: (a) that a radius-1 CA *cannot* compute parity at L=16 (constructive
  counterexamples exist: 2602.01651, 2603.29069); (b) that per-tick supervision is universally
  harmful. Only this parameterization + init + loss landscape failed to reach the solution.
- **Analogy, not mechanism:** physical metaphors ("domain-wall pinning", "dissipation",
  "symmetry breaking") are unlicensed unless energy/Lyapunov/spectral quantities are actually
  measured. Keep claims at the level of causal-graph diameter, probe decay, and Jacobian magnitude.

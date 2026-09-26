# Titan Text: Research Epistemology & Scientific Standards

This document defines the strict epistemic distinctions and experimental criteria that govern all research, analysis, and agent deliberation in the Titan Text campaign.

Agents and researchers must **never** collapse these distinct concepts into one another.

---

## 1. The Epistemic Taxonomy: Core Distinctions

| Concept | Precise Operational Definition | Required Empirical Evidence | What It Does NOT Prove |
| :--- | :--- | :--- | :--- |
| **Interesting Dynamics** | Non-trivial time evolution of field states (e.g. oscillations, non-zero derivatives $\Delta x \ne 0$). | Phase-space trajectory logs, Lyapunov exponent estimates, norm trajectories. | Does not prove task relevance, decodability, or computation. |
| **Representation** | Hidden state $x_t$ contains geometric structures that vary with input tokens. | Clustering in activation space, PCA/t-SNE separation, cosine distance matrices. | Does not prove the downstream readout uses this geometry. |
| **Decodability** | An external probe (Linear/MLP) can predict ground truth from $x_t$. | Trained probe with Cross-Validation accuracy above chance floor on held-out seeds. | Does not prove causal utility; the probe may extract passive correlation unused by the model. |
| **Correlation** | Statistical co-occurrence between latent state features and target labels. | Mutual information, Pearson/Spearman $r$, linear probe alignment. | Does not establish causal mechanism ($A \to B$ vs $C \to A, B$). |
| **Information Storage** | Latent field preserves past information across $t$ ticks without degradation. | Mutual information $I(x_t; x_0) > 0$; linear decodability of prefix at $t=T$. | Does not prove transport across space or iterative transformation. |
| **Information Transport** | Information moves from lattice cell $i$ to distant cell $j$ across latent ticks. | Directional transfer entropy, causal stencil reachability, cell-separated probing. | Does not prove algorithmic transformation (could be passive diffusion or wave reflection). |
| **Iterative Transformation** | Latent state undergoes successive step-by-step mathematical refinements $\Delta x_t = f(x_t)$ that increase target prediction probability over time. | Monotonic decrease in target task cross-entropy across unroll steps $t \in [1..T]$ on identical static inputs. | Does not prove instance-specific computation; could be generic settling into an input-independent attractor. |
| **Recurrence** | Mathematical execution of state-to-state feedback: $x_{t+1} = x_t + \Delta x_t$. | Architecture definition; unrolling verification. | Does not prove the recurrence is causally necessary for task performance. |
| **Causal Dependence on Recurrence** | Disrupting latent recurrence degrades model output. | State lesions (`--lesion-state`), Zero-tick ablations ($\tau=0$), gain inversions ($\Delta x \to -\Delta x$). | Does not prove instance-specific processing; recurrence could merely provide a static uncalibrated bias or gain boost. |
| **Instance-Specific Causal Dependence** | The recurrent updates are functionally tailored to the specific input sequence, not a generic drift. | Cross-batch state shuffling (`--lesion-shuffle` or carry swap): model accuracy collapses when donor state is swapped in. Identity gap $G_{\text{identity}} > 0$ with $p < 0.01$. | Does not prove unbounded algorithmic generalization or pushdown automaton equivalence. |
| **Algorithmic Generalization** | Model correctly computes the underlying rule on inputs strictly out-of-distribution (e.g. sequence length $L > L_{\text{train}}$, depth $D > D_{\text{train}}$). | Out-of-distribution evaluation ($2\times, 4\times, 8\times$ train length) with performance sustained above Markov / chance floors. | Does not prove unbounded Chomsky Type-2 recognition on finite hardware. |
| **Latent Computation** | Coordinated execution of storage, transport, and iterative transformation causally required to produce the correct target output. | Double dissociation: lesions to specific channels destroy non-local performance while sparing local bigrams; counterfactual transplants transfer output identity. | Does not imply human-like "reasoning". |
| **Autonomous Dynamical Behavior** | Field exhibits coherent self-organizing attractors, soliton-like wave packets, or fixed points under free unrolling. | Spectral radius $\rho \le 1.0$, contractive fixed-point TOST equivalence ($\Delta \|H\| \le \epsilon$), invariant energy profiles. | Does not guarantee algorithmic capability. |

---

## 2. Forbidden Leaps & Anti-Patterns

1. **"The model understands / reasons"**:
   - **Forbidden**. Neural cellular automata perform continuous/discrete tensor updates on 1D lattices. Use precise descriptions: *"The model resolves non-local bracket matching via discrete carry registers."*
2. **"Emergent Attractor Manifold"**:
   - **Forbidden without measurement**. Prohibited unless supported by contractive fixed-point tests, Jacobian eigenvalues, or TOST equivalence tests ($\Delta \|H\| \le \epsilon$).
3. **"Infinite Stack / Chomsky Type-2 Equivalence"**:
   - **Forbidden on finite grids**. Any machine on a finite 1D lattice with finite channels and finite precision is mathematically a Chomsky Type-3 Finite State Automaton. Claims must be strictly formulated as *Bounded DPDA Emulation up to capacity $C_{\text{stack}}$*.
4. **"Baselines collapse to 0.0%"**:
   - **Forbidden without verification of trained checkpoints**. Always audit training curves and weight manifests to ensure baselines are properly converged and initialized.
5. **"Lesion has no effect (Aggregate Averaging Confound)"**:
   - **Forbidden without depth stratification**. Never compute an unweighted aggregate metric across sequence distributions where shallow cases ($D=1$) dominate. Always stratify by difficulty/depth.

---

## 3. The 15-Step Empirical Research Loop

The scientific campaign strictly follows this lifecycle:

```
OBSERVATION
  ↓
COMPETING HYPOTHESES (Conservative vs Novel)
  ↓
RECURSIVE SELF-QUESTIONING (13-Step Interrogation)
  ↓
DISCRIMINATING EXPERIMENT DESIGN (Multi-Branch Separation)
  ↓
INSTRUMENTATION (Probes, Masks, Diagnostics)
  ↓
IMPLEMENTATION (Rust kernel / PyTorch harness)
  ↓
DETERMINISTIC TEST (Unit tests & CLI verification)
  ↓
REAL RUN (Multi-seed evaluation: N >= 3 seeds)
  ↓
ARTIFACT INSPECTION (Logit temperature, RNG seeding, masking)
  ↓
STATISTICAL ANALYSIS (Paired Cohen's d, TOST equivalence, p-values)
  ↓
CAUSAL INTERVENTION (Lesion, transplant, roll, conjugate)
  ↓
ADVERSARIAL REVIEW (Skeptical agent / Falsification arbiter)
  ↓
IDEATION CYCLE (Novel tasks, synthetic worlds, counterfactuals)
  ↓
UPDATED CANONICAL RESEARCH STATE (.agents/state/research_packet.json)
  ↓
NEXT DISCRIMINATING EXPERIMENT
```

---

## 4. Persistent Research Memory

- Every empirical finding must be anchored by git commits, run manifests, and checkpoint weights.
- Historical failures, flawed baselines, and debunked artifacts must remain explicitly recorded in the `FALSIFIED / SUPERSEDED` ledger.
- The ultimate goal is not to prove that Titan Text works.
- **The goal is to find out WHAT IT ACTUALLY DOES, discover capabilities or dynamics we did not anticipate, aggressively falsify attractive explanations, and deliberately create conditions under which stronger forms of useful recurrent latent computation can emerge.**

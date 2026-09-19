=== [ADVERSARIAL-REVIEWER] (Model: deepseek/deepseek-v4.1-flash, Time: 10.78s) ===
# Adversarial Audit: The Composition Barrier Is Not a Transport Problem

## 0. Executive Deflation

Both tracks failed, and they failed *identically*. The temptation is to narrate this as "two hard problems, two failures." The correct reading is harsher: **you ran two different tasks and got the same wall, which means the wall is not task-specific — it is architectural.** The invariant (Z₂ vs ℝ), the coupling (Walsh vs Lipschitz), the coarsening (s=2 vs s=1), and the macro-potential (bistable vs running-sum) are all *decorations*. The substrate underneath them is a 1D radius-1 continuous NCA, and that substrate has a hard information-theoretic ceiling that none of your interventions touched.

The most damning number in the entire report is not 45.62% or 17.50%. It is **33%**. Your local-only baseline beats both Track 2 arms by ~2×. You did not build a long-range transport mechanism that underperformed. You built a mechanism that is *worse than not having one*.

---

## 1. The Mundane Explanations You Have Not Ruled Out

Before accepting any "emergent dynamical" framing, eliminate the boring ones. Several are live.

### 1.1 Feedforward depth specialization masquerading as recurrence
Your NCA has τ=16 developmental steps. A radius-1 update applied 16 times has a *receptive field* of 17 cells. L=16. So the receptive field is nominally sufficient — but only if the update rule is *not* contractive and *not* saturated. Check: what is the effective Lipschitz constant of the composed 16-step map? If it is <1, the receptive field is a fiction; information from slot 0 is exponentially attenuated by the time it reaches slot 15. **Falsification experiment:** inject a delta perturbation at slot 0 at t=0, measure the magnitude of the response at slot 15 at t=16. If it is below numerical noise, you have a contraction, not a transport failure. This is a one-afternoon experiment and it should have been in the report.

### 1.2 Temperature / saturation collapse
Bistable drift λw(1−w²) with λ large enough to establish ±1 attractors will *also* saturate the state. Once w ∈ {−1, +1} to machine precision, the gradient ∂w/∂(neighbor) ≈ 0. The system has committed to a fixed point and can no longer propagate anything. Your "successful" establishment of ±1 attractors in Track 1 is not a feature — it is the *cause* of the interior-slot chance performance. Slot 0 works because it is read out before saturation propagates. **Falsification:** sweep λ and report slot-1 accuracy vs λ. If accuracy is non-monotonic with a peak at small λ, you have a saturation artifact, not a composition barrier.

### 1.3 Right-copy / identity heuristics
Track 1 slot 0 = 68.75%, slot 3 = 57.50%, interior = chance. This is the signature of a *boundary-anchored* readout: the model learns to copy the nearest boundary token or a trivial function of it. Slot 0 is adjacent to the input boundary; slot 3 is adjacent to the output boundary. Interior slots have no boundary anchor. **Falsification:** shuffle the input tokens spatially and re-measure per-slot accuracy. If slot 0 accuracy collapses, it was a boundary-copy heuristic, not parity computation.

### 1.4 The "paired gap" is noise dressed as signal
Cohen's d = +0.94 on n=5 seeds is *not* strong evidence. With 5 seeds, the 95% CI on d is roughly [−0.3, +2.2]. You are one seed away from a null result. The +1.41% gap on a 10-class task is 0.14 classes. This is not a mechanism; it is a rounding error with a confidence interval.

**Verdict on §1:** Until you run the contraction test, the λ-sweep, and the shuffle test, you cannot claim "composition barrier." You can only claim "we did not check whether our system is even capable of propagating a signal."

---

## 2. Why Continuous Lipschitz Did Not Unlock Composition

The report frames the Track 1 → Track 2 switch as "discrete parity → continuous running sum." This framing is wrong and it is hiding the real problem.

**Parity is not a hard composition problem. Running sum is not a hard composition problem. Both are trivially composable in a *sequential* architecture.** The reason both fail in your NCA is that neither is being computed sequentially. They are being computed *simultaneously* by a fixed-point iteration, and fixed-point iterations do not compose functions — they relax to a fixed point.

Formally: a radius-1 NCA computes w* = lim_{t→∞} F^t(w₀, x). This is a *relaxation*, not a *fold*. The composition z_{i+1} = T(z_i, x_{i+1}) requires a *directed* dependency: slot i+1's output depends on slot i's output, which depends on slot i−1's output, etc. A relaxation has no direction. Every cell sees every other cell through the same isotropic kernel, and the fixed point is a *joint* function of all inputs, not a *chained* function.

This is why changing the invariant did nothing. The invariant lives in the *output* space. The barrier lives in the *dependency structure* of the update. You changed the paint; the engine is the same.

**The correct name for the barrier is not "transport" and not "stateful sequential composition." It is "absence of a directed dependency graph."** A 1D radius-1 NCA has an undirected dependency graph (each cell depends on its two neighbors, symmetrically). Sequential composition requires a DAG. You cannot get a DAG from a symmetric kernel by iterating it — iteration of a symmetric operator gives you a symmetric operator's fixed point.

---

## 3. The Recurrence-Depth Confusion

The report asks: "why does developmental recurrence (τ=16) fail to implement sequential token-by-token recurrence across L=16 spatial cells?"

Because **developmental recurrence and sequential recurrence are different things and you have been conflating them.**

- Developmental recurrence: apply the same map F repeatedly to a *fixed* input. This is a relaxation. It computes a fixed point. It has no notion of "token order."
- Sequential recurrence: apply a map T to a *stream* of inputs, carrying state. This is a fold. It has a strict token order.

A GRU is a sequential recurrence. Your NCA is a developmental recurrence. They are not the same object, and no amount of τ will convert one into the other. Increasing τ from 16 to 160 will give you a better-converged fixed point of the same symmetric operator. It will not give you a fold.

This directly answers the user's original question about Titan Image/Audio memory: **those architectures use sequential recurrence (or attention, which is a soft fold) over a token stream. Your NCA uses developmental recurrence over a spatial field. The "memory" in Titan is a state carried across time steps of a sequence. Your NCA has no sequence. It has a field.** You cannot take "memory" from Titan without taking the sequence structure that makes memory meaningful.

---

## 4. The Definitive Verdict

**Track 1 and Track 2 did not fail because the tasks are hard. They failed because the architecture cannot represent the dependency structure the tasks require.**

The evidence:
1. Interior slots at chance in both tracks, across two different invariants, two different couplings, two different coarsenings.
2. Local-only baseline (33%) beats both Track 2 arms (17.5%, 16.1%). The "long-range" mechanism is actively harmful.
3. Boundary slots outperform interior slots in both tracks — the signature of boundary-anchored heuristics, not composition.
4. The paired gap (d=0.94, n=5) is within noise.

**The composition barrier is real, but it is not a dynamical-systems mystery. It is a graph-theoretic fact: a symmetric radius-1 kernel iterated to a fixed point cannot produce a directed dependency chain.** No invariant change, no coarsening scheme, no macro-potential will fix this. You are trying to compute a fold with a relaxation.

---

## 5. Falsification Experiments (Run These Before Any Further Architecture Work)

1. **Contraction test.** Delta perturbation at slot 0, measure response at slot 15 after τ steps. If attenuated below 1e-3, you have contraction, not transport failure.
2. **λ-sweep.** Slot-1 accuracy vs bistable λ. Non-monotonic ⇒ saturation artifact.
3. **Spatial shuffle.** Shuffle input tokens, re-measure per-slot accuracy. Slot-0 collapse ⇒ boundary-copy heuristic.
4. **Seed expansion.** n=5 → n=20 on Track 2. If paired gap CI crosses zero, the "advantage" was noise.
5. **Directed-kernel ablation.** Replace symmetric radius-1 kernel with a *directed* radius-1 kernel (cell i depends only on i−1). If interior slots jump to >70%, the barrier was the symmetry, not the task.
6. **Sequential baseline.** Run a 1-layer GRU with the same parameter count on the same tasks. If it hits >90%, the NCA is the problem, not the task.

Experiment 5 is the decisive one. If a directed kernel solves it, you have proven the barrier is graph-theoretic and you have a clear architectural path. If a directed kernel *also* fails, the problem is deeper (likely optimization or readout) and you should stop claiming "emergent dynamics" until you have isolated it.

---

## 6. Roadmap (Conditional on Experiment 5)

If directed kernel works: the path to genuine emergent cellular synthesis is **directed NCAs** — asymmetric kernels that induce a DAG over cells. This is not "weak mimicry" of a GRU; it is a spatial fold, which is a legitimate and distinct primitive. The GRU analogy is: a directed NCA is to a GRU what a convolution is to an MLP — same operation, different weight-sharing structure.

If directed kernel fails: stop. The problem is not the kernel. Audit the readout, the loss, and the optimizer before touching the architecture again.

**Do not run another 300-epoch campaign until Experiment 5 is done.** You have spent two tracks and 300 epochs confirming that a symmetric relaxation cannot compute a fold. That is a fact you could have established in one afternoon with a perturbation test.

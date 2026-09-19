=== [ADVERSARIAL-REVIEWER] (Model: deepseek/deepseek-v4.1-flash, Time: 279.08s) ===
# Audit of Experiment 5: The DAG Stencil Did Not Do What the Council Thinks It Did

I'll be blunt: the framing of this result as "the causal stencil solved the relaxation problem" is doing a lot of rhetorical work that the numbers do not support. Let me separate what is actually demonstrated from what is being narrated.

---

## 1. What the data actually shows (stripped of narrative)

Look at the Parity slot table again:

| Slot | Cell | Epoch 100 | Epoch 200 | Epoch 300 |
|---|---|---|---|---|
| 0 | 3 | 85.6% | 83.1% | 80.6% |
| 1 | 7 | 50.0% | 52.9% | 50.8% |
| 2 | 11 | 50.2% | 47.2% | 48.5% |
| 3 | 15 | 56.2% | 51.8% | 50.8% |

**Slot 0 is decaying monotonically** (85.6 → 83.1 → 80.6). Slots 1–3 are at chance and *also* drifting toward chance. This is not a model that "solved Slot 0 and failed to propagate." This is a model that is **overfitting a local shortcut at Slot 0 and slowly losing even that**. The intact accuracy is also decaying (60.5 → 58.8 → 57.7). The training is not converging to a sequential-transport solution; it is converging to a *degenerate local solution* and then drifting.

The "G_identity doubled" claim is also suspect. G_identity = intact − batch_shuffled. If intact is 60% and shuffled is 50%, G_identity = +10%. But **Slot 0 alone at 85.6% with three slots at chance gives an expected intact of (85.6 + 50 + 50 + 56.2)/4 ≈ 60.5%** — exactly what is observed. So G_identity is *entirely* explained by Slot 0. It is not evidence of "instance-specific sequential state transport." It is evidence that **one cell learned a local function of its own input window**.

This is the single most important falsification: **G_identity is not a measure of recurrence. It is a measure of "does the model use the input at all."** A feedforward MLP with a 3-cell receptive field at cell 3 would produce the same G_identity.

---

## 2. Falsifying the "DAG solved relaxation" claim

The Council's theorem says: causal stencil ⇒ lower-bidiagonal Jacobian ⇒ no 2-cycles ⇒ no relaxation. True as linear algebra. But the theorem is about the *linearized* dynamics. The actual update is:

```
x_{t+1} = x_t + α · tanh(W_δ h1) · σ(W_g h1)
```

with `h1 = tanh(W_1 [x, left, diff])`. This is a **gated residual with a saturating nonlinearity**. The Jacobian is lower-bidiagonal *only if the gates and deltas are evaluated at the current point*. In practice:

- If `gate → 0` for cells i > 3, the state at those cells is frozen at initialization. **The DAG is causal, but the signal is gated off.**
- If `delta → 0`, same effect.
- The `diff = x − left` feature means cell i's update depends on `x_i − x_{i−1}`. If the model learns to make `x_i ≈ x_{i−1}` (a fixed point of the diff feature), then `diff ≈ 0` and the update is quenched.

**The DAG eliminated backward propagation, but it did not eliminate forward quenching.** A causal stencil with a gated residual can still fail to transport information forward if the gate saturates. The Council's theorem addressed the wrong failure mode.

**Falsification experiment 1:** Log the per-cell, per-tick gate values `σ(W_g h1)` and delta magnitudes `|tanh(W_δ h1)|` at epoch 100 and 300. Prediction: gates at cells 4–15 collapse toward 0 (or toward a constant that makes `x_{t+1} ≈ x_t`). If so, the "DAG" is causally correct but dynamically inert downstream of cell 3. This is a **gate collapse**, not a relaxation problem.

---

## 3. The "effective depth 256" hypothesis is almost certainly wrong

The Council asks whether 16 cells × 16 ticks = 256 unrolled steps causes gradient vanishing. Let's check the arithmetic. The path from cell 0 to cell 15 at tick 16 has length 15 (spatial) + 16 (temporal) = 31, not 256. The 256 figure is the *total* unroll, but gradients to Slot 3 flow along a *single* causal path of length ~31, not through all 256 nodes. Backprop through 31 gated residual steps is not a vanishing-gradient catastrophe — residual connections were invented precisely to make this tractable.

Moreover, if depth were the bottleneck, **Slot 3 (cell 15, farthest) should be worst and Slot 0 (cell 3, nearest) best** — which is observed. But the *magnitude* of the gap is the tell: Slot 0 at 85.6% and Slot 1 at 50.0% is not a gradual decay. It is a **cliff**. A depth/vanishing story predicts a smooth gradient of degradation. A cliff at the first query boundary predicts a **structural** failure: the model learned a function on cells {0,1,2,3} and nothing downstream.

**Falsification experiment 2:** Probe intermediate cells (4, 5, 6) with a linear readout trained post-hoc on the frozen latent state. If cells 4–6 carry *any* linearly decodable parity information, the transport is happening but the readout is failing. If they carry none, the transport itself is dead. This cleanly separates "readout bottleneck" from "transport bottleneck."

---

## 4. The loss-gradient masking hypothesis is the strongest candidate — and it's testable

The loss is CrossEntropy summed over cells {3, 7, 11, 15} at tick T=16. Consider the gradient flow:

- Slot 0's gradient flows through ~3 spatial steps × 16 ticks.
- Slot 3's gradient flows through ~15 spatial steps × 16 ticks.

But more importantly: **Slot 0's loss can be minimized by a purely local function of the input window at cell 3.** The model has no incentive to build a transport mechanism, because the easiest descent direction is "learn a local classifier at cell 3." Once that local classifier is learned, the parameters `W_1, W_δ, W_g` are shaped to serve cell 3's local computation. Cells 4–15 inherit those same shared parameters, but the shared parameters are now specialized to a local task.

This is **not** gradient masking in the classical sense (Slot 0's gradient dominating Slot 3's). It is **parameter sharing under a loss that admits a local solution**. The shared transition kernel is pulled toward the local optimum, and the sequential solution is never discovered because it requires the kernel to do something *different* at cell 3 than at cells 4–15.

**Falsification experiment 3 (the decisive one):** Train with the loss applied **only at Slot 3** (cell 15), with Slots 0–2 unsupervised. If the model can learn Slot 3 alone, the transport mechanism exists and the multi-slot loss was the culprit. If Slot 3 alone also fails, the transport mechanism is architecturally absent. This is a clean, cheap, decisive test.

**Falsification experiment 4:** Train with **per-cell independent transition kernels** (no weight sharing across cells). If Slot 0 still learns and Slots 1–3 still fail, the problem is not parameter sharing. If all slots learn, the problem was shared-kernel specialization.

---

## 5. The "curriculum" hypothesis is a red herring

The Council notes that on L=8 both slots learned, but on L=16 only Slot 0. This is being read as "length curriculum needed." But there's a simpler explanation: **on L=8, the query cells are at 3 and 7. Cell 7 is the last cell, so its "transport" is only 4 steps from cell 3.** On L=16, Slot 1 is at cell 7 (4 steps from cell 3) and Slot 2 at cell 11 (8 steps). If the model can do 4 steps but not 8, the L=8 result is consistent with a **short-range transport limit**, not a curriculum effect.

Wait — but on L=16, Slot 1 (cell 7, 4 steps from cell 3) is at *chance*, while on L=8 the same cell 7 was learned. That's the real anomaly. Same cell index, same distance, different result. The difference is **sequence length and the presence of cells 8–15**. This points to **interference from the longer sequence**: the shared kernel is being pulled by the need to handle cells 8–15, and this degrades the 4-step transport that worked at L=8.

**Falsification experiment 5:** Train on L=16 but with the loss only at cells {3, 7} (matching the L=8 query set). If Slot 1 recovers to L=8 levels, the failure is interference from the longer-range slots. If Slot 1 still fails, the failure is length-dependent in the transition kernel itself.

---

## 6. The readout is probably not the bottleneck, but it's cheap to rule out

The readout is a per-cell linear projection from C=16 to |V|. This is standard and should be sufficient if the latent carries the information. The fact that Slot 0 reads out at 85.6% proves the readout *can* work. The question is whether cells 4–15 carry decodable information. Experiment 2 above settles this.

One subtlety: the readout is applied at tick T=16. If the latent at cells 4–15 is still evolving (not converged) at tick 16, the readout sees a transient. But with a gated residual and 16 ticks, convergence should be fast. **Log the per-cell latent norm across ticks.** If cells 4–15 are still moving at tick 16, extend T. If they're frozen at initialization, the gate collapsed (Experiment 1).

---

## 7. What is actually going on — my best reconstruction

Putting it together, the most parsimonious explanation is:

1. **The DAG stencil did eliminate backward relaxation.** This is real and the G_identity jump on Sum (+10.3% vs +1.1%) is genuine evidence that the symmetric stencil was destroying instance-specific information. Credit where due.

2. **But the model found a local shortcut at Slot 0** that satisfies a large fraction of the loss with minimal transport. The shared transition kernel specialized to this local computation.

3. **Downstream cells are either gated off or carry a latent that is a fixed function of the local input window**, not a transported state. The cliff at Slot 1 is the signature of a kernel that never learned to *use* its left-neighbor input for anything other than the local diff feature.

4. **The 256-step depth framing is a distraction.** The relevant path length is ~31, and residual connections make that tractable. The failure is structural (kernel specialization + gate collapse), not a vanishing-gradient depth problem.

5. **The L=8 vs L=16 discrepancy is interference**, not curriculum. Longer sequences with more query slots pull the shared kernel in incompatible directions.

---

## 8. Concrete falsification agenda (ranked by decisiveness)

| # | Experiment | Prediction if my reconstruction is right | Prediction if Council's depth/masking story is right |
|---|---|---|---|
| 1 | Log per-cell gates & deltas over ticks | Gates collapse at cells ≥ 4 | Gates uniform, gradients small |
| 2 | Post-hoc linear probe on cells 4–6 | No decodable parity | Decodable but readout fails |
| 3 | Loss only at Slot 3 | Slot 3 still fails | Slot 3 learns (masking was culprit) |
| 4 | Per-cell independent kernels | All slots learn | Slot 0 still dominates |
| 5 | L=16, loss only at {3,7} | Slot 1 recovers to L=8 level | Slot 1 still fails |
| 6 | Extend T to 64 | No change (frozen) | Slots improve (transient) |

If Experiments 1, 3, and 4 come back as I predict, the Council's "DAG solved relaxation, now we need curriculum" narrative is falsified, and the real story is **shared-kernel specialization to a local shortcut under a multi-slot loss**. The fix is not curriculum — it is either (a) loss shaping that forces transport (e.g., supervise intermediate cells, or use a transport-consistency auxiliary loss), or (b) architectural separation of the local and transport computations (e.g., a dedicated carry channel that is not shared with the local update).

---

## 9. One more skeptical note on the metrics

The Council reports "Slot 0 = 85.6%, highest recorded in project history" as a triumph. But Slot 0 is cell 3, which is **3 cells from the left boundary**. With a causal stencil and Dirichlet left boundary, cell 3 sees a 4-cell input window. Parity on a 4-cell window is a 16-pattern problem — trivially learnable by a local MLP. **85.6% on a 4-cell local parity is not impressive; it is the expected performance of a feedforward classifier.** The fact that it's the "highest recorded" says more about how weak the baselines were than about the CCA's sequential capabilities.

The real test of the CCA is whether it can do parity on a 16-cell window with a *single* shared kernel and *no* local shortcut. It cannot, and the data shows it cannot. The DAG stencil was necessary but nowhere near sufficient.

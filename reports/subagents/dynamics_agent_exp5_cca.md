=== [DYNAMICS-AGENT] (Model: deepseek/deepseek-v4.1-flash, Time: 89.50s) ===
# Dynamical Systems Analysis of Experiment 5: The Propagation Bottleneck

## 1. Diagnosis: The CCA Is Not a Recurrent System — It Is a 256-Layer Feedforward Net with Shared Weights

The Adversarial Reviewer is *partially* correct but mischaracterizes the mechanism. Let me be precise about what the telemetry tells us.

### 1.1 Lyapunov spectrum of the causal Jacobian

With the strictly causal stencil, the activated Jacobian at tick $t$ is

$$J_t = I + \alpha \cdot \mathrm{diag}(g_t)\,\partial_\delta\!\big(\tanh\big)(\cdot)\,W_\delta\,J_{h_1,t}\,[\,I\;\;|\;\;-\!I\;\;|\;\;I\,]\;\cdots$$

but crucially, the *spatial* coupling is now lower-bidiagonal in the cell index: cell $i$ at tick $t+1$ depends on cells $\{i-1,i\}$ at tick $t$. Compose over $T=16$ ticks and $L=16$ cells and you get a triangular DAG with longest path

$$\Lambda = L \cdot T = 256 \text{ directed steps}, \qquad \text{fan-in} = 2^{\Lambda}.$$

This is the structural absurdity hiding in plain sight: **a causal cellular automaton with $T$ internal ticks is not a CA at all — it is a depth-$LT$ feedforward network with tied weights and $O(LT)$ effective receptive field.** Any claim of "recurrence" in Track 1 is false. There is no cycle. There is no attractor. There is no Lyapunov spectrum in the usual sense — only a *product of triangular Jacobians* whose diagonal entries are the local amplification factors

$$\lambda_i^{(t)} = \partial x_i^{t+1}/\partial x_i^{t} = 1 + \alpha\, g_i^t\,\delta'_i \cdot (\partial \text{perc}_{x}),$$

and whose sub-diagonal entries are the transport couplings $\kappa_i^{(t)} = \partial x_i^{t+1}/\partial x_{i-1}^{t}$.

### 1.2 Why Slot 0 is a feedforward shadow, not a carry state

Slot 0 sits at cell 3. Its receptive cone into cell 3 at tick $T=16$ spans cells $\{0,1,2,3\}$ — four cells, i.e., **$L_0 = 4 \ll L$**. Slot 0 does not require any sequential carry *through the query boundary*. It is solvable by a purely local, feedforward 4-cell window. The 85.6% is exactly what a 4-cell perceptron with a shared kernel and 16 ticks of residual refinement can achieve. **The learning signal never had to cross a spatial fold to score Slot 0.**

### 1.3 Why Slots 1–3 are pinned at chance: a *gradient-cone shrinking* problem, not vanishing gradients

The Adversarial Reviewer says "vanishing gradients." That is the standard excuse and it is wrong here — AdamW with residual accumulation $(x + \alpha\delta g)$ does not typically vanish; it *scrambles*. The real mechanism is more subtle and more damning:

**The Jacobian from Slot 1's loss into the parameters at cell $i \le 3$ traverses the sub-diagonal transport chain**

$$\frac{\partial x_7^{(16)}}{\partial \theta_{0..3}} \propto \prod_{j=4}^{7} \kappa_j^{(t_j)} \cdot \prod_{t} \lambda^{(t)}.$$

Because the kernel is *tied* across cells (translation equivariance), the same $(\lambda, \kappa)$ pair governs Slot 0's local solve and Slot 1's transport. Gradient descent faces a **two-objective competition on the same parameters**:

- Slot 0's loss wants $(\lambda, \kappa)$ tuned to a *local identity/parity detector* — it prefers $\kappa \approx 0$ (minimize leakage, short receptive field is enough) and $\lambda \approx 1$ (persist local bit).
- Slot 1's loss wants $\kappa \neq 0$ with *nontrivial signed structure* to carry a sequential invariant across 4 cells without corruption.

In a translation-equivariant kernel, $\kappa \approx 0$ is the *lower-loss fixed point for the joint objective* because it wins Slot 0 outright and only defers Slots 1–3. This is **gradient masking by an easy sub-problem**, not vanishing gradient. The telemetry confirms it: Slot 2 on parity sits at **50.2%**, essentially identical to chance — not "weakly above chance via leaked signal" but *exactly at the uninformative fixed point*, indicating the transport channel is not merely attenuated but **actively quenched to zero by the learned gate** $g$.

Check the readout side: the gate $\mathrm{sigmoid}(W_g h_1)$ receives inputs from cell $i$'s *own* perception $[x_i, x_{i-1}, x_i - x_{i-1}]$. Nothing in the gate's receptive field knows that cell $i$ is a query site or that it must *forward* a state. The gate is a **local model of local self-relevance**, and it converges to "I am my own answer" — killing the transport term.

---

## 2. Why a Translation-Equivariant CA Cannot, By Itself, Learn an Invariant Carry

Here is the deep architectural argument the Council needs to internalize:

**A translation-equivariant, strictly causal CA with a per-cell readout has no structurally privileged location for a global carry state.** The "state" the network carries is *per-cell* residual content $x_i^{(t)}$. To carry information from cell 0 to cell 15 you need the kernel at every cell to *refuse to overwrite* the inbound signal — but nothing in the loss or architecture enforces that refusal. The gate can always decide "I am terminal" at every $i$.

This is a **symmetry-breaking defect**: translation equivariance is correct for the *physics* (the automaton rule) but wrong for the *task interface* (only cells 3, 7, 11, 15 are queried). The network correctly infers from the loss that "most cells are not queried" and — given a shared kernel — allocates its finite representational budget to the *densely supervised* leftmost queried cell (Slot 0), which happens to be the only one with a short receptive cone. The equivariance that made the model honest on Track 2's "shared rule" aspect is precisely what starves Slots 1–3.

**This is not a bug. It is the theorem.** Translation-equivariant causal propagation with spatially sparse supervision and a tied kernel converges to the fixpoint where the transport coefficient is quenched in favor of the locally-solvable subproblem.

---

## 3. The Resolution: Force an Invariant Sequential Carry

There are four classes of intervention. They are not equivalent; the Council should pre-register which one they use and why. I present them in order of dynamical-systems soundness.

### Intervention A: **Causal Recurrence with an Explicit Carry Channel (RECOMMENDED)**

Decouple the *cell state* from the *carry state*. Introduce a second, non-spatial (or shared-register) latent $c^{(t)} \in \mathbb{R}^C$ updated by a **global** transport:

$$c^{(t+1)} = \mathrm{GRU}\big(c^{(t)}, \textstyle\sum_i w_i \odot x_i^{(t)}\big), \qquad x_i^{(t+1)} = x_i^{(t)} + \alpha g_i^t \delta_i^t + \gamma\, c^{(t+1)}.$$

The readout at cell $i$ uses $[\,x_i^{(T)};\, c^{(T)}\,]$ for queried cells only. This is a genuine recurrent system: there *is* a Lyapunov spectrum of the $c$-update, there *are* attractors, and the carry is architecturally non-local so it *cannot* be memorized by a 4-cell window. Slot 0's local shortcut is still available, but it is now competed against by a mechanism that trivially solves Slots 1–3. Gradient masking disappears because the carry channel offers all four queried cells a *shared, cheap* solution.

**Dynamical justification**: the $c$-channel adds a slow mode $\lambda_c \approx 1$ orthogonal to the fast local residual modes. Recurrent training then has a well-conditioned gradient path (BPTT over the $c$-chain only, depth $T=16$, not $LT=256$).

### Intervention B: **Non-Equivariant Position Embeddings on the Transport Channel Only**

Keep the CA kernel translation-equivariant (preserving the physics), but add a learned per-cell transport gain:

$$x_i^{(t+1)} = x_i^{(t)} + \alpha\, g_i^t\,\delta_i^t + \eta_i \cdot \text{Transport}(x_{i-1}^{(t)}).$$

The $\eta_i$ are the *only* position-dependent parameters. On the shared kernel they add $O(L)$ parameters and *break the equivariance degeneracy that kills transport* without breaking the automaton's local rule. The network can now learn "cells 0–3 are transmitters (high $\eta$), cells 4–15 are terminal (low $\eta$)" — which is exactly the structure the parity task demands. **Caution**: this is a weaker intervention than A; it can still be gradient-masked if $\eta_i$ is learned jointly. Fix by initializing $\eta_i \equiv 1$ and freezing for the first 50 epochs.

### Intervention C: **Per-Slot Loss Reweighting / Reverse Curriculum**

The current loss supervises all four slots from tick $T=16$ simultaneously. Reverse the temporal and spatial curriculum:

- Phase 1 (epochs 0–50): supervise **only Slot 3** (cell 15) at $T=16$. Force the network to solve the *hardest* transport problem first. The kernel is *obligated* to develop a transport mode because there is no local shortcut to Slot 3.
- Phase 2 (epochs 50–150): add Slot 2, then Slot 1, then Slot 0 (or jointly with Slot 3 dominating).
- **Weight the Slot 3 loss by its receptive cone length** ($\propto 16$) to balance gradient contributions.

This directly refutes the loss-masking hypothesis by *inverting* the difficulty ordering. If Slot 3 does not come up under this curriculum, the culprit is truly architectural; if it does, the original runs were a curriculum pathology, not a capacity limitation.

### Intervention D: **Single-Shot Carry Regularization**

Add a loss term that *forbids* the network from reading out Slot 1 at time $T$ without having consulted the state at cell 0:

$$\mathcal{L}_{\text{carry}} = \big\|\partial x_{7}^{(T)}/\partial x_{0}^{(0)}\big\|_F \ \ \text{or, more tractably,}\quad \mathcal{L}_{\text{probe}} = \mathbb{E}\big[\|\text{stopgrad}(x_{4}^{(t)}) - \text{stopgrad}(x_{3}^{(t-1)})\|^2\big]^{-1}.$$

Soft but diagnostically decisive: if the carry term cannot be raised, the problem is optimization; if it can but accuracy does not follow, the problem is readout capacity.

---

## 4. Direct Answers to the Council's Three Questions

**Q1 — Spatial bottleneck / vanishing gradients across depth 256.**
The 16×16 = 256-step unrolled chain is a *feedforward depth*, not a recurrent depth. Gradients do not "vanish" so much as they *compete*: Slot 0's loss reaches the parameters through depth $\sim 48$ steps; Slot 3's through depth $\sim 256$. With a tied kernel and AdamW, the shallow objective's gradient signal is $O(5\times)$ larger and structurally easier, so the kernel converges into the local-solve basin. The fix is **depth decoupling** (Intervention A: BPTT over a $T=16$ carry chain, not $LT=256$).

**Q2 — Loss gradient masking.**
Correct, and this is the dominant effect. Slot 0 stabilizes the shared kernel into "short-range parity detector." Because the kernel is tied, this directly caps Slot 1's achievable transport — but the network *cannot recover*, because perturbing the kernel to help Slot 1 immediately worsens Slot 0 in the current parameter region. This is a **shared-parameter local minimum**, the classic contrastive-multitask trap. Fix with **Intervention C (reverse curriculum) or A (decoupled carry)**.

**Q3 — What breaks the spatial barrier?**
The barrier is broken only by an intervention that makes the *transport channel structurally non-substitutable*. In decreasing order of expected efficacy:
1. **A (explicit shared carry register)** — provides a cheap, non-local solution no 4-cell window can mimic.
2. **C (reverse curriculum, Slot 3 first)** — removes the local shortcut's dominance in the loss.
3. **B (per-cell transport gain, frozen init)** — breaks equivariance on the one channel that matters.
4. **D (carry regularization)** — diagnostic, not a solution alone.

---

## 5. Adversarial Reviewer's Claim, Adjudicated

> "Slot 0's 85.6% is merely a local 4-cell feedforward memorization, and the shared kernel specialized into a local shortcut while downstream gates quenched."

**Verdict: substantially correct, with one important correction.** It is not *memorization of the batch* — the identity gap $G_{\text{identity}} = +10.4\%$ rules out instance-specific rote storage. It is **local function solving in a shared kernel** — a legitimate but non-sequential computation over a 4-cell receptive cone. The downstream gates quenched because a translation-equivariant gate has no positional signal that says "you are a transmitter." To force genuine sequential carry, the Council must either (i) give the system a non-local recurrent register, or (ii) break the equivariance degeneracy on the transport channel. Doing neither will reproduce the current result on $L=32$ with an even starker Slot-0-only collapse.

---

## 6. Pre-Registered Recommendation

For Experiment 6, run a $2\times 2$ factorial: {Intervention A, Intervention C} × {stochastic depth on transport, no stochastic depth}, with the primary endpoint being **Slot 3 accuracy on Parity** (not Slot 0). Slot 0 has saturated its diagnostic value. The decisive test of whether CA learning can carry sequential invariant state is *whether the far queried cell comes online* — everything else is bookkeeping.

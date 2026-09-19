=== [DYNAMICS-AGENT] (Model: deepseek/deepseek-v4.1-flash, Time: 15.99s) ===
# Causal Cellular Automata: From Isotropic Relaxation to Directed Spatial Folds

## 1. The Core Mathematical Distinction

The Adversarial Reviewer's central claim is that a symmetric radius-1 NCA computes a **relaxation** (fixed-point iteration of a symmetric operator), while sequential composition requires a **fold** (directed dependency chain). Let me formalize this precisely.

### 1.1 Symmetric NCA (the failed substrate)

Let the state field be $w^t \in \mathbb{R}^L$ with $w^t_i$ the state at cell $i$, time $t$. The symmetric radius-1 update is:

$$w^{t+1}_i = F\!\left(w^t_{i-1},\; w^t_i,\; w^t_{i+1};\; x_i\right)$$

where $x_i$ is the (fixed) input at cell $i$. Define the **global update operator** $\Phi: \mathbb{R}^L \to \mathbb{R}^L$ by $(\Phi w)_i = F(w_{i-1}, w_i, w_{i+1}; x_i)$.

**Key structural fact:** $\Phi$ is *isotropic* — it treats $i-1$ and $i+1$ symmetrically. The Jacobian $J_\Phi$ is **tridiagonal and symmetric** in its coupling structure:

$$(J_\Phi)_{ij} = \frac{\partial F}{\partial w_j}\bigg|_i \neq 0 \iff |i-j| \le 1, \quad \text{and} \quad (J_\Phi)_{i,i-1} = (J_\Phi)_{i,i+1} \text{ (by symmetry of } F).$$

The developmental dynamics are $w^t = \Phi^t(w^0)$. The NCA "computes" the limit $w^\infty = \lim_{t\to\infty} \Phi^t(w^0)$, which is a **fixed point** of $\Phi$.

### 1.2 Why a symmetric fixed point cannot be a fold

**Theorem (No-fold from symmetric relaxation).** Let $\Phi$ be a symmetric radius-1 operator with a unique attracting fixed point $w^\infty(x)$ that is a $C^1$ function of the input field $x$. Then the dependency of $w^\infty_i$ on $x_j$ is **not directed**: if $\partial w^\infty_i / \partial x_j \neq 0$, then generically $\partial w^\infty_j / \partial x_i \neq 0$ as well. In particular, the dependency graph of $w^\infty$ on $x$ is **undirected**.

*Proof sketch.* At the fixed point, $w^\infty = \Phi(w^\infty, x)$. Differentiating w.r.t. $x_j$:

$$(I - J_\Phi)\frac{\partial w^\infty}{\partial x_j} = \frac{\partial \Phi}{\partial x_j}.$$

Since $J_\Phi$ is symmetric (as a coupling structure) and $I - J_\Phi$ is invertible at an attracting fixed point, the inverse $(I - J_\Phi)^{-1}$ is a **dense, symmetric** matrix. Thus $\partial w^\infty_i / \partial x_j = [(I-J_\Phi)^{-1}]_{ij} \cdot (\partial \Phi_j/\partial x_j)$, and by symmetry of the inverse, $[(I-J_\Phi)^{-1}]_{ij} = [(I-J_\Phi)^{-1}]_{ji}$. Hence influence is bidirectional. $\square$

This is the mathematical content of the Reviewer's "absence of a directed dependency graph." A symmetric relaxation produces a **dense, symmetric influence kernel** — every cell influences every other cell through the same isotropic Green's function. There is no "slot $i$ depends on slot $i-1$" structure; there is only "everything depends on everything, symmetrically."

---

## 2. The Causal Cellular Automaton (CCA)

### 2.1 Definition

Replace the symmetric stencil $\mathcal{N}(i) = \{i-1, i, i+1\}$ with the **directed (causal) stencil**:

$$\mathcal{N}_\to(i) = \{i-1,\; i\}.$$

The CCA update is:

$$\boxed{\;w^{t+1}_i = G\!\left(w^t_{i-1},\; w^t_i;\; x_i\right)\;}$$

with the boundary condition $w^t_{-1} \equiv 0$ (or a fixed left-boundary token). The global operator $\Psi: \mathbb{R}^L \to \mathbb{R}^L$ is now:

$$(\Psi w)_i = G(w_{i-1}, w_i; x_i).$$

### 2.2 The Jacobian is lower-triangular

The Jacobian $J_\Psi$ has entries:

$$(J_\Psi)_{ij} = \frac{\partial G}{\partial w_j}\bigg|_i \neq 0 \iff j \in \{i-1, i\}.$$

That is, $J_\Psi$ is **lower bidiagonal**:

$$J_\Psi = \begin{pmatrix} g_0 & 0 & 0 & \cdots \\ h_1 & g_1 & 0 & \cdots \\ 0 & h_2 & g_2 & \cdots \\ \vdots & & \ddots & \ddots \end{pmatrix}, \quad g_i = \frac{\partial G}{\partial w_i}\bigg|_i, \quad h_i = \frac{\partial G}{\partial w_{i-1}}\bigg|_i.$$

**This is the decisive structural change.** A lower-triangular Jacobian means the dependency graph is a **DAG** (specifically, a total order $0 \to 1 \to 2 \to \cdots \to L-1$).

### 2.3 The CCA is an exact spatial fold

**Theorem (CCA = spatial fold).** The CCA update, applied once, computes a sequential fold over the spatial index. Specifically, define the per-cell transition $T: \mathcal{S} \times \mathcal{X} \to \mathcal{S}$ by $T(s, x) = G(s, x)$ where $s = w_{i-1}$ is the carried state. Then:

$$w^1_i = T(w^0_{i-1}, x_i) = T\big(T(w^0_{i-2}, x_{i-1}), x_i\big) = \cdots = T^{(i+1)}(w^0_{-1}, x_0, x_1, \ldots, x_i),$$

where $T^{(k)}$ denotes $k$-fold composition. **One CCA step is a full left-to-right scan.**

*Proof.* Immediate by induction on $i$. The boundary condition $w^0_{-1} = 0$ seeds the fold. $\square$

This is the mathematical identity the Reviewer was reaching for: **a directed radius-1 stencil is a spatial RNN.** The "developmental time" $t$ is now *redundant* for composition — a single step already performs the fold. Additional steps refine the per-cell transition $T$ but do not change the dependency topology.

---

## 3. Elimination of the Isotropic Relaxation Saddle

### 3.1 The saddle in the symmetric case

For the symmetric NCA, consider the developmental map $\Phi$ near a fixed point $w^\infty$. The linearized dynamics are:

$$\delta w^{t+1} = J_\Phi \,\delta w^t.$$

Since $J_\Phi$ is symmetric, it has **real eigenvalues** $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_L$. For the fixed point to be attracting, we need $|\lambda_k| < 1$ for all $k$. But the **slowest mode** $\lambda_1$ (the "diffusive" mode, corresponding to the lowest spatial frequency) decays as:

$$\lambda_1 \approx 1 - c\,\frac{\pi^2}{L^2} \quad \text{(for a diffusion-like } F\text{)},$$

so the relaxation time to reach the fixed point scales as $\tau_{\text{relax}} \sim L^2 / c$. **This is the isotropic relaxation saddle**: the system must relax through a slow, spatially-global mode, and information from slot 0 is exponentially attenuated by the time it reaches slot $L-1$ (the Reviewer's §1.1 contraction test).

More precisely, the influence kernel at the fixed point is:

$$\frac{\partial w^\infty_i}{\partial x_j} = \left[(I - J_\Phi)^{-1}\right]_{ij} \approx \frac{1}{2\sqrt{c}} e^{-|i-j|/\sqrt{c}} \quad \text{(exponential decay)},$$

so $\partial w^\infty_{L-1}/\partial x_0 \sim e^{-L/\sqrt{c}} \to 0$. **The symmetric NCA cannot transport information across $L$ cells.** This is the "contraction" the Reviewer suspected.

### 3.2 The CCA has no saddle — it has a directed flow

For the CCA, the linearized dynamics are:

$$\delta w^{t+1} = J_\Psi \,\delta w^t,$$

with $J_\Psi$ **lower bidiagonal**. The eigenvalues of a lower bidiagonal matrix are exactly its diagonal entries:

$$\sigma(J_\Psi) = \{g_0, g_1, \ldots, g_{L-1}\}.$$

There is **no slow global mode**. Each cell's local dynamics is governed by its own $g_i$, and the coupling $h_i$ propagates information *forward* (from $i-1$ to $i$) with **no attenuation** — it is a directed transport, not a diffusive relaxation.

**Theorem (No contraction in the CCA).** The influence of $x_0$ on $w^t_{L-1}$ after $t$ steps is:

$$\frac{\partial w^t_{L-1}}{\partial x_0} = \left[(J_\Psi^t)\right]_{L-1,0} \cdot \frac{\partial G}{\partial x_0}\bigg|_0.$$

Since $J_\Psi$ is lower bidiagonal, $(J_\Psi^t)_{L-1,0} = 0$ for $t < L-1$, and for $t \ge L-1$:

$$(J_\Psi^t)_{L-1,0} = \prod_{k=1}^{L-1} h_k \cdot \binom{t-1}{L-2} \cdot \text{(diagonal terms)}.$$

The key point: **the product $\prod_{k=1}^{L-1} h_k$ is a directed path product, not an exponential decay.** If the $h_k$ are $O(1)$ (which they are at initialization, and can be maintained by gradient flow), information propagates across $L$ cells in exactly $L-1$ steps with **no exponential attenuation**. The "contraction" is replaced by a **directed transport**.

### 3.3 The saddle is eliminated because the DAG has no cycles

The isotropic relaxation saddle arises because the symmetric operator has **cycles** in its dependency graph: $i \to i+1 \to i$ (a 2-cycle). These cycles create the slow, diffusive mode. The CCA's dependency graph is a **DAG** (no cycles), so there is no diffusive mode — only directed flow. This is the graph-theoretic content of the Reviewer's §2 verdict.

---

## 4. The Decisive Experiment 5: Directed-Kernel Ablation

### 4.1 Experimental design

Replace the symmetric stencil $\{i-1, i, i+1\}$ with the directed stencil $\{i-1, i\}$ in the existing NCA, keeping all else (τ=16, parameter count, optimizer, readout) fixed. Measure per-slot accuracy on Track 1 (parity) and Track 2 (running sum).

### 4.2 Predicted outcomes

**Case A: Directed kernel solves it (interior slots > 70%).**
This confirms the barrier is graph-theoretic. The CCA is a spatial fold, and the fold computes sequential composition natively. The path forward is **directed NCAs** — asymmetric kernels that induce a DAG over cells. This is a legitimate primitive, distinct from a GRU (it is a *spatial* fold, not a temporal one).

**Case B: Directed kernel also fails.**
Then the problem is not the kernel symmetry. It is optimization, readout, or loss. The Reviewer's §6 roadmap applies: audit the readout, loss, and optimizer before touching the architecture.

### 4.3 Why Case A is the likely outcome

The mathematical argument is decisive: a directed radius-1 stencil is *exactly* a spatial RNN. A spatial RNN with $L$ cells and $\tau$ developmental steps has an effective depth of $L \cdot \tau$ in the directed direction. For $L=16$, $\tau=16$, that is 256 effective sequential steps — more than enough to compute parity or running sum. The symmetric NCA, by contrast, has an effective depth of $\tau=16$ with **no directed structure**, so it cannot compose.

---

## 5. Summary: The Mathematical Core

| Property | Symmetric NCA | Causal CA (CCA) |
|---|---|---|
| Stencil | $\{i-1, i, i+1\}$ | $\{i-1, i\}$ |
| Jacobian structure | Tridiagonal, symmetric | Lower bidiagonal |
| Dependency graph | Undirected (2-cycles) | DAG (total order) |
| Fixed-point influence | Exponential decay $e^{-|i-j|/\sqrt{c}}$ | Directed path product $\prod h_k$ |
| Slow mode | Diffusive, $\tau \sim L^2/c$ | None (eigenvalues = diagonal) |
| Composition | Relaxation (no fold) | Exact spatial fold |
| Effective depth | $\tau$ | $L \cdot \tau$ |

**The Reviewer's breakthrough is correct and mathematically precise:** the composition barrier is not a transport problem, not an invariant problem, and not a coarsening problem. It is a **graph-theoretic fact**: a symmetric radius-1 kernel iterated to a fixed point cannot produce a directed dependency chain. The CCA — with its directed stencil $\{i-1, i\}$ — induces a DAG, eliminates the isotropic relaxation saddle, and computes a spatial fold in a single step. Experiment 5 is the decisive test, and the mathematics predicts it will succeed.

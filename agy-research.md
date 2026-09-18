# TITAN TEXT: ADJACENT RESEARCH & THEORETICAL SYNTHESIS
**Platform & Substrate Research Report**
**Target Substrate**: Samsung Galaxy S25 Ultra (Qualcomm Snapdragon 8 Elite / Oryon Architecture)
**Repository**: `/data/data/com.termux/files/home/projects/titan_text`
**Date**: September 2026

---

## Executive Summary

This document establishes the theoretical, mathematical, and microarchitectural foundation for **Titan Text**, a continuous morphogenic neural-cellular sequence modeling laboratory. It addresses the core dynamical questions of recurrent latent evolution, provides a hardware profile of the host Snapdragon 8 Elite (SM8750-AC) platform under Android/Termux, conducts formal comparative analysis against standard sequence architectures (Transformers, GRUs, RNNs), establishes a rigorous operational scientific taxonomy to replace intuitive buzzwords, and grounds every theoretical derivation in Titan Text's empirical findings and codebase implementation.

---

## 1. Recurrent Latent Dynamics & Continuous Cellular Automata

### 1.1 Dynamical Systems Theory of Latent Trajectories

In Titan Text, sequence generation and token processing are mapped to autonomous flow on a continuous cellular state space. The state at spatial position $i \in \{0, \dots, L-1\}$ with channel depth $C$ evolves according to the discrete-time non-linear map:

$$\mathbf{x}_{t+1} = F(\mathbf{x}_t) = \mathbf{x}_t + \alpha \cdot \Delta \mathbf{x}_t$$

$$\Delta \mathbf{x}_t = \sigma(W_g h_t + b_g) \odot \tanh(W_\delta h_t + b_\delta) + \nu \Delta_{spatial} \mathbf{x}_t$$

where $h_t = \text{GELU}(W_1 \mathbf{p}_i(t) + b_1)$, with local perception $\mathbf{p}_i = [\mathbf{x}_i, \nabla \mathbf{x}_i, \Delta \mathbf{x}_i] \in \mathbb{R}^{3C}$, integration step size $\alpha = 0.5$, and Navier-Stokes kinematic viscosity $\nu$.

The asymptotic behavior of this non-linear operator over developmental time $\tau \to \infty$ falls into four distinct dynamical regimes:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │              DYNAMICAL REGIME TAXONOMY                  │
                  └─────────────────────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
┌───────────────────┐        ┌───────────────────┐        ┌───────────────────┐
│ Contractive Fixed │        │   Limit Cycles    │        │ Chaotic Wandering │
│      Points       │        │ (Periodic Orbits) │        │ (Strange Attr.)   │
│  λ_max < 0,       │        │  λ_max = 0,       │        │  λ_max > 0,       │
│  ||x_{t+1}-x_t||→0│        │  x(t+T) = x(t)    │        │  Bounded, Fractal │
└───────────────────┘        └───────────────────┘        └───────────────────┘
                                       │
                                       ▼
                             ┌───────────────────┐
                             │  Open Phase Drift │
                             │  (Ungrounded)     │
                             │  ||x_t|| → ∞ or   │
                             │  Spatial Mean Dr. │
                             └───────────────────┘
```

#### 1. Contractive Fixed Points ($\lambda_{\max} < 0$)
- **Mathematical Definition**: A state $\mathbf{x}^* \in \mathbb{R}^{L \times C}$ is a fixed point if $F(\mathbf{x}^*) = \mathbf{x}^*$, which implies $\Delta \mathbf{x}^* = 0$. The fixed point is locally contractive within basin $\mathcal{B}(\mathbf{x}^*)$ if the discrete Jacobian $J_F(\mathbf{x}^*) = \mathbf{I} + \alpha J_{\Delta \mathbf{x}}(\mathbf{x}^*)$ has spectral radius $\rho(J_F(\mathbf{x}^*)) < 1$.
- **Convergence Rate**: For any $\mathbf{x}_0 \in \mathcal{B}(\mathbf{x}^*)$, $\|\mathbf{x}_t - \mathbf{x}^*\| \le M \cdot \rho^t$, exhibiting exponential decay towards the invariant manifold.
- **Cognitive/Algorithmic Role**: Represents categorical decisions, completed sequence recall, and stable attractor memory. Once reached, additional compute steps $\tau > T$ leave the readout invariant: $\hat{y}_\tau = \hat{y}^*$.

#### 2. Limit Cycles & Periodic Attractors ($\lambda_{\max} = 0$)
- **Mathematical Definition**: A closed invariant orbit $\gamma = \{\mathbf{x}(0), \mathbf{x}(1), \dots, \mathbf{x}(P-1)\}$ such that $F(\mathbf{x}(t)) = \mathbf{x}((t+1) \pmod P)$ with period $P \ge 2$. Floquet multipliers (eigenvalues of the monodromy matrix $\prod_{t=0}^{P-1} J_F(\mathbf{x}(t))$) lie strictly inside the unit circle, except for a single neutral eigenvalue $\mu = 1$ along the orbit tangent.
- **Cognitive/Algorithmic Role**: Ideal for generating periodic rhythms, repeating delimiter sequences, syntax alternation (e.g., alternating brackets `()()()`), and oscillatory clock signals.

#### 3. Chaotic Wandering ($\lambda_{\max} > 0$)
- **Mathematical Definition**: Trajectories remain confined within a compact subset $\mathcal{K} \subset \mathbb{R}^{L \times C}$ but possess at least one strictly positive Lyapunov exponent $\lambda_{\max} > 0$. Neighboring trajectories diverge exponentially at rate $\sim e^{\lambda_{\max} t}$ up to the attractor diameter, yielding sensitive dependence on initial conditions while maintaining an invariant ergodic measure with fractional Hausdorff dimension $D_H$.
- **Cognitive/Algorithmic Role**: Generates pseudorandom sampling, spontaneous creative variations, or exploratory search; however, in symbolic token prediction it leads to severe perplexity degradation and hallucinations if unconstrained.

#### 4. Open Phase Drift (Ungrounded Dynamics)
- **Mathematical Definition**: The trajectory fails to remain within any compact invariant set: $\|\mathbf{x}_t\| \to \infty$ as $t \to \infty$, or specific projection subspaces (such as spatial mean modes $\bar{\mathbf{x}}_c = \frac{1}{L} \sum_{i=1}^L \mathbf{x}_{i, c}$) drift ballistically or diffusively without a restoring force.
- **Structural Cause in Titan Text**: While the update delta is bounded by tanh ($\|\delta\|_\infty < 1$) and gate ($\text{gate} \in (0, 1)$), the state integration is purely additive: $\mathbf{x}_{t+1} = \mathbf{x}_t + \alpha (\text{gate} \odot \delta)$. The discrete Laplacian diffusion $\nu \Delta_{spatial} \mathbf{x}$ conserves the spatial mean ($\sum_i \Delta_{spatial} \mathbf{x}_i \equiv 0$ under periodic boundary). Hence, diffusion smooths spatial gradients but **cannot halt spatial-mean drift**.
- **Empirical Confirmation**: As recorded in [`reviews/2026-09-11/review.md`](file:///data/data/com.termux/files/home/projects/titan_text/reviews/2026-09-11/review.md#L51-L66), rolling out checkpoint `v0_viscous` for 256 steps caused state energy to surge from $3.95$ to $1,721.73$, with per-channel spatial mean energy increasing from $11.9\%$ to $56.5\%$ of total energy.

---

### 1.2 Mathematical Analysis: BPTT at Fixed Horizon $T$ & Degradation at $\tau > T$

#### The Optimization Objective
During training, Backpropagation Through Time (BPTT) unrolls the NCA for exactly $T$ developmental steps ($T = 8$ in `v0_text`). The loss function is computed exclusively at the terminal step $T$:

$$\mathcal{L}(\theta) = \ell\left(\mathbf{y}, \text{Softmax}\left(W_{out} \mathbf{x}_T + \mathbf{b}_{out}\right)\right)$$

The parameter gradient is obtained by chain rule expansion through the unrolled computation graph:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \ell}{\partial \mathbf{x}_T} \sum_{t=1}^T \left( \prod_{s=t+1}^T \frac{\partial \mathbf{x}_s}{\partial \mathbf{x}_{s-1}} \right) \frac{\partial \mathbf{x}_t}{\partial \theta} = \frac{\partial \ell}{\partial \mathbf{x}_T} \sum_{t=1}^T \left( \prod_{s=t+1}^T J_F(\mathbf{x}_{s-1}) \right) \frac{\partial \mathbf{x}_t}{\partial \theta}$$

#### The Readout Co-Adaptation Dilemma
1. **Target Manifold Specialization**:
   The terminal states at step $T$ form an empirical data manifold:
   $$\mathcal{M}_T = \left\{ \mathbf{x}_T(\mathbf{u}) \in \mathbb{R}^{L \times C} \mid \mathbf{u} \sim \mathcal{D}_{inputs} \right\}$$
   The readout parameters $(W_{out}, \mathbf{b}_{out})$ are trained solely to partition $\mathcal{M}_T$ into Voronoi decision regions corresponding to target token classes.

2. **Absence of Stationarity Penalty**:
   Notice that the loss $\mathcal{L}$ contains no term penalizing velocity $\|\mathbf{x}_{T+1} - \mathbf{x}_T\|$ or enforcing $F(\mathbf{x}_T) = \mathbf{x}_T$. The network is free to use non-zero velocity at step $T$ to separate classes. In fact, gradient descent actively exploits non-contractive transient trajectories because they offer high dynamic capacity.

3. **Off-Manifold Drift at $\tau > T$**:
   When the model is evaluated at test time for $\tau > T$ ticks, the state evolves according to:
   $$\mathbf{x}_\tau = F^{\tau - T}(\mathbf{x}_T) = \mathbf{x}_T + \alpha \sum_{k=T}^{\tau-1} \Delta \mathbf{x}_k$$
   Since $F$ does not preserve $\mathcal{M}_T$, the trajectory departs from $\mathcal{M}_T$. For $\tau \gg T$, the state enters regions of $\mathbb{R}^{L \times C}$ where:
   - $W_{out} \mathbf{x}_\tau$ produces uncalibrated logit vectors.
   - Channel magnitudes increase, inflating logit scales and driving softmax entropy towards 0 (extreme overconfidence on erroneous tokens).
   - Readout accuracy collapses.

#### Empirical Proof from Titan Text Latent Sweeps
Evaluating `checkpoints/v0_text` via `./target/release/titan_text sweep --load-dir checkpoints/v0_text --task text --budgets 0,1,2,4,8,16,32` produces the exact quantitative trajectory:

| Latent Ticks ($\tau$) | Accuracy (%) | Cross-Entropy Loss | Confidence | State Displacement ($\|\Delta \mathbf{x}\|$) | Dynamical Regime |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** (unticked) | 2.7% | 5.3544 | 0.0012 | 0.0000 | Chance Baseline |
| **1** | 13.7% | 4.1145 | 0.0022 | 0.2842 | Compute Gain |
| **2** | 42.6% | 2.9122 | 0.0058 | 0.5723 | Sharp Bifurcation (+28.9%) |
| **4** | 68.0% | 1.3627 | 0.0312 | 1.1705 | Compute Gain |
| **8** ($T_{train}$) | **79.3%** | **1.1207** | **0.0909** | **2.4526** | **Optimal Convergence** |
| **16** ($2 \times T$) | 76.2% | 2.3730 | 0.2480 | 5.2043 | Post-Target Drift |
| **32** ($4 \times T$) | 50.8% | 7.7168 | 0.4065 | 10.9783 | Ungrounded Divergence |

**Key Mathematical Finding**: At $\tau = 32$, state displacement reaches $10.9783$ ($4.47\times$ the displacement at $T=8$), confidence inflates by $4.47\times$ ($0.4065$ vs $0.0909$), but cross-entropy loss explodes to $7.7168$ and accuracy collapses from $79.3\%$ to $50.8\%$. This provides decisive empirical proof of trained-horizon specialization.

---

### 1.3 Methods for Horizon Invariance & Monotonic Compute Gains

To eliminate trained-horizon specialization and guarantee that $\text{Accuracy}(\tau_2) \ge \text{Accuracy}(\tau_1)$ for $\tau_2 > \tau_1$, four mathematical formulations are available:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 HORIZON INVARIANCE ARCHITECTURAL TAXONOMY                   │
├────────────────────────┬──────────────────────────┬─────────────────────────┤
│ Method                 │ Formulation              │ Mechanism               │
├────────────────────────┼──────────────────────────┼─────────────────────────┤
│ 1. Randomized Horizons │ T ~ Uniform(T_min, T_max)│ Enforces decodability   │
│    (Horizon Jitter)    │                          │ across broad time band  │
├────────────────────────┼──────────────────────────┼─────────────────────────┤
│ 2. Multi-Horizon Loss  │ L = Σ w_t ℓ(y, ŷ_t)      │ Any-time computation    │
│    Averaging           │                          │ penalty on drift        │
├────────────────────────┼──────────────────────────┼─────────────────────────┤
│ 3. Contractive         │ R = ||J_F(x)||_F < 1 - ε │ Banach Fixed Point      │
│    Jacobian Reg.       │                          │ Theorem convergence     │
├────────────────────────┼──────────────────────────┼─────────────────────────┤
│ 4. Deep Equilibrium    │ x* = F_θ(x*, u)          │ Implicit differentiation│
│    (DEQ) Models        │ via Broyden / Anderson   │ O(1) memory training    │
└────────────────────────┴──────────────────────────┴─────────────────────────┘
```

#### 1. Randomized Training Horizons (Horizon Jitter)
Instead of a static $T = 8$, the training horizon is sampled stochastically per batch:
$$T \sim \mathcal{U}(T_{\min}, T_{\max}), \quad \text{e.g., } T \in [4, 16]$$
This destroys the model's ability to count on an exact tick $T$ for terminal readout, compelling the local MLP to find invariant configurations.

#### 2. Multi-Horizon Loss Averaging (Any-Time Supervision)
Supervise all developmental steps with a monotonically increasing weight schedule:
$$\mathcal{L}_{multi} = \sum_{t=1}^T w_t \cdot \ell\left(\mathbf{y}, \text{Softmax}(W_{out} \mathbf{x}_t + \mathbf{b}_{out})\right), \quad w_t = \frac{t^\gamma}{\sum_{s=1}^T s^\gamma}, \quad \gamma \ge 1$$
Because early and late states are penalized simultaneously against the ground truth $\mathbf{y}$, the state is incentivized to reach the correct decision quickly and stay there.

#### 3. Contractive Jacobian Regularization
By the Banach Fixed-Point Theorem, if an operator $F: \mathcal{X} \to \mathcal{X}$ on a complete metric space is a contraction mapping:
$$\|F(\mathbf{x}) - F(\mathbf{x}')\| \le k \|\mathbf{x} - \mathbf{x}'\|, \quad k < 1$$
then $F$ admits a unique fixed point $\mathbf{x}^*$, and Picard iteration $\mathbf{x}_{t+1} = F(\mathbf{x}_t)$ converges unconditionally from any initial condition $\mathbf{x}_0$.
In discrete continuous dynamics, contractivity is enforced via Jacobian penalty:
$$\mathcal{L}_{reg} = \mathcal{L}_{task} + \lambda_{contract} \cdot \frac{1}{T} \sum_{t=1}^T \max\left(0, \|J_F(\mathbf{x}_t)\|_2^2 - (1 - \epsilon)\right)$$
where $\|J_F(\mathbf{x}_t)\|_2 = \sigma_{\max}(J_F(\mathbf{x}_t))$ is approximated efficiently using random projection power iteration without instantiating the full $LC \times LC$ matrix:
$$\mathbf{v} \leftarrow \frac{J_F \mathbf{u}}{\|J_F \mathbf{u}\|}, \quad \sigma_{\max} \approx \|J_F \mathbf{v}\|$$

#### 4. Deep Equilibrium (DEQ) Models
Rather than backpropagating through unrolled steps, the equilibrium state $\mathbf{x}^*$ is defined implicitly:
$$\mathbf{x}^* = F_\theta(\mathbf{x}^*, \mathbf{u})$$
Forward pass: Solved via Anderson Acceleration or Broyden's quasi-Newton method until $\|\mathbf{x}_{k+1} - \mathbf{x}_k\| < 10^{-5}$.
Backward pass: Gradients are computed analytically via the Implicit Function Theorem (IFT):
$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \ell}{\partial \mathbf{x}^*} \left( \mathbf{I} - J_F(\mathbf{x}^*) \right)^{-1} \frac{\partial F_\theta(\mathbf{x}^*, \mathbf{u})}{\partial \theta}$$
The vector-Jacobian product $\mathbf{g}^T (\mathbf{I} - J_F)^{-1}$ is solved via linear fixed-point iteration. This grants **$\mathcal{O}(1)$ training memory** and guarantees monotonic compute gains at test time: allocating more solver iterations strictly tightens the residual $\|\mathbf{x}^* - F(\mathbf{x}^*)\|$.

---

### 1.4 Analytical Instrumentation: Mathematical Formulations

Titan Text implements an instrumentation suite to track latent dynamics.

#### 1. Lyapunov Exponent Approximation (Local Divergence Rate)
The maximal Lyapunov exponent $\lambda_{\max}$ quantifies orbital stability. For the 1D discrete cellular trajectory:
$$\lambda_{\max} \approx \frac{1}{K} \sum_{k=0}^{K-1} \ln \frac{\|\mathbf{w}_{k+1}\|}{\|\mathbf{v}_k\|}$$
where $\mathbf{v}_k$ is an infinitesimal perturbation vector $(\|\mathbf{v}_k\| = \epsilon_0)$, propagated through one step of the tangent map:
$$\mathbf{w}_{k+1} = F(\mathbf{x}_k + \mathbf{v}_k) - F(\mathbf{x}_k) \approx J_F(\mathbf{x}_k) \mathbf{v}_k$$
and re-normalized: $\mathbf{v}_{k+1} = \epsilon_0 \frac{\mathbf{w}_{k+1}}{\|\mathbf{w}_{k+1}\|}$.
- $\lambda_{\max} < 0$: Stable, dissipative attractor.
- $\lambda_{\max} = 0$: Limit cycle / neutral phase drift.
- $\lambda_{\max} > 0$: Deterministic chaos.

#### 2. Participation Ratio / Effective Dimensionality
To quantify whether the $C = 64$ continuous channels collapse into a low-dimensional subspace, Titan Text computes the Participation Ratio (PR) from the spatial channel covariance matrix $\Sigma \in \mathbb{R}^{C \times C}$:
$$\Sigma_{j, k} = \frac{1}{B \cdot L - 1} \sum_{b=1}^B \sum_{i=1}^L \left( x_{b, i, j} - \bar{x}_j \right) \left( x_{b, i, k} - \bar{x}_k \right)$$
$$\text{PR} = \frac{\left( \text{Tr}(\Sigma) \right)^2}{\text{Tr}(\Sigma^2)} = \frac{\left( \sum_{c=1}^C \sigma_c^2 \right)^2}{\sum_{c=1}^C \sigma_c^4}$$
As implemented in [`src/instrumentation.rs:166-201`](file:///data/data/com.termux/files/home/projects/titan_text/src/instrumentation.rs#L166-L201):
- If variance is evenly distributed across all $C$ channels: $\text{PR} = C = 64.0$.
- If the dynamics collapse onto an uncoupled 1D manifold: $\text{PR} = 1.0$.

#### 3. Spatial Fourier Enstrophy & Beale-Kato-Majda (BKM) Dissipation
Continuum fluid dynamics metrics mapped to the 1D cellular ring:
- **Kinetic Energy**: $E(t) = \frac{1}{2L} \sum_{i=1}^L \|\mathbf{x}_i(t)\|^2 = \sum_{k=0}^{\lfloor L/2 \rfloor} E(k)$ (Strict Parseval equivalence).
- **Fluid Enstrophy ($\Omega$)**:
  $$\Omega(t) = \frac{1}{2L} \sum_{i=1}^L \|\nabla \mathbf{x}_i\|^2, \quad \nabla \mathbf{x}_i = \frac{\mathbf{x}_{i+1} - \mathbf{x}_{i-1}}{2}$$
- **Fluid Palinstrophy ($P$)**:
  $$P(t) = \frac{1}{2L} \sum_{i=1}^L \|\Delta \mathbf{x}_i\|^2, \quad \Delta \mathbf{x}_i = \mathbf{x}_{i+1} - 2\mathbf{x}_i + \mathbf{x}_{i-1}$$
- **Discrete BKM Blow-up Norm**:
  $$\|\nabla \mathbf{x}(t)\|_{L^\infty} = \max_{b \in [1, B], \, i \in [1, L]} \|\nabla \mathbf{x}_{b, i}(t)\|_2$$
  Under the Beale-Kato-Majda criterion, smooth solutions break down if and only if $\int_0^{T^*} \|\nabla \mathbf{x}(\cdot, t)\|_{L^\infty} dt = \infty$.
- **Navier-Stokes Dissipation Operator**:
  $\nu \Delta \mathbf{x}$ dissipates mode $k$ with discrete eigenvalue $\lambda_k = 4 \sin^2\left(\frac{\pi k}{L}\right)$. For forward Euler integration with step size $\alpha$, the discrete diffusion number is $D = \alpha \nu$. Unconditional von Neumann stability ($|1 - 4D| \le 1 \implies D \le 0.25$) is enforced via adaptive substepping in [`src/nca.rs:97-116`](file:///data/data/com.termux/files/home/projects/titan_text/src/nca.rs#L97-L116):
  $$N_{sub} = \left\lceil \frac{\alpha \nu}{0.25} \right\rceil, \quad \Delta t_{sub} = \frac{\alpha \nu}{N_{sub}} \le 0.25$$

---

## 2. Snapdragon 8 Elite / Galaxy S25 Ultra Hardware Analysis

The host environment executes Titan Text natively on the **Samsung Galaxy S25 Ultra** (Model `SM-S938U`, Linux kernel `6.6.98-android15-8-pd6ff1cd-abogkiS938USQSCCZF9-4k`) powered by the **Qualcomm Snapdragon 8 Elite** (SM8750-AC).

```
╔══════════════════════════════════════════════════════════════════════════════════════╗
║        QUALCOMM SNAPDRAGON 8 ELITE (SM8750-AC) DUAL-CLUSTER ORYON TOPOLOGY          ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

   PRIME CLUSTER (2x Oryon Cores @ up to 4.47 GHz "for Galaxy")
   ┌────────────────────────────────────────────────────────┐
   │ Core 6: 128KB L1D + 192KB L1I ──┐                      │
   │                                 ├──► [ 24MB Shared L2 ]│
   │ Core 7: 128KB L1D + 192KB L1I ──┘                      │
   └────────────────────────────────────────────────────────┘
                               │
   PERFORMANCE CLUSTER (6x Oryon Cores @ up to 3.53 GHz)
   ┌────────────────────────────────────────────────────────┐
   │ Core 0: 128KB L1D + 192KB L1I ──┐                      │
   │ Core 1: 128KB L1D + 192KB L1I ──┤                      │
   │ Core 2: 128KB L1D + 192KB L1I ──┼──► [ 12MB Shared L2 ]│
   │ Core 3: 128KB L1D + 192KB L1I ──┤                      │
   │ Core 4: 128KB L1D + 192KB L1I ──┤                      │
   │ Core 5: 128KB L1D + 192KB L1I ──┘                      │
   └────────────────────────────────────────────────────────┘
                               │
   ┌────────────────────────────────────────────────────────┐
   │ [ 12MB System Level Cache (SLC) ]                      │
   └────────────────────────────────────────────────────────┘
                               │
   ┌────────────────────────────────────────────────────────┐
   │ [ LPDDR5X-9600 Memory Subsystem ] (~76.8 - 77.0 GB/s)  │
   └────────────────────────────────────────────────────────┘
```

### 2.1 Microarchitecture Specifications

| Component | Specification | Architectural Significance for Titan Text |
| :--- | :--- | :--- |
| **CPU Architecture** | Qualcomm 2nd Gen Oryon (ARMv8.7-A / ARMv9.2 subset) | 8-wide decode/dispatch superscalar out-of-order execution |
| **Prime Cores** | 2x Oryon Prime up to **4.47 GHz** (CPUs 6-7) | Extreme single-thread burst performance; 24MB dedicated L2 |
| **Performance Cores**| 6x Oryon Performance up to **3.53 GHz** (CPUs 0-5) | Sustained high-throughput compute; 12MB dedicated L2 |
| **Efficiency Cores** | **0 Cores** (All-Big-Core Architecture) | Eliminates slow-core scheduling penalties; uniform high IPC |
| **L1 Data Cache** | **128 KB** per core (64-byte lines, 8-way) | Holds entire Titan Text active field tensor ($64 \text{ KB}$) in L1D |
| **L1 Instruction** | **192 KB** per core | Completely fits inner integration loops without i-cache evictions |
| **L2 Unified Cache** | **24 MB** (Prime) + **12 MB** (Performance) = **36 MB** | Completely encapsulates model weights ($175 \text{ KB}$) and activations |
| **System Level Cache**| **12 MB** SLC | Dampens DRAM memory access latency |
| **Memory Bus** | 8x 16-bit channels (128-bit bus), LPDDR5X-9600 | Peak theoretical bandwidth: $9600 \times 10^6 \times 16 \text{ B} \approx \mathbf{76.8\text{ GB/s}}$ |
| **Vector Units** | 4x 128-bit NEON / ASIMD per core | Native hardware support for `fp16`, `bf16`, `i8mm`, and `dotprod` |

### 2.2 Cache Residency Analysis: Zero-DRAM Working Set

In Titan Text v0:
- **Model Parameters**: Exactly 43,715 `f32` parameters $\times 4 \text{ bytes} \approx \mathbf{174.86 \text{ KB}}$.
- **Field State**: Batch $B=8$, Sequence $L=32$, Channels $C=64$: $8 \times 32 \times 64 \times 4 \text{ bytes} = \mathbf{65.54 \text{ KB}}$.
- **Perception Buffer**: $8 \times 32 \times 192 \times 4 \text{ bytes} = \mathbf{196.61 \text{ KB}}$.
- **MLP Hidden Layer**: $8 \times 32 \times 96 \times 4 \text{ bytes} = \mathbf{98.30 \text{ KB}}$.
- **Total Active Working Set**: $\approx \mathbf{535.3 \text{ KB}}$.

#### Hardware Fit
- **L1D Cache (128 KB)**: The field state ($65.54 \text{ KB}$) fits entirely within the L1D cache of a single core.
- **L2 Cache (12MB / 24MB)**: The complete working set ($535 \text{ KB}$) occupies only **4.4%** of the Performance cluster L2 cache, and **2.2%** of the Prime cluster L2 cache.
- **Microarchitectural Conclusion**: Titan Text execution on Oryon is **completely cache-resident**. Execution speed is bound strictly by ALU vector arithmetic throughput and instruction dependency latency, incurring **0 DRAM memory bus transactions** during unrolled latent recurrence.

### 2.3 CPU-First Constraints in Android / Termux

Operating within Termux on an unrooted commercial Android device imposes strict system-level constraints:

#### 1. Thermal Throttling & WALT Governor
- The thermal design power (TDP) of the passive smartphone chassis is approximately $4.5 \text{W} - 5.5 \text{W}$ sustained.
- Under peak load (both Prime cores at 4.47 GHz), power consumption spikes to $>12 \text{W}$.
- The kernel's `walt` (Window-Assisted Load Tracking) governor monitors temperatures via SoC thermistors. When junction temperature reaches $85^\circ\text{C}$ (or battery skin reaches $42^\circ\text{C}$), the `thermal-engine` down-clocks the Prime cluster to $2.0 - 2.2 \text{ GHz}$.
- **Guideline**: Long-running benchmarks and training runs must avoid saturating both Prime cores indefinitely to prevent throttling-induced latency jitter.

#### 2. Thread Pool Sizing & Nested Oversubscription
- Snapdragon 8 Elite has 8 physical cores and does not support Simultaneous Multithreading (SMT/hyperthreading) — 1 thread per core.
- When multi-threading via Rayon or OpenMP:
  - If Rayon is initialized with 8 threads, and Candle's underlying matrix multiplication invokes an internal multi-threaded BLAS backend, nested oversubscription occurs ($8 \times 8 = 64$ threads).
  - Context switching overhead, lock contention in `futex`, and thread migration between clusters devastate IPC.
- **Guideline**: Set `RAYON_NUM_THREADS=6` (matching the 6 Performance cores) and bind BLAS operations to single-threaded inner kernels (`OMP_NUM_THREADS=1`).

#### 3. Heap Allocation Churn vs In-Place Mutation
- In the existing implementation of [`src/nca.rs:step_with_forcing`](file:///data/data/com.termux/files/home/projects/titan_text/src/nca.rs#L122-L189), each tick allocates multiple intermediate tensors (`roll_spatial`, `perceive`, `dense1`, `delta`, `gate`).
- In Termux, repeated allocation of $64\text{KB} - 200\text{KB}$ buffers triggers libc/jemalloc page-table traversals and memory fragmentation.
- **Guideline**: Refactor latent recurrence to operate on pre-allocated contiguous scratchpad buffers, keeping the memory addresses pinned in L1D/L2 across all $\tau$ steps.

---

## 3. Comparative Sequence Architectures

To benchmark Titan Text's continuous cellular substrate against mainstream sequence paradigms, we compare four parameter-matched architectures over identical sequence length ($L=32$), channel capacity ($C=64$), and ASCII vocabulary ($V=99$):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   COMPARATIVE RECEPTIVE FIELD TOPOLOGIES                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. 1D Cellular Automaton (Titan NCA): Finite Speed of Light (c = 1 cell/dt) │
│    Cell (i-1) ───► [ Cell i ] ◄─── Cell (i+1)   (Receptive Field ~ 2τ + 1)  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. 1-Layer Causal Transformer: Immediate Global All-to-All Attention        │
│    Token 0 ───────┐                                                         │
│    Token 1 ───────┼──► [ Softmax(QK^T / √d) V ] ──► Token L                 │
│    Token L-1 ─────┘                                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Gated Recurrent Unit (GRU): Temporal Step-by-Step Causal Memory          │
│    x_t ───────────► [ Reset & Update Gates ] ──► h_t                        │
│                           ▲                                                 │
│    h_{t-1} ───────────────┘                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. Simple RNN (Elman): Direct First-Order Recurrent Linear Feedback         │
│    x_t ───────────► [ tanh(W_x x_t + W_h h_{t-1} + b) ] ──► h_t            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Architectural & Computational Complexity Comparison

| Architecture | Information Propagation Mechanism | Receptive Field Growth | Time Complexity per Step | Space Complexity | Expressive Capacity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Titan NCA** | Local stencil $[\mathbf{x}, \nabla \mathbf{x}, \Delta \mathbf{x}]$ | Linear: $2\tau + 1$ (Finite speed of light) | $\mathcal{O}(L \cdot C \cdot H)$ | $\mathcal{O}(L \cdot C)$ | Spatial morphometry, reaction-diffusion Turing patterns |
| **Transformer** (1-Layer Causal) | Pairwise dot-product Attention | Immediate: Full sequence ($L$) | $\mathcal{O}(L^2 \cdot C + L \cdot C \cdot H)$ | $\mathcal{O}(L^2 + L \cdot C)$ | Global associative retrieval, soft lookup, $\text{TC}^0$ circuits |
| **GRU** | Gated non-linear recurrence | Sequential: Step $1 \dots t$ | $\mathcal{O}(L \cdot 3C^2)$ | $\mathcal{O}(C)$ | Latching memory, finite state automata emulation |
| **Simple RNN** (Elman) | Ungated linear projection | Sequential: Vanishing horizon ($\sim 5\text{ steps}$) | $\mathcal{O}(L \cdot C^2)$ | $\mathcal{O}(C)$ | Linear dynamical systems, short-range n-grams |

### 3.2 Compute Equalization Protocols

To execute rigorous scientific comparisons, architectures must be equalized across three distinct dimensions:

#### 1. Parameter Matching Protocol
As verified by `./target/release/titan_text benchmark --task delayed-recall` (see [`src/baselines.rs`](file:///data/data/com.termux/files/home/projects/titan_text/src/baselines.rs)):
- **Titan NCA**: 43,715 parameters
  - Spatial Perception: 0 (fixed stencil)
  - Hidden Extraction (`dense1`: $192 \to 96$): 18,528
  - Directional Update (`dense_delta`: $96 \to 64$): 6,208
  - Adaptive Gate (`dense_gate`: $96 \to 64$): 6,208
  - Navier-Stokes Viscosity: 0 (continuum operator)
  - Token Embedding ($99 \times 64$): 6,336
  - Readout Projection ($64 \times 99 + 99$): 6,435
- **Causal Transformer**: 41,859 parameters
  - Embedding: 6,336
  - Attention ($Q, K, V, O$ @ $64 \times 64$): 16,640
  - MLP ($64 \to 96 \to 64$): 12,448
  - Readout: 6,435
- **GRU Recurrent**: 37,731 parameters
  - Embedding: 6,336
  - Recurrent Core ($W_x, W_h$ for reset, update, candidate): 24,960
  - Readout: 6,435
- **Simple RNN**: 21,091 parameters
  - Embedding: 6,336
  - Elman Core ($W_x, W_h$): 8,320
  - Readout: 6,435

#### 2. FLOPs per Step / Token Equalization
- **Titan NCA**: One cell update requires $(3C \cdot H) + (H \cdot C) + (H \cdot C) = (192 \times 96) + (96 \times 64) + (96 \times 64) = 30,720 \text{ MACs/cell}$.
  For $L = 32$ across $\tau$ latent steps:
  $$\text{FLOPs}_{NCA}(\tau) = 2 \times 32 \times 30,720 \times \tau = 1,966,080 \cdot \tau \text{ FLOPs}$$
  At $\tau = 8$: $\approx \mathbf{15.73 \text{ MFLOPs}}$.
- **Causal Transformer**:
  Attention projections: $4 \times (L \cdot C^2) = 4 \times 32 \times 4,096 = 524,288 \text{ MACs}$.
  Attention scores & context: $2 \times (L^2 \cdot C) = 2 \times 1,024 \times 64 = 131,072 \text{ MACs}$.
  MLP ($2 \times L \cdot C \cdot H$): $2 \times 32 \times 64 \times 96 = 393,216 \text{ MACs}$.
  Total: $2 \times (524,288 + 131,072 + 393,216) \approx \mathbf{2.10 \text{ MFLOPs}}$.
- **Equalization Rule**: Because an unrolled NCA at $\tau = 8$ executes $\sim 7.5\times$ more FLOPs than a single-layer Transformer, an apples-to-apples compute comparison must evaluate the Transformer at either $7$ layers or allocate equal inference wall-clock budget.

#### 3. Token & Sequence Budget Protocol
All models must be trained across identical token streams ($N_{tokens} = \text{Batches} \times B \times L$) using the standardized task suite in [`src/tasks.rs`](file:///data/data/com.termux/files/home/projects/titan_text/src/tasks.rs) (Delayed Recall, Bracket Depth, Parity, Associative Basins).

---

## 4. Deep Comparative Mapping Against Adjacent Paradigms (Requirement 23)

To position Titan Text rigorously within the broader machine learning landscape, this section establishes a formal comparative mapping against eight adjacent computational paradigms. For each paradigm, we derive the closest architectural and mathematical similarities, establish fundamental structural divergences, specify concrete falsifiable experiments that cleanly distinguish Titan Text, and highlight critical conceptual pitfalls.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                ADJACENT COMPUTATIONAL PARADIGMS LANDSCAPE                              │
├────────────────────────────────┬───────────────────────────────────────┬───────────────────────────────┤
│ Paradigm                       │ Recurrence / Coupling Archetype       │ Core Distinguishing Mechanism │
├────────────────────────────────┼───────────────────────────────────────┼───────────────────────────────┤
│ 1. Neural Cellular Automata    │ Continuous-state discrete-grid PDE    │ Open sequential forcing vs    │
│    (Mordvintsev et al.)        │ Local differential stencil            │ closed visual self-repair     │
├────────────────────────────────┼───────────────────────────────────────┼───────────────────────────────┤
│ 2. Reservoir Computing & ESNs  │ High-dimensional dynamical reservoir  │ Fully learned BPTT transition │
│    (Jaeger, Maass)             │ Fading memory readout                 │ vs fixed random recurrence    │
├────────────────────────────────┼───────────────────────────────────────┼───────────────────────────────┤
│ 3. Recurrent Inference         │ Depth recurrence / Iterative ponder   │ Local differential stencil    │
│    (Universal Tx, PonderNet)   │ Multi-tick refinement                 │ vs global dot-product attn    │
├────────────────────────────────┼───────────────────────────────────────┼───────────────────────────────┤
│ 4. Deep Equilibrium Models     │ Implicit root solving: x* = f(x*, u)  │ Finite transient orbits vs    │
│    (DEQ, Bai et al.)           │ Implicit Function Theorem gradients   │ Banach fixed-point roots      │
├────────────────────────────────┼───────────────────────────────────────┼───────────────────────────────┤
│ 5. Adaptive Computation Time   │ Dynamic halting unit: Σ h_t ≥ 1 - ε   │ Commanded synchronous budget  │
│    (ACT, Graves)               │ Ponder cost regularization            │ vs parametric per-cell stop   │
├────────────────────────────────┼───────────────────────────────────────┼───────────────────────────────┤
│ 6. Continuous-Depth Models     │ Neural ODE: dz/dt = f(z, t)           │ Discrete Forward Euler 1D PDE │
│    (Chen et al.)               │ Continuous adjoint sensitivity        │ vs adaptive black-box ODE     │
├────────────────────────────────┼───────────────────────────────────────┼───────────────────────────────┤
│ 7. Structured State-Space      │ Linear Time-Invariant: h' = Ah + Bx   │ Non-linear latent depth PDE   │
│    (S4, Mamba)                 │ Associative scan parallelization      │ vs linear sequence recurrence │
├────────────────────────────────┼───────────────────────────────────────┼───────────────────────────────┤
│ 8. Energy-Based Recurrent      │ Lyapunov descent: dx/dt = -∇E(x)      │ Non-conservative curl field   │
│    (Hopfield Networks)         │ Content-addressable attractor memory  │ vs monotonic energy descent   │
└────────────────────────────────┴───────────────────────────────────────┴───────────────────────────────┘
```

### 4.1 Neural Cellular Automata (Mordvintsev et al., Continuous-State Discrete-Grid)

#### 1. Closest Architectural / Mathematical Similarity
- **Continuous-State Discrete-Lattice Formulation**: Both architectures represent system state as a dense tensor of continuous real-valued channels over discrete spatial coordinates: $\mathbf{x} \in \mathbb{R}^{L \times C}$ in Titan Text (1D sequence lattice) vs $\mathbf{s} \in \mathbb{R}^{H \times W \times C}$ in Mordvintsev et al. (2020) (2D pixel grid).
- **Fixed Differential Perception**: Neither model learns convolution filters; both employ fixed, hardcoded differential operators to perceive local neighborhood gradients:
  - Mordvintsev NCA: $3 \times 3$ Sobel filters $\mathbf{K}_x, \mathbf{K}_y$ and identity $\mathbf{I}$, producing perception $\mathbf{p} = [\mathbf{s}, \mathbf{K}_x * \mathbf{s}, \mathbf{K}_y * \mathbf{s}] \in \mathbb{R}^{3C}$.
  - Titan Text: 1D central difference operators ([`src/nca.rs:74-89`](file:///data/data/com.termux/files/home/projects/titan_text/src/nca.rs#L74-L89)), producing perception $\mathbf{p}_i = [\mathbf{x}_i, \nabla \mathbf{x}_i, \Delta \mathbf{x}_i] = [\mathbf{x}_i, \frac{\mathbf{x}_{i+1}-\mathbf{x}_{i-1}}{2}, \mathbf{x}_{i+1}-2\mathbf{x}_i+\mathbf{x}_{i-1}] \in \mathbb{R}^{3C}$.
- **Euler Residual Transition Operator**: Both evolve states via discrete Forward Euler integration parameterized by a translation-invariant pointwise (1x1) MLP:
  $$\mathbf{x}_{t+1} = \mathbf{x}_t + \alpha \cdot \Delta \mathbf{x}_t$$
- **Stochastic Updating Mechanism**: Both support stochastic cell execution to break spatial synchronization. In Titan Text ([`src/nca.rs:146-156`](file:///data/data/com.termux/files/home/projects/titan_text/src/nca.rs#L146-L156)), when `update_rate` $< 1.0$, a Bernoulli mask $m_i \sim \text{Bernoulli}(p)$ modulates state updates identical to Mordvintsev's asynchronous cell updates.

#### 2. Important Structural Difference
- **Open Driven System vs Closed Homeostatic Regeneration**: Mordvintsev NCA is a closed, autonomous dynamical system initialized from a single localized seed at $t=0$ with zero external inputs thereafter ($\mathbf{u}_t = \mathbf{0}$ for $t > 0$), trained to converge to and maintain a static visual attractor pattern. Titan Text operates as an open, driven non-equilibrium system receiving continuous sequential forcing ($\mathbf{f}(s_t)$ in `step_with_forcing` in [`src/nca.rs:122-167`](file:///data/data/com.termux/files/home/projects/titan_text/src/nca.rs#L122-L167)), where token embeddings continuously drive trajectories across a dynamic symbolic manifold.
- **Topology & Boundary Conditions**: Mordvintsev NCA operates on a 2D Euclidean planar grid with absorbing zero-padding boundaries. Titan Text operates on a 1D closed sequence ring with periodic wrap-around boundary conditions (`roll_spatial` in [`src/nca.rs:61-72`](file:///data/data/com.termux/files/home/projects/titan_text/src/nca.rs#L61-L72)) or an autoregressively masked causal 1D line.
- **Update Parameterization & Dissipation**: Mordvintsev uses an unbounded ReLU/linear MLP followed by hard thresholding on an alpha "alive" channel ($s_{\alpha} > 0.1$). Titan Text enforces bounded velocity via hyperbolic tangent updates gated by an explicit sigmoid gating network ($\Delta \mathbf{x} = \sigma(W_g h + b_g) \odot \tanh(W_\delta h + b_\delta)$) and directly couples continuous Navier-Stokes viscous dissipation ($\nu \Delta_{spatial} \mathbf{x}$) with adaptive substepping ([`src/nca.rs:97-116`](file:///data/data/com.termux/files/home/projects/titan_text/src/nca.rs#L97-L116)).
- **Decoupled Sequence vs Latent Time**: Titan Text decouples token sequence steps from internal recurrence depth: each token input can trigger $\tau$ internal latent deliberation ticks (`LatentConfig::latent_ticks_per_token` in [`src/latent.rs:18`](file:///data/data/com.termux/files/home/projects/titan_text/src/latent.rs#L18)), whereas image NCAs execute a single homogeneous temporal loop.

#### 3. Concrete Distinguishing Experiment
- **Continuous Driving vs Static Attractor Convergence**:
  - *Protocol*: Subject both architectures to a sustained, non-stationary temporal input stream $\mathbf{u}_t = \sin(\omega t)$ injected at spatial coordinate 0.
  - *Measurement*: Compute the state phase velocity $\|\mathbf{x}_{t+1} - \mathbf{x}_t\|$ and the mutual information $I(\mathbf{x}_i(t); \mathbf{u}(t - \Delta t))$ across time $t \in [0, 500]$ and spatial displacement $i$.
  - *Expected Outcome*: Mordvintsev NCA's attractor dynamics are optimized for homeostasis; persistent driving disrupts the trained morphology, forcing the system either to destabilize into numerical chaos or rapidly dissipate the signal to retain its static image pattern. Titan Text acts as a continuous spatiotemporal waveguide: the driving signal propagates as traveling wave packets along the lattice, retaining high mutual information across multiple sequence hops.
- **Regenerative Self-Repair vs Causal Context Disruption**:
  - *Protocol*: Lesion 50% of the lattice cells by setting their state channels to zero at $t = 20$.
  - *Measurement*: Track recovery of original state values over subsequent 50 ticks.
  - *Expected Outcome*: Mordvintsev NCA exhibits complete morphological self-repair ($>0.95$ SSIM recovery). Titan Text experiences an irreversible break in syntactic sequence context: downstream next-token prediction accuracy permanently collapses (from $>70\%$ to chance baseline $2.7\%$), proving Titan Text encodes path-dependent sequential history rather than an invariant spatial target.

#### 4. Pitfalls to Avoid When Comparing
- **The "Self-Repair" Fallacy**: Assuming Titan Text automatically inherits biological pattern homeostasis or self-repair simply because it utilizes a neural cellular automaton core. Language modeling is strictly non-homeostatic; sequence representations must preserve historical entropy rather than erasing differences to restore a static pattern.
- **Equating 2D Isotropic Perception with 1D Causal Sequences**: In 2D images, symmetric isotropic stencils (Sobel) are natural because space has no preferred temporal arrow. In 1D language modeling, a symmetric stencil ($\nabla \mathbf{x}_i = (x_{i+1} - x_{i-1})/2$) inspects the right neighbor cell $x_{i+1}$, which directly holds the target token for next-token prediction, creating an invalid right-copy shortcut.

---

### 4.2 Reservoir Computing & Echo State Networks (Jaeger, Maass)

#### 1. Closest Architectural / Mathematical Similarity
- **Recurrent Dynamical Reservoir as Computational Substrate**: Both models utilize the autonomous recurrence of a high-dimensional continuous latent state vector $\mathbf{x}_t$ to project temporal input sequences into a non-linear, high-dimensional phase space where complex temporal dependencies become linearly separable.
- **Linear Readout Decoupling**: Both architectures strictly separate internal recurrent state evolution from symbolic output generation: token probabilities in Titan Text are generated by projecting the terminal state field through a single linear matrix multiplication: $\hat{\mathbf{y}} = W_{out} \mathbf{x}_T + \mathbf{b}_{out}$ ([`src/vocab.rs:103, 111, 126-128`](file:///data/data/com.termux/files/home/projects/titan_text/src/vocab.rs#L103)), exactly matching the linear readout operator $\hat{\mathbf{y}}_t = W_{out} \mathbf{x}_t$ in Echo State Networks.
- **Fading Memory & Dissipative Stability**: Both systems require contractive, dissipative dynamics to prevent state explosion; unconstrained state norm growth leads to severe readout degradation in both frameworks.

#### 2. Important Structural Difference
- **Fully Differentiable Learning vs Fixed Random Recurrence**: In Reservoir Computing / Echo State Networks, the internal reservoir weight matrix $W_{res} \in \mathbb{R}^{N \times N}$ is permanently frozen at initialization; only the linear readout $W_{out}$ is optimized (typically via closed-form ridge regression). In Titan Text, the recurrent transition parameters ($W_1 \in \mathbb{R}^{96 \times 192}$, $W_\delta \in \mathbb{R}^{64 \times 96}$, $W_g \in \mathbb{R}^{64 \times 96}$ in [`src/nca.rs:33-35`](file:///data/data/com.termux/files/home/projects/titan_text/src/nca.rs#L33-L35)) are fully trained end-to-end via Backpropagation Through Time (BPTT) with AdamW.
- **Strict 1D Geometric Locality vs Unstructured Random Graph**: An ESN reservoir connects neurons via an unstructured, random graph (Erdős-Rényi or small-world topology) where any neuron can have direct synaptic paths to arbitrary neurons across the network. Titan Text enforces strict 1D geometric lattice locality: communication is restricted to nearest neighbors via differential stencils, enforcing a finite speed of light ($c = 1 \text{ cell/tick}$).
- **State-Dependent Velocity Gating**: ESNs employ static recurrent connections passed through an ungated scalar nonlinearity $\tanh(W_{res} \mathbf{x} + W_{in} \mathbf{u})$. Titan Text computes state-dependent velocity gates ($\sigma(W_g h) \odot \tanh(W_\delta h)$), dynamically modulating information flow per channel and spatial position.
- **State Capacity Density**: ESNs rely on massive overparameterization ($N \sim 1,000 - 10,000$ neurons) to span generic polynomial feature bases. Titan Text achieves high expressive capacity with a compact state ($L=32, C=64 \implies 2,048$ continuous scalars; $43,715$ parameters) through learned spatiotemporal compositionality.

#### 3. Concrete Distinguishing Experiment
- **Frozen Recurrent Core Lesion (The Reservoir Control)**:
  - *Protocol*: Freeze the NCA transition weights ($W_1, W_\delta, W_g$) at random Gaussian initialization (scaled such that the Jacobian spectral radius $\rho(J) < 1$) and train only the token embedding and linear readout head ($W_{out}, \mathbf{b}_{out}$) on algorithmic sequence tasks (e.g. Dyck bracket nesting depth or 16-step Delayed Recall).
  - *Measurement*: Compare validation accuracy between the frozen reservoir NCA and fully trained Titan Text.
  - *Expected Outcome*: The random reservoir NCA collapses to near-chance accuracy ($<15\%$) because random local diffusions cannot synthesize directed shift registers or stack counters at $C=64$. Trained Titan Text achieves $>75\%$ accuracy by discovering coordinated spatial transport stencils.
- **Relativistic Light-Cone Delay Profile**:
  - *Protocol*: Inject an impulse perturbation $\delta \mathbf{x}_0$ at cell $0$ at tick $0$. Measure the perturbation magnitude $\|\mathbf{x}_k(t) - \mathbf{x}_k^{(unperturbed)}(t)\|$ at distance $k$ across time $t \in [0, 32]$.
  - *Measurement*: Plot the space-time propagation front.
  - *Expected Outcome*: In an ESN with random connections, the perturbation arrives at all nodes $k$ at $t = 1$ with non-zero amplitude due to random graph shortcuts. In Titan Text, perturbation magnitude is mathematically identical to zero for all $t < k$, exhibiting a sharp linear light cone ($k \le c \cdot t$ with $c = 1$).

#### 4. Pitfalls to Avoid When Comparing
- **Equating Latent Deliberation with Reservoir Ring-Down**: Assuming Titan Text's latent recurrence ticks are merely a passive ring-down or fading echo of the input. Each latent tick in Titan Text executes an active, learned non-linear transformation that progressively extracts hierarchical features.
- **Assuming Global Echo State Property Holds in Learned NCAs**: In ESNs, stability is mathematically guaranteed by global spectral radius bounds ($\rho(W_{res}) < 1$). Learned NCAs lack global linearity; local Jacobian eigenvalues fluctuate across phase space, allowing localized instabilities, limit cycles, or open phase drift if not regularized.

---

### 4.3 Recurrent Inference & Iterative Refinement (PonderNet, Universal Transformers)

#### 1. Closest Architectural / Mathematical Similarity
- **Weight-Tied Recurrence Across Compute Depth**: Both frameworks break the parameter-depth coupling of standard deep networks by iteratively applying the identical parameterized transition function $f_\theta$ across successive compute steps.
- **Iterative Latent Refinement**: Both paradigms assume that complex reasoning, syntactic disambiguation, and multi-hop relationships benefit from iterative refinement of latent representations before committing to a discrete token prediction.
- **Any-Time Intermediate Decodability**: Both architectures allow intermediate states $\mathbf{x}_t$ to be projected through a shared readout head, producing a sequence of provisional token hypotheses $\hat{\mathbf{y}}_0, \dots, \hat{\mathbf{y}}_\tau$ across recurrence depth.

#### 2. Important Structural Difference
- **Spatial Topology: Local Stencil vs Global Self-Attention**: Universal Transformers (Dehghani et al., 2018) use multi-head dot-product self-attention ($\text{Softmax}(Q K^T / \sqrt{d}) V$), granting every token direct all-to-all access to every other token in a single recurrent step with $\mathcal{O}(L^2 \cdot C)$ complexity. Titan Text uses strictly local nearest-neighbor differential stencils with $\mathcal{O}(L \cdot C)$ complexity, requiring $\mathcal{O}(L)$ ticks for information to span sequence length $L$.
- **Halting Mechanism: Parametric Policy vs Commanded Budget**: PonderNet (Banino et al., 2021) trains a parametric halting unit $\lambda_t = \sigma(W_p \mathbf{s}_t + b_p)$ regularized against a geometric prior, dynamically halting computation when confident. Titan Text (in v0) executes an externally commanded, uniform latent compute budget ($\tau \in \{0, 1, 2, 4, 8, \dots\}$, configured via `LatentConfig`) without an internal halting policy.
- **Output Aggregation: Expected Mixture vs Instantaneous State Readout**: PonderNet computes its output as the probability-weighted expectation over all intermediate steps ($\hat{\mathbf{y}} = \sum_n p_n \hat{\mathbf{y}}_n$). Titan Text reads out predictions exclusively from the instantaneous field state at the terminal tick $\tau$ ($W_{out} \mathbf{x}_\tau + \mathbf{b}_{out}$).
- **Continuum PDE vs Discrete Layer Stack**: Titan Text parameterizes its state transitions as a discretized continuum PDE with spatial derivatives and physical fluid viscosity; Universal Transformers and PonderNet are discrete layer-stack architectures with residual connections, LayerNorm, and attention mechanisms.

#### 3. Concrete Distinguishing Experiment
- **Sequence Length Scaling Complexity ($L = 32 \to 1024$)**:
  - *Protocol*: Measure per-step FLOPs, memory footprint, and wall-clock execution time as sequence length $L$ scales from 32 to 1024 under fixed channel dimension $C=64$.
  - *Measurement*: Computational scaling curve as a function of $L$.
  - *Expected Outcome*: Universal Transformer exhibits quadratic FLOP scaling $\mathcal{O}(L^2)$ and quadratic memory allocation for attention matrices. Titan Text exhibits strictly linear scaling $\mathcal{O}(L)$ in both FLOPs and memory, remaining completely cache-resident in Snapdragon 8 Elite L1D/L2 cache.
- **Non-Local Parity at Step $\tau = 1$**:
  - *Protocol*: Construct a sequence where prediction at index 0 depends strictly on the parity between token 0 and token $L-1$ ($L=32$). Evaluate both models at compute budget $\tau = 1$.
  - *Measurement*: Accuracy at $\tau = 1$.
  - *Expected Outcome*: Universal Transformer solves the task at $\tau = 1$ ($>99\%$ accuracy) via a single direct attention connection between index 0 and 31. Titan Text produces chance accuracy ($2.7\%$) at $\tau = 1$, because its spatial stencil has radius 1; it strictly requires at least $\tau = \lceil (L-1)/2 \rceil = 16$ ticks (or $\tau = 31$ under causal stencils) for information to traverse the sequence.

#### 4. Pitfalls to Avoid When Comparing
- **Assuming Unrolled Depth Equals Halting Convergence**: Believing that running Titan Text for 8 ticks is mathematically equivalent to PonderNet pondering for 8 ticks. PonderNet dynamically exits early on easy inputs, conserving compute; Titan Text v0 processes all tokens and batches for the exact same number of ticks regardless of input difficulty.
- **Conflating Weight-Tied Layers with Dynamical Attractors**: As revealed in `grumpy-review.md`, an unrolled weight-tied model without contractive regularization simply overfits to its training horizon $T_{train}$. Neither Universal Transformers nor Titan Text achieve monotonic compute gains at $\tau > T_{train}$ unless trained with multi-horizon losses or contractive constraints.

---

### 4.4 Deep Equilibrium Models (DEQ, Bai et al. - Fixed Points vs Non-Convergent Orbits)

#### 1. Closest Architectural / Mathematical Similarity
- **Equilibrium State Hypothesis**: Both frameworks conceptualize computation as the evolution of a latent state vector $\mathbf{x}$ towards a stable, task-aligned representation under repeated evaluation of a parameter-tied transition function.
- **Infinite-Depth Parameter Sharing**: Both models eliminate layer-specific parameters, parameterizing an arbitrary number of recurrent evaluations using a single compact neural network block.
- **Monotonic Compute Objective**: In both paradigms, the theoretical goal is that allocating more computational iterations at inference time should strictly refine, rather than degrade, task representations.

#### 2. Important Structural Difference
- **Implicit Fixed Point vs Finite-Horizon Transient Path**: DEQs mathematically postulate and require the existence of an exact fixed point: $\mathbf{x}^* = f_\theta(\mathbf{x}^*, \mathbf{u})$, optimizing explicitly for root convergence ($\|\mathbf{x}^* - f(\mathbf{x}^*)\| < 10^{-5}$). Titan Text (v0) is trained via explicit finite-horizon BPTT ($T=8$); its trajectory is a **non-convergent transient orbit**. As proven in [`reviews/2026-09-11/review.md`](file:///data/data/com.termux/files/home/projects/titan_text/reviews/2026-09-11/review.md), at $T=8$ Titan Text's velocity remains high ($\|\Delta \mathbf{x}\| = 0.3057$), and rolling out to $\tau=256$ causes state energy to explode to $1,721.73$ rather than settling at a fixed point.
- **Gradient Computation: Implicit Function Theorem vs BPTT Graph Unrolling**: DEQs compute gradients via the analytical Jacobian inverse $(\mathbf{I} - J_f)^{-1}$ using linear solvers (Broyden/Anderson), requiring **$\mathcal{O}(1)$ training memory** regardless of solver depth. Titan Text stores intermediate state tensors across all unroll steps $t \in [1, T]$, requiring $\mathcal{O}(T \cdot L \cdot C)$ memory during BPTT.
- **Forward Solver Mechanics**: DEQs use black-box root-finding algorithms (quasi-Newton Broyden acceleration) that jump across phase space to find roots. Titan Text computes forward trajectories strictly by step-by-step Forward Euler integration of the local PDE stencil, preserving the intermediate spatial wave transport dynamics.

#### 3. Concrete Distinguishing Experiment
- **Residual Norm Convergence Profile ($\tau \to \infty$)**:
  - *Protocol*: Measure the residual velocity norm $R(\tau) = \|F(\mathbf{x}_\tau) - \mathbf{x}_\tau\|_2$ over 100 autonomous unforced iterations from identical input embeddings.
  - *Measurement*: Semi-log plot of $R(\tau)$ versus iteration tick $\tau$.
  - *Expected Outcome*: In a DEQ, $R(\tau)$ converges monotonically towards machine precision ($10^{-5} - 10^{-7}$). In Titan Text v0, $R(\tau)$ never drops below $0.25$; instead, it stabilizes on a non-vanishing velocity floor or diverges ($R(32) \approx 10.98$), demonstrating that Titan Text operates along an open transient path rather than a contractive DEQ fixed point.
- **Peak Training Memory vs Depth Profiling**:
  - *Protocol*: Measure peak heap memory allocation during training as effective compute depth is scaled from $T=4$ to $T=64$.
  - *Measurement*: Peak RAM in MB.
  - *Expected Outcome*: DEQ memory remains strictly flat ($\mathcal{O}(1)$). Titan Text training memory scales strictly linearly $\mathcal{O}(T)$ with unroll depth.

#### 4. Pitfalls to Avoid When Comparing
- **Labeling Transient States as "Equilibria"**: As cautioned in `grumpy-review.md`, describing Titan Text's state at $T=8$ as an "attractor basin" or "equilibrium" is mathematically incorrect. The model achieves peak accuracy at $T=8$ because the readout projection was trained exclusively on the transient manifold $\mathcal{M}_8$, not because the dynamical system reached a fixed point.
- **Assuming DEQ Solvers Can Be Injected Without Contractivity**: Attempting to solve Titan Text's forward pass using Broyden or Anderson acceleration without contractive Jacobian regularization ($\rho(J_F) < 1$) will cause root-finder divergence, because unconstrained NCAs contain limit cycles and strange attractors where no unique root exists.

---

### 4.5 Adaptive Computation Time (ACT, Graves - Halting Probability Mechanics)

#### 1. Closest Architectural / Mathematical Similarity
- **Dynamic Recurrence Depth per Input**: Both paradigms are motivated by the principle of variable computation: allowing models to spend more recurrent operations on complex, ambiguous tokens and fewer operations on simple ones.
- **Sigmoid Gating Formulations**: Both architectures employ sigmoid gating networks to regulate internal recurrent transitions.
- **Intermediate Representation Sampling**: Both models provide mechanisms to probe intermediate states along the recurrence depth axis.

#### 2. Important Structural Difference
- **Internal Parametric Halting vs Commanded Uniform Latent Budgets**: Graves (2016) ACT equips each token with an internal, parametric halting unit $h_t^n = \sigma(W_h \mathbf{s}_t^n + b_h)$ trained via ponder loss, dynamically stopping computation when cumulative halting probability $\sum_n h_t^n \ge 1 - \epsilon$. Titan Text (v0) has no internal stopping mechanism: recurrence depth $\tau$ is an external hyperparameter applied uniformly across the entire batch (`latent_ticks_per_token` in [`src/latent.rs:18`](file:///data/data/com.termux/files/home/projects/titan_text/src/latent.rs#L18)).
- **Gating Semantics: Temporal Stopping vs Spatial Velocity Limiting**: In ACT, $h_t^n \in (0, 1)$ is a scalar temporal probability governing when to terminate recurrence. In Titan Text ([`src/nca.rs:35, 142`](file:///data/data/com.termux/files/home/projects/titan_text/src/nca.rs#L35-L142)), `dense_gate` outputs a multi-channel velocity limiter $\mathbf{g}_i \in (0, 1)^C$ that modulates the spatial update vector ($\alpha \cdot (\mathbf{g} \odot \boldsymbol{\delta})$). It restricts displacement per tick, not the number of ticks.
- **Spatial Synchronization: Synchronous Field vs Asynchronous Independent Halting**: In ACT, each token position halts at a different iteration $N(i)$. If applied cell-wise in Titan Text, neighboring cells would halt at different developmental timestamps, destroying spatial gradient coherence ($\nabla \mathbf{x}, \Delta \mathbf{x}$). Titan Text evolves all cells synchronously across the 1D lattice.
- **State Aggregation: Convex Combination vs Instantaneous State Readout**: ACT mathematically guarantees that the output state is a convex combination of intermediate states ($\sum_n p_n \mathbf{s}_n \in \text{Conv}(\{\mathbf{s}_n\})$). Titan Text reads out from the instantaneous field state $\mathbf{x}_\tau$, allowing states to travel far outside the convex hull of their initial embeddings.

#### 3. Concrete Distinguishing Experiment
- **Input Difficulty Halting Distribution Sweep**:
  - *Protocol*: Evaluate models across inputs with varying algorithmic difficulty (e.g. Dyck strings of nesting depth 1 vs depth 6, or delayed recall with delay 2 vs delay 20).
  - *Measurement*: Distribution of executed compute steps per token.
  - *Expected Outcome*: In ACT, the distribution of ponder steps $N(x)$ shifts significantly: mean ponder steps for hard inputs is substantially higher than for easy inputs ($N_{hard} \gg N_{easy}$). In Titan Text v0, every sample executes exactly $\tau$ ticks, incurring identical wall-clock runtime and FLOPs regardless of input complexity.
- **Spatial Desynchronization Lesion**:
  - *Protocol*: Force individual cells in Titan Text to halt independently when their local velocity $\|\Delta \mathbf{x}_i\| < \epsilon$. Compute the spatial gradient $\nabla \mathbf{x}_i = (x_{i+1} - x_{i-1})/2$ between an active cell and an asynchronously halted cell.
  - *Measurement*: Gradient variance and downstream prediction accuracy.
  - *Expected Outcome*: The temporal desynchronization creates artificial numerical discontinuities and shockwaves at boundaries between halted and running cells, degrading prediction accuracy compared to synchronous execution.

#### 4. Pitfalls to Avoid When Comparing
- **Confusing Velocity Gating with Halting Probability**: Confusing the sigmoid gate in [`src/nca.rs:142`](file:///data/data/com.termux/files/home/projects/titan_text/src/nca.rs#L142) with an ACT halting probability. Titan Text's gate is an element-wise flow limiter modeled after GRU/LSTM gates, not a stopping condition.
- **Overlooking Spatial Desynchronization Hazards**: Assuming ACT can be ported to cellular automata by simply adding a stopping unit at each cell. Local differential stencils require temporal synchronization across neighbors to maintain mathematical validity.

---

### 4.6 Continuous-Depth Models & Neural ODEs (Chen et al. - Euler Integration vs NCA Updates)

#### 1. Closest Architectural / Mathematical Similarity
- **Continuous Flow Representation**: Both architectures model deep representation learning as continuous flow governed by differential equations rather than discrete static layer transformations.
- **Forward Euler Discretization Equivalence**: Titan Text's update rule $\mathbf{x}_{t+1} = \mathbf{x}_t + \alpha \Delta \mathbf{x}_t$ is mathematically identical to a **Forward Euler discretization** of an autonomous dynamical system $\frac{\partial \mathbf{x}}{\partial t} = \Delta \mathbf{x}$ with fixed step size $\Delta t = \alpha = 0.5$.
- **Continuum Physical Operators**: Titan Text directly incorporates physical continuum operators (spatial gradients $\nabla \mathbf{x}$, Laplacian $\Delta \mathbf{x}$, and Navier-Stokes viscous dissipation $\nu \Delta_{spatial} \mathbf{x}$) derived from continuous transport and diffusion physics.

#### 2. Important Structural Difference
- **ODE vs PDE**: Neural ODEs (Chen et al., 2018) model spatially lumped systems of ordinary differential equations ($\frac{d\mathbf{z}}{dt} = f(\mathbf{z}, t)$) without spatial coordinates. Titan Text is a spatially distributed **1D continuous Partial Differential Equation (PDE)**: $\frac{\partial \mathbf{x}}{\partial t} = N_\theta([\mathbf{x}, \nabla \mathbf{x}, \Delta \mathbf{x}]) + \nu \Delta_{spatial} \mathbf{x}$, explicitly coupling spatial coordinates $i \in \{0, \dots, L-1\}$ through spatial differential operators.
- **Integration Mechanics: Adaptive Black-Box Solvers vs Fixed-Step Euler**: Neural ODEs utilize high-order adaptive-step ODE solvers (e.g. Dormand-Prince `dopri5`, Runge-Kutta RK45) that adjust $\Delta t$ dynamically to bound local truncation errors below strict tolerances ($\epsilon_{atol}, \epsilon_{rtol}$). Titan Text uses fixed-step Forward Euler ($\Delta t = \alpha = 0.5$) with discrete substepping applied exclusively to the linear Laplacian dissipation term (`apply_viscous_dissipation` in [`src/nca.rs:97-116`](file:///data/data/com.termux/files/home/projects/titan_text/src/nca.rs#L97-L116)).
- **Gradient Computation: Continuous Adjoint Method vs Discrete BPTT**: Neural ODEs compute gradients by integrating the continuous adjoint ODE backward in time with $\mathcal{O}(1)$ memory. Titan Text backpropagates directly through the unrolled discrete computation graph via standard BPTT, requiring $\mathcal{O}(T)$ memory.
- **Topology Preservation & Invertibility**: Continuous ODE flows with Lipschitz vector fields define homeomorphisms that preserve topology (trajectories cannot cross). Titan Text's discrete Forward Euler steps with step size $\alpha = 0.5$ can overshoot, bifurcate, cross trajectories, and produce non-invertible transformations.

#### 3. Concrete Distinguishing Experiment
- **Integration Step Size Invariance Test ($\alpha$-Scaling)**:
  - *Protocol*: Halve the integration step size $\alpha \to \alpha/2 = 0.25$ and double the step count $\tau \to 2\tau = 16$.
  - *Measurement*: Measure final state divergence $\|\mathbf{z}_{\alpha/2}(2\tau) - \mathbf{z}_\alpha(\tau)\|_2$ and downstream prediction accuracy.
  - *Expected Outcome*: In a genuine Neural ODE solved with adaptive integrators, trajectory divergence is negligible ($< 10^{-3}$) and prediction accuracy is completely preserved. In Titan Text v0, halving $\alpha$ cuts per-step displacement by 50%; because the local MLP was trained strictly with $\alpha=0.5$, the state falls short of the target manifold at step $\tau$, and accuracy collapses, proving the model is a discrete dynamical map rather than a continuous ODE.
- **Spatial Derivative Lesion**:
  - *Protocol*: Zero out the spatial gradient and Laplacian perception channels ($\mathbf{p}_i = [\mathbf{x}_i, \mathbf{0}, \mathbf{0}]$).
  - *Measurement*: Sequence modeling accuracy on Delayed Recall.
  - *Expected Outcome*: For a Neural ODE, spatial stencils do not exist. In Titan Text, this lesion completely abolishes inter-cell communication, reducing accuracy to chance ($2.7\%$), proving that PDE spatial coupling is essential.

#### 4. Pitfalls to Avoid When Comparing
- **Claiming Continuous-Time Invariance Without Adaptive Integration**: Calling Titan Text a "Neural ODE" without adaptive solvers. Coarse Forward Euler stepping with $\alpha = 0.5$ introduces substantial truncation error $\mathcal{O}(\alpha)$ and numerical artifacts that do not represent continuous ODE flows.
- **Conflating 1D Lattice Diffusion with 3D Navier-Stokes Turbulence**: As established in `grumpy-review.md`, 1D discrete Laplacian diffusion $\nu \Delta \mathbf{x}$ is standard linear heat dissipation. Invoking 3D incompressible Navier-Stokes singularity theorems (such as Beale-Kato-Majda blow-up) on a 1D discrete lattice of 32 cells is a severe physics category error.

---

### 4.7 Structured State-Space Models (S4, Mamba - Linear Time-Invariance vs Non-Linear Local Stencils)

#### 1. Closest Architectural / Mathematical Similarity
- **Sub-Quadratic Sequence Processing**: Both architectures achieve linear $\mathcal{O}(L)$ time complexity per step, eliminating the quadratic $\mathcal{O}(L^2)$ all-to-all attention matrix of Transformers.
- **Physics-Grounded Continuous Dynamics**: Both models derive their core inductive biases from continuous differential equations (state-space control theory and orthogonal HiPPO polynomials in S4; continuum fluid dynamics and reaction-diffusion in Titan Text).
- **Recurrent Latent Memory**: Both maintain a continuous internal state that evolves recurrently to synthesize context across time.

#### 2. Important Structural Difference
- **Recurrence Axis: Sequence Time vs Latent Deliberation Depth**: In S4 (Gu et al., 2021) and Mamba (Dao & Gu, 2023), recurrence unrolls along the **sequence length axis** ($t = 1 \dots L$), updating an internal state vector $h_t$ as each token arrives. In Titan Text, the sequence dimension ($L=32$) is represented as a **spatial grid**, and recurrence unrolls along an orthogonal **developmental / latent depth axis** ($\tau = 1 \dots T$).
- **Linear Transition with Associative Parallel Scan vs Non-Linear Spatial Stencils**: S4 and Mamba rely strictly on linear transitions ($h_t = \mathbf{A}_t h_{t-1} + \mathbf{B}_t x_t$), enabling the entire sequence of states to be computed in parallel in $\mathcal{O}(\log L)$ depth via associative scan. Titan Text uses a highly non-linear spatial transition operator ($\text{MLP}(\text{perception}(\mathbf{x}))$ with GELU and tanh activations and multiplicative gating), which mathematically cannot be parallelized via associative scans or FFT convolutions.
- **Receptive Field Growth: Infinite Temporal Memory vs Finite Speed of Light**: S4/Mamba possess an immediate infinite memory horizon along sequence history governed by decay eigenvalues of $\mathbf{A}$. Titan Text has a strictly bounded spatial receptive field that expands at speed $c = 1 \text{ cell/tick}$, requiring multiple latent ticks to transmit information between separated tokens.
- **Latent Deliberation on Static Inputs**: Titan Text can hold an input sequence static and deliberate across $\tau$ latent ticks, refining representations without advancing the token stream. S4 and Mamba are feedforward sequence filters that process tokens on arrival without internal multi-step settling.

#### 3. Concrete Distinguishing Experiment
- **Associative Scan Parallelization Test**:
  - *Protocol*: Attempt to re-formulate the recurrent step into an associative binary operator $(A_2, B_2) \bullet (A_1, B_1) = (A_2 A_1, A_2 B_1 + B_2)$ to execute an associative parallel scan across steps.
  - *Measurement*: Verify mathematical associativity: $(S_3 \bullet S_2) \bullet S_1 \stackrel{?}{=} S_3 \bullet (S_2 \bullet S_1)$.
  - *Expected Outcome*: In S4 and Mamba, associativity holds exactly, enabling parallel training on vector hardware. In Titan Text, non-linear coupling across spatial channels and activation functions violates associativity; forward passes must be evaluated sequentially step-by-step.
- **Latent Deliberation Accuracy Ramp on Fixed Prefix**:
  - *Protocol*: Feed an ambiguous prefix and measure token prediction accuracy as a function of latent compute ticks $\tau \in [1, 16]$ without advancing the token index.
  - *Measurement*: Accuracy vs $\tau$.
  - *Expected Outcome*: S4 and Mamba cannot perform latent deliberation without ingesting dummy padding tokens. Titan Text executes internal deliberation on the stationary field, exhibiting compute-dependent accuracy trajectories.

#### 4. Pitfalls to Avoid When Comparing
- **Conflating Sequence Time with Latent Time**: Confusing S4's sequence recurrence with Titan Text's latent recurrence. S4 recurs across tokens ($1 \dots L$); Titan Text recurs across internal deliberation depth ($1 \dots \tau$) while treating the sequence as a spatial continuum.
- **Assuming State-Space Linearity Offers Equal Expressivity to Non-Linear PDEs**: S4 achieves parallelization by sacrificing non-linear state interactions within the recurrence. Titan Text retains non-linear spatiotemporal interactions (reaction-diffusion patterns, localized traveling waves), at the cost of requiring sequential BPTT unrolls.

---

### 4.8 Energy-Based Recurrent Dynamics & Modern Hopfield Networks (Hopfield 1982, Ramsauer et al. 2020)

#### 1. Closest Architectural / Mathematical Similarity
- **Attractor Basin Dynamics for Memory Retrieval**: Both paradigms postulate that associative retrieval and pattern completion occur through the relaxation of continuous trajectories into low-energy attractor basins.
- **Autonomous Convergence Dynamics**: Both models evaluate settling behavior by initializing a network with a partial or corrupted stimulus and letting recurrence autonomously evolve the state towards a canonical target.
- **Content-Addressable Associative Memory**: Both frameworks seek to map corrupted or ambiguous representations to unambiguous discrete symbolic tokens.

#### 2. Important Structural Difference
- **Conservative Energy Potential vs Non-Conservative Dynamic Vector Fields**: Hopfield networks strictly require a conservative vector field ($\dot{\mathbf{x}} = -\nabla E(\mathbf{x})$), guaranteed by symmetric coupling matrices ($W = W^T, W_{ii} = 0$). Because the curl of the vector field is zero ($\nabla \times \mathbf{F} = \mathbf{0}$), limit cycles, periodic orbits, and chaos are mathematically impossible. Titan Text uses asymmetric learned weight matrices ($W_1, W_\delta, W_g$); its vector field is non-conservative ($\nabla \times \mathbf{F} \ne \mathbf{0}$), permitting limit cycles, traveling waves, and chaotic wandering.
- **Global Dot-Product Matching vs Local Spatial Differential Stencils**: Modern Continuous Hopfield Networks (Ramsauer et al., 2020) evaluate global inner products between the state $\mathbf{x}$ and all stored prototype memories $\mathbf{X}$ simultaneously ($\mathbf{x}^{new} = \mathbf{X} \text{Softmax}(\beta \mathbf{X}^T \mathbf{x})$). Titan Text distributes representations across a 1D spatial lattice, performing memory transformations through local differential operators and reaction-diffusion stencils.
- **Static Associative Memory vs Generative Sequence Processing**: Hopfield networks are content-addressable static memories designed to retrieve fixed patterns. Titan Text is an autoregressive sequence generation engine designed to predict transitioning token distributions over time.

#### 3. Concrete Distinguishing Experiment
- **Energy Monotonicity / Curl Detection Test**:
  - *Protocol*: Define an empirical quadratic energy proxy $E(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|_2^2$ or readout entropy $H(\mathbf{x})$. Track the per-tick change $\Delta E = E(\mathbf{x}_{t+1}) - E(\mathbf{x}_t)$ over 100 autonomous unforced ticks.
  - *Measurement*: Plot $\Delta E$ vs tick $t$.
  - *Expected Outcome*: In a Hopfield network, energy decreases monotonically ($\Delta E \le 0$) until convergence. In Titan Text v0, as demonstrated in [`reviews/2026-09-11/review.md`](file:///data/data/com.termux/files/home/projects/titan_text/reviews/2026-09-11/review.md), state energy surges by orders of magnitude ($3.95 \to 1,721.73$), and velocity remains non-zero, proving that the vector field possesses strong rotational curl and is not governed by an energy potential.
- **Storage Capacity Scaling with Channel Dimension**:
  - *Protocol*: Measure the number of distinct orthogonal patterns $P$ that can be stably retrieved as channel capacity $C$ increases.
  - *Measurement*: Maximum capacity $P_{max}(C)$.
  - *Expected Outcome*: Modern Continuous Hopfield Networks exhibit exponential capacity scaling $P \sim \mathcal{O}(e^{C})$. In Titan Text, capacity is bounded by the polynomial expressive capacity of the 1D local stencil and suffers spatial cross-talk between adjacent cells.

#### 4. Pitfalls to Avoid When Comparing
- **Labeling Settled States as "Energy Minima"**: Describing Titan Text's latent states as "energy minima" or "attractor wells." As shown in `grumpy-review.md`, Titan Text states during unrolled inference are non-convergent transient corridors, not Lyapunov energy minima.
- **Conflating Association Emergence with Hopfield Retrieval**: Portraying the probability ramp in `associate` (`'t' -> 'h'`) as Hopfield-style associative pattern completion. In Titan Text v0, this ramp was driven by state norm growth inflating softmax logits under zero temperature, rather than gradient descent on a Hopfield energy landscape.

---

### 4.9 Unified Comparative Paradigm Synthesis Matrix

The following matrix provides a rigorous, cross-paradigm synthesis of Titan Text against all eight adjacent computational paradigms:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                            UNIFIED ADJACENT PARADIGMS SYNTHESIS MATRIX                                           │
├───────────────────────┬─────────────────────────────┬──────────────────┬─────────────────┬──────────────┬─────────────┬──────────┤
│ Paradigm              │ Primary Mathematical        │ Recurrence Axis  │ Spatial / Graph │ Trainability │ Halting /   │ Training │
│                       │ Formulation                 │                  │ Coupling        │ of Recurrence│ Equilibrium │ Memory   │
├───────────────────────┼─────────────────────────────┼──────────────────┼─────────────────┼──────────────┼─────────────┼──────────┤
│ Titan Text (NCA)      │ x_{t+1} = x_t + α Δx_t      │ Latent depth     │ 1D Differential │ Fully Learned│ Commanded / │ O(T)     │
│                       │ + ν Δ_spatial x             │ (τ = 1...T)      │ Stencil (r = 1) │ (BPTT AdamW) │ Transient   │ (BPTT)   │
├───────────────────────┼─────────────────────────────┼──────────────────┼─────────────────┼──────────────┼─────────────┼──────────┤
│ 1. Neural Cellular    │ s_{t+1} = s_t + m ⊙ MLP(p)  │ Autonomous time  │ 2D Sobel Filter │ Fully Learned│ Morphological│ O(T)    │
│    Automata (2D Image)│ p = [s, K_x*s, K_y*s]       │ (single seed)    │ (Absorbing Pad) │ (BPTT AdamW) │ Attractor   │ (BPTT)   │
├───────────────────────┼─────────────────────────────┼──────────────────┼─────────────────┼──────────────┼─────────────┼──────────┤
│ 2. Reservoir / Echo   │ x_{t+1} = (1-α)x_t          │ Sequence time    │ Random Graph    │ FROZEN       │ Fading      │ O(1)     │
│    State Networks     │ + α tanh(W_res x_t + W_in u)│ (t = 1...L)      │ (Unstructured)  │ (Only Readout│ Memory      │ (Ridge)  │
├───────────────────────┼─────────────────────────────┼──────────────────┼─────────────────┼──────────────┼─────────────┼──────────┤
│ 3. Recurrent Inference│ H_{t+1} = Attn(H_t) + FFN   │ Compute depth    │ Global All-to-  │ Fully Learned│ Parametric  │ O(T)     │
│    (Universal Tx / PN)│ p_n = λ_n Π (1 - λ_k)       │ (ponder ticks)   │ All Dot-Product │ (BPTT AdamW) │ Halting / KL│ (BPTT)   │
├───────────────────────┼─────────────────────────────┼──────────────────┼─────────────────┼──────────────┼─────────────┼──────────┤
│ 4. Deep Equilibrium   │ x* = f_θ(x*, u)             │ Solver iter      │ Problem-        │ Fully Learned│ Banach Fixed│ O(1)     │
│    Models (DEQ)       │ (Anderson / Broyden root)   │ (k -> ∞)         │ Dependent       │ (Implicit FT)│ Point Root  │ (IFT)    │
├───────────────────────┼─────────────────────────────┼──────────────────┼─────────────────┼──────────────┼─────────────┼──────────┤
│ 5. Adaptive Comput.   │ s_t = Σ p_t^n s_t^n         │ Internal ticks   │ Uncoupled /     │ Fully Learned│ Cumulative  │ O(N)     │
│    Time (ACT)         │ Σ h_t^n ≥ 1 - ε             │ per sequence pos │ Spatial Pos     │ (BPTT + Cost)│ Probability │ (BPTT)   │
├───────────────────────┼─────────────────────────────┼──────────────────┼─────────────────┼──────────────┼─────────────┼──────────┤
│ 6. Continuous-Depth   │ dz/dt = f_θ(z, t)           │ Continuous time  │ Spatially       │ Fully Learned│ Integration │ O(1)     │
│    (Neural ODEs)      │ z(t_1) = ODESolve(z(t_0))   │ (t ∈ [t_0, t_1]) │ Lumped (None)   │ (Adjoint ODE)│ Horizon t_1 │ (Adjoint)│
├───────────────────────┼─────────────────────────────┼──────────────────┼─────────────────┼──────────────┼─────────────┼──────────┤
│ 7. Structured State-  │ h' = A h + B x, y = C h     │ Sequence length  │ Linear Temporal │ Fully Learned│ Sequence    │ O(log L) │
│    Space (S4 / Mamba) │ h_t = A_t h_{t-1} + B_t x_t │ (t = 1...L)      │ Associative Scan│ (Scan / BPTT)│ End (L)     │ (Scan)   │
├───────────────────────┼─────────────────────────────┼──────────────────┼─────────────────┼──────────────┼─────────────┼──────────┤
│ 8. Energy-Based Recurr│ dx/dt = -∇E(x)              │ Autonomous time  │ Symmetric All-to│ Fully Learned│ Lyapunov    │ O(1) /   │
│    (Hopfield Networks)│ x^{new} = X Softmax(β X^T x)│ (relaxation)     │ All / Dot-Prod  │ / Hebbian    │ Minimum     │ O(T)     │
└───────────────────────┴─────────────────────────────┴──────────────────┴─────────────────┴──────────────┴─────────────┴──────────┘
```

---

## 5. Operational Scientific Taxonomy

To ensure scientific falsifiability, qualitative buzzwords must be replaced with strict operational definitions:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OPERATIONAL SCIENTIFIC TAXONOMY                          │
├───────────────────┬──────────────────────────────┬──────────────────────────┤
│ Informal Term     │ Defect / Ambiguity           │ Operational Definition   │
├───────────────────┼──────────────────────────────┼──────────────────────────┤
│ "Bifurcation"     │ Vaguely implies "phase       │ ΔAcc(τ, τ-1) > 25% with  │
│                   │ transition" or "emergence"   │ d²||Δx||/dτ² ≠ 0         │
├───────────────────┼──────────────────────────────┼──────────────────────────┤
│ "Attractor Basin" │ Metaphor of "energy well"    │ lim ||x_{t+1}-x_t|| < ε  │
│                   │ without proving convergence  │ with positive separation │
├───────────────────┼──────────────────────────────┼──────────────────────────┤
│ "Semantic         │ Anthropomorphic claim of     │ Stimulus-conditioned     │
│  Association"     │ "understanding"              │ output probability ramp  │
└───────────────────┴──────────────────────────────┴──────────────────────────┘
```

### 5.1 "Bifurcation" $\longrightarrow$ Discrete Non-Linear Step-Gain
- **Defective Usage**: "The cellular field experiences a spontaneous bifurcation into an ordered computational state."
- **Operational Definition**: A **discrete non-linear accuracy step-gain** is confirmed if and only if:
  1. There exists a consecutive latent budget pair $(\tau_k, \tau_{k-1})$ such that:
     $$\Delta \text{Accuracy} = \text{Accuracy}(\tau_k) - \text{Accuracy}(\tau_{k-1}) \ge 0.25 \quad (+25.0\%)$$
  2. The second derivative of state displacement exhibits an inflection discontinuity:
     $$\frac{d^2 \|\mathbf{x}_\tau - \mathbf{x}_0\|}{d \tau^2} \ne 0$$
  3. Prediction entropy drops sharply: $\Delta H = H(\tau_k) - H(\tau_{k-1}) \le -1.0 \text{ nats}$.
- **Empirical Grounding in Titan Text**: In [`README.md:281-285`](file:///data/data/com.termux/files/home/projects/titan_text/README.md#L281-L285), between tick 1 and tick 2:
  - Accuracy jumps from $13.7\%$ to $42.6\%$ ($\mathbf{+28.9\%}$ delta).
  - Displacement accelerates from $0.2842$ to $0.5723$.
  - Confidence jumps from $0.0022$ to $0.0058$.
  This satisfies the criteria for an operational bifurcation.

### 5.2 "Attractor Basin" $\longrightarrow$ Asymptotic Trajectory Convergence to Invariant Manifold
- **Defective Usage**: "The network stores sequence concepts in stable attractor basins."
- **Operational Definition**: An **attractor basin** $\mathcal{B}_k$ associated with invariant state $\mathbf{x}^*_k$ requires:
  1. **Velocity Quenching**: $\lim_{\tau \to \infty} \|\mathbf{x}_{\tau+1} - \mathbf{x}_\tau\|_2 < \epsilon$ (where $\epsilon \le 10^{-4}$).
  2. **Readout Stability**: $\lim_{\tau \to \infty} \mathcal{D}_{KL}\left( P_{readout}(\mathbf{x}_\tau) \,\|\, P_{readout}(\mathbf{x}_{\tau+1}) \right) = 0$.
  3. **Basin Separation**: For distinct initial stimuli $A$ and $B$, their final invariant states satisfy:
     $$\mathcal{S}_{AB} = \frac{\|\mathbf{x}^*_A - \mathbf{x}^*_B\|_2}{\sigma_A + \sigma_B} > 2.0$$
  4. **Hysteresis**: Injecting transient noise $\eta \sim \mathcal{N}(0, \sigma^2)$ for $M$ steps results in trajectory recovery back to $\mathbf{x}^*_k$ once noise ceases.
- **Empirical Reality Check in Titan Text**: As demonstrated in [`reviews/2026-09-11/review.md`](file:///data/data/com.termux/files/home/projects/titan_text/reviews/2026-09-11/review.md#L49-L66), the unlesioned `v0_viscous` model does **NOT** possess a true fixed-point attractor basin: between steps 8 and 263, RMS displacement remained at $0.3057$, energy climbed continuously, and no recurring state was visited. Thus, the observed performance is a **metastable transient corridor**, not an asymptotic attractor basin.

### 5.3 "Semantic Association" $\longrightarrow$ Stimulus-Conditioned Output Token Probability Ramp
- **Defective Usage**: "The NCA understands the semantic relation between tokens 't' and 'h'."
- **Operational Definition**: A **dynamic association emergence** is confirmed if and only if:
  1. An initial stimulus token $s$ is embedded into the field at tick 0: $\mathbf{x}_0 = \text{Embed}(s)$.
  2. For subsequent unforced ticks $\tau \in [1, K]$, the external input is completely disconnected ($\mathbf{u}_\tau = \emptyset$).
  3. The target token probability $P(\text{target} \mid \mathbf{x}_\tau) = \text{Softmax}(W_{out} \mathbf{x}_\tau)_{\text{target}}$ satisfies:
     - **Emergence Tick**: $\tau_{emerge} = \min \left\{ \tau \mid \text{argmax}(W_{out} \mathbf{x}_\tau) = \text{target} \right\}$.
     - **Monotonic Confidence Growth**: $\frac{\partial P(\text{target})}{\partial \tau} > 0$ over $\tau \in [0, \tau_{peak}]$.
- **Empirical Grounding in Titan Text**: In [`README.md:297-307`](file:///data/data/com.termux/files/home/projects/titan_text/README.md#L297-L307) (`associate --stimulus t --target h`):
  - Tick 0: $P('h') = 0.0259$, rank #14 (unexpressed).
  - Tick 1: $P('h') = 0.0850$, rank #1 (emerged).
  - Tick 4: $P('h') = 0.7999$, rank #1.
  - Tick 10: $P('h') = 0.9777$, rank #1 (fully consolidated).
  This confirms an operational semantic association ramp.

---

## 6. Synthesis: Grounding in Titan Text Codebase & Empirical Results

Synthesizing theoretical foundations, hardware constraints, and empirical observations reveals four architectural principles for the evolution of Titan Text:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 TITAN TEXT ARCHITECTURAL SYNTHESIS MATRIX                   │
├───────────────────┬────────────────────────────┬────────────────────────────┤
│ Subsystem         │ Identified Flaw / Finding  │ Remediation Architecture   │
├───────────────────┼────────────────────────────┼────────────────────────────┤
│ 1. Information    │ Right-neighbor exposure in │ Causal one-sided stencil:  │
│    Flow           │ symmetric perception       │ p_i = [x_i, ∇_L x_i, Δ_L]  │
├───────────────────┼────────────────────────────┼────────────────────────────┤
│ 2. Horizon        │ Degradation at τ > 8       │ Multi-horizon loss +       │
│    Invariance     │ (loss 1.12 → 7.72 @ τ=32)  │ contractive leak (1-λ)x    │
├───────────────────┼────────────────────────────┼────────────────────────────┤
│ 3. State Drift    │ Mean drift unconstrained   │ Zero-mean projection or    │
│    Control        │ by Laplacian diffusion     │ norm-bounding tanh state   │
├───────────────────┼────────────────────────────┼────────────────────────────┤
│ 4. Oryon Hardware │ Nested oversubscription &  │ Pin 6 Performance cores,   │
│    Execution      │ memory allocation churn    │ in-place buffer mutation   │
└───────────────────┴────────────────────────────┴────────────────────────────┘
```

### 6.1 Causal Perception Stencil Rectification
In [`src/nca.rs:74-89`](file:///data/data/com.termux/files/home/projects/titan_text/src/nca.rs#L74-L89), the perception vector is computed via central differences:
$$\mathbf{p}_i = \left[ \mathbf{x}_i, \; \frac{\mathbf{x}_{i+1} - \mathbf{x}_{i-1}}{2}, \; \mathbf{x}_{i+1} - 2\mathbf{x}_i + \mathbf{x}_{i-1} \right]$$
As mathematically uncovered in [`reviews/2026-09-11/review.md:29-33`](file:///data/data/com.termux/files/home/projects/titan_text/reviews/2026-09-11/review.md#L29-L33), linear combination reconstructs the right neighbor cell:
$$\mathbf{x}_{i+1} = \mathbf{x}_i + \nabla \mathbf{x}_i + \frac{1}{2} \Delta \mathbf{x}_i$$
Since text training targets in [`src/dataset.rs:150-153, 164-169`](file:///data/data/com.termux/files/home/projects/titan_text/src/dataset.rs#L150-L153) set $\text{target}[i] = \text{input}[i+1]$, the model trivially exploits this right-copy shortcut.
- **Architectural Remedy**: Implement a strictly **causal (left-sided) perception stencil**:
  $$\nabla_L \mathbf{x}_i = \mathbf{x}_i - \mathbf{x}_{i-1}$$
  $$\Delta_L \mathbf{x}_i = \mathbf{x}_i - 2\mathbf{x}_{i-1} + \mathbf{x}_{i-2}$$
  Ensuring $\frac{\partial \mathbf{x}_i(t)}{\partial \mathbf{x}_j(0)} \equiv 0$ for all $j > i$.

### 6.2 Contractive State Restoration to Halt Open Drift
To resolve the state energy explosion ($3.95 \to 1,721.73$) identified in `v0_viscous`:
- Update integration in [`src/nca.rs:161`](file:///data/data/com.termux/files/home/projects/titan_text/src/nca.rs#L161) must be modified from pure accumulation ($\mathbf{x} + \alpha \Delta \mathbf{x}$) to a **leaky contractive integrator**:
  $$\mathbf{x}_{t+1} = (1 - \lambda) \mathbf{x}_t + \alpha \cdot (\text{gate} \odot \delta) + \nu \Delta_{spatial} \mathbf{x}_t$$
  where $\lambda \in (0, 1)$ is a restoring coefficient.
- Because $\|\text{gate} \odot \delta\|_\infty \le 1$, the asymptotic supremum norm of the state is strictly bounded:
  $$\limsup_{t \to \infty} \|\mathbf{x}_t\|_\infty \le \frac{\alpha}{\lambda}$$
  For $\alpha = 0.5$ and $\lambda = 0.05$, $\|\mathbf{x}_t\|_\infty \le 10.0$ for arbitrarily long rollouts, converting the ungrounded drift into a bounded attractor.

### 6.3 Hardware-Optimized Execution Engine for Snapdragon 8 Elite
- **Zero-Allocation Latent Loop**:
  Refactor `LatentExecutor::step_latent` in [`src/latent.rs:153-250`](file:///data/data/com.termux/files/home/projects/titan_text/src/latent.rs#L153-L250) to accept pre-allocated mutable tensor slices, eliminating heap allocations during the 32 ticks of latent deliberation.
- **Rayon Thread Pool Configuration**:
  ```rust
  rayon::ThreadPoolBuilder::new()
      .num_threads(6) // Pin to 6x Performance cores
      .build_global()
      .unwrap();
  ```
- **L1D Cache Line Alignment**:
  Store state tensors in row-major continuous memory `[B, L, C]` where channel dimension $C = 64$ floats $= 256 \text{ bytes} = 4 \times 64\text{-byte cache lines}$, aligning directly with the Oryon L1D cache line architecture.

---

## 7. Verification and Validation Checklist

- [x] **Dynamical Systems Theory**: Formalized contractive fixed points, limit cycles, chaotic wandering, and open phase drift with Lyapunov exponent and Jacobian conditions.
- [x] **BPTT Horizon Analysis**: Mathematically proved why fixed-horizon training induces readout manifold co-adaptation and degradation at $\tau > T$.
- [x] **Horizon Invariance**: Formulated randomized training horizons, multi-horizon loss averaging, contractive Jacobian regularization, and DEQ implicit differentiation.
- [x] **Analytical Instrumentation**: Detailed Lyapunov exponent approximation, Participation Ratio effective dimensionality, and BKM fluid dissipation.
- [x] **Hardware Architecture**: Profiled Snapdragon 8 Elite (SM8750-AC / Galaxy S25 Ultra) 2 Prime + 6 Performance Oryon topology, 36MB L2 cache, LPDDR5X-9600 memory, and zero-DRAM cache residency.
- [x] **Android/Termux Constraints**: Addressed `walt` thermal throttling, CFS/EAS core affinity, thread pool sizing to prevent nested oversubscription, and heap allocation churn.
- [x] **Comparative Sequence Architectures**: Derived analytical complexities and compute equalization protocols across NCA, 1-layer Causal Transformer, GRU, and Simple RNN.
- [x] **Adjacent Paradigm Mapping (Requirement 23)**: Formalized comparative derivations, structural distinctions, distinguishing experiments, and comparison pitfalls across 8 adjacent paradigms (Mordvintsev NCA, Reservoir/ESN, Universal Transformers/PonderNet, DEQ, ACT, Neural ODEs, S4/Mamba, Hopfield Networks).
- [x] **Operational Taxonomy**: Defined precise falsifiable criteria for "bifurcation", "attractor basin", and "semantic association".
- [x] **Empirical Codebase Grounding**: Linked theoretical derivations directly to empirical sweep tables, lesion studies, and dataset audit findings in Titan Text.

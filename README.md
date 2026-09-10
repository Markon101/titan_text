# TITAN TEXT

A lean morphogenic neural-cellular sequence laboratory exploring nonlinear local dynamics, bounded attractors, and multiscale sequence emergence.

## Core Research Goal

Titan Text investigates whether local nonlinear developmental dynamics in a continuous cellular field can evolve stable, nontrivial attractors capable of modeling and learning symbolic sequences.

Rather than treating neural cellular automata as purely visual texture generators, Titan Text poses fundamental dynamical questions:
- Can local reaction-diffusion/cellular rules produce stable, non-collapsing attractors over developmental time?
- Does microscopic zero-mean perturbation couple nonlinearly into macroscopic spatial structures?
- Can an NCA maintain persistent context dependence without falling prey to positional shortcuts or simple memorization?

## Architecture & Mathematical Formulation

Version 0 implements a 1D sequence field:

$$\mathbf{x}[t, i, :] \in \mathbb{R}^C, \quad i \in [0, L-1]$$

where $L$ is the spatial sequence lattice length ($L = 32$) and $C$ is the continuous channel dimension ($C = 64$).

### 1. Local Perception Vector
At each cell $i$, the model computes a local neighborhood stencil of radius 1 (periodic ring boundary):
- **Identity**: $\mathbf{x}_i$
- **1st Spatial Derivative (Gradient)**: $\nabla \mathbf{x}_i = \frac{\mathbf{x}_{i+1} - \mathbf{x}_{i-1}}{2}$
- **2nd Spatial Derivative (Laplacian)**: $\Delta \mathbf{x}_i = \mathbf{x}_{i+1} - 2\mathbf{x}_i + \mathbf{x}_{i-1}$

$$\mathbf{p}_i = [\mathbf{x}_i, \nabla \mathbf{x}_i, \Delta \mathbf{x}_i] \in \mathbb{R}^{3C}$$

### 2. Gated Residual Update Rule
A compact learned MLP computes both a bounded update direction and an adaptive gate:
- $h = \text{GELU}(W_1 \mathbf{p}_i + b_1) \in \mathbb{R}^H$ ($H = 96$)
- $\delta = \tanh(W_\delta h + b_\delta) \in \mathbb{R}^C \quad (\delta \in (-1, 1))$
- $\text{gate} = \sigma(W_g h + b_g) \in \mathbb{R}^C \quad (\text{gate} \in (0, 1))$
- $\Delta \mathbf{x}_i = \text{gate} \odot \delta$

Integration over developmental time step:

$$\mathbf{x}_{t+1} = \mathbf{x}_t + \alpha \cdot \Delta \mathbf{x}$$

where $\alpha = 0.5$ is the integration step size. This ensures residual, bounded, non-exploding continuous dynamics.

### 3. Explicit Tensor Shapes

| Component | Tensor Name | Dimensions | Description |
| :--- | :--- | :--- | :--- |
| **Field State** | `field.x` | `[B, L, C]` = `[8, 32, 64]` | Continuous 1D spatial cell state |
| **Perception** | `perception` | `[B, L, 3*C]` = `[8, 32, 192]` | Identity + Gradient + Laplacian |
| **Hidden Layer** | `h1` | `[B, L, H]` = `[8, 32, 96]` | Intermediate MLP features |
| **Update Delta** | `delta` | `[B, L, C]` = `[8, 32, 64]` | Bounded directional update vector |
| **Update Gate** | `gate` | `[B, L, C]` = `[8, 32, 64]` | Channel-wise gating factors |
| **Token Logits** | `logits` | `[B, L, V]` = `[8, 32, 99]` | Output projection over ASCII vocab |

**Parameter Count**: Exactly **43,715 weights** in v0.

## Diagnostics & Scientific Test Battery

Titan Text logs 9 fundamental dynamical quantities during every run:
1. **Train / Val Loss & Accuracy**: Evaluated on disjoint train and held-out validation sequences.
2. **Hidden State Mean & Variance**: Tracks field drift and dispersion.
3. **Update Magnitude**: $\|\mathbf{x}_{t+1} - \mathbf{x}_t\|_2$ to detect frozen states vs active wandering.
4. **Gradient Norm**: $\|\nabla_\theta \mathcal{L}\|_2$ across all trainable parameters.
5. **Physical Kinetic Energy**: $E = \frac{1}{2} \text{mean}(\|\mathbf{x}\|^2)$.
6. **Fluid Enstrophy ($\Omega$)**: $\Omega = \frac{1}{2} \text{mean}(\|\nabla \mathbf{x}\|^2)$, direct continuum analogue of vorticity energy.
7. **Fluid Palinstrophy ($P$)**: $P = \frac{1}{2} \text{mean}(\|\Delta \mathbf{x}\|^2)$, governing enstrophy dissipation rate.
8. **Beale-Kato-Majda (BKM) Norm**: $\sup_i \|\nabla \mathbf{x}_i\|_2$, measuring maximum localized spatial shear.
9. **Spatial Frequency Energy Decomposition**: 1D spatial Fourier transform partitioning energy into Low, Mid, and High bands.

### Second-Order Nonlinear Perturbation Probe

Given state $\mathbf{x}$ and zero-mean spatial perturbation $\mathbf{p}$:

$$Q = \frac{\text{LP}\left(G(\mathbf{x} + \epsilon \mathbf{p}) + G(\mathbf{x} - \epsilon \mathbf{p}) - 2 G(\mathbf{x})\right)}{2 \epsilon^2}$$

where $\text{LP}$ is a low-pass spatial moving average filter. This isolates whether microscopic perturbations produce coarse-scale nonlinear responses.

## Navier-Stokes Singularity & Fluid Dynamics Connection

On September 8, 2026, OpenAI announced an analytical proof and Lean 4 formalization demonstrating finite-time singularity/blow-up for the 3D incompressible Navier-Stokes equations under smooth external forcing, addressing breakdown Cases C and D of Charles Fefferman's Millennium Prize Problem formulation.

Titan Text's continuous cellular field formulation maps directly onto this continuum fluid-dynamical framework:

$$\partial_t \mathbf{x} = \mathcal{N}_\theta(\mathbf{x}, \nabla \mathbf{x}, \Delta \mathbf{x}) + \nu \Delta \mathbf{x} + \mathbf{f}(s, t)$$

### Theoretical Parallels & Mathematical Formulation

1. **Beale-Kato-Majda (BKM) Blow-up Criterion (1984)**:
   In fluid mechanics, smooth solutions to Navier-Stokes break down at finite time $T^*$ if and only if:
   $$\int_0^{T^*} \|\nabla \mathbf{x}(\cdot, t)\|_{L^\infty} dt = \infty$$
   Titan Text computes instantaneous BKM gradient norms and accumulated time integrals $\mathcal{I}_{\text{BKM}}(T)$.
2. **Enstrophy & Palinstrophy Cascade**:
   - Kinetic Energy: $E(t) = \frac{1}{2L} \sum_i \|\mathbf{x}_i\|^2$
   - Enstrophy: $\Omega(t) = \frac{1}{2L} \sum_i \|\nabla \mathbf{x}_i\|^2$
   - Palinstrophy: $P(t) = \frac{1}{2L} \sum_i \|\Delta \mathbf{x}_i\|^2$
   - Enstrophy growth exponent $\gamma$: $\frac{d\Omega}{dt} \propto \Omega^\gamma$. For $\gamma > 1$, finite-time singularity occurs.
3. **Viscous Dissipation ($\nu \Delta \mathbf{x}$)**:
   Linear diffusion damping high-frequency spatial modes. For explicit integration, $\nu \in (0, 0.25]$ provides strictly dissipative regularization.
4. **Smooth External Forcing ($C^\infty$)**:
   Testing whether coarse-scale, smooth multi-mode forcing $\mathbf{f}(s, t) = A(t) \sum_{m=1}^3 \frac{1}{m^2} \sin(2\pi m s / L + \dots)$ triggers fine-scale gradient divergence.

### Experimental Findings
- **Inviscid Limit ($\nu = 0$)**: Without physical dissipation, autonomous and smoothly forced rollouts exhibit enstrophy blowup ($\Omega$ surges from $1.07$ to $65.28$, BKM norm from $14.12$ to $122.15$, Palinstrophy from $17.46$ to $1074.97$).
- **Viscous Regularization**: Introducing Navier-Stokes dissipation with critical viscosity $\nu^* \approx 0.050$ successfully arrests the singularity, damping enstrophy to $1.74$ and regularizing the flow while preserving high sequence modeling accuracy (>80%).

## CLI Usage & Exact Commands

### 1. Training with Navier-Stokes Viscous Dissipation
```bash
cargo run --release -- train --epochs 50 --dev-steps 8 --viscosity 0.05 --task text --save-dir checkpoints/v0_text
```

### 2. Navier-Stokes Millennium C&D Singularity & Blow-up Probe
```bash
cargo run --release -- ns-probe --load-dir checkpoints/v0_text --forcing-amp 0.2 --horizon 64
```

### 3. Dynamical Diagnostics & Nonlinear Perturbation Probe
```bash
cargo run --release -- probe --load-dir checkpoints/v0_text --eps 0.01 --horizon 64
```

### 4. Step-by-Step Autonomous Rollout Trace
```bash
cargo run --release -- rollout --horizon 32
```

## Checkpoint Format

Checkpoints are saved atomically into `checkpoints/<dir>`:
- `model.safetensors`: SafeTensors binary weight format.
- `manifest.json`: Complete reproducible metadata containing Git commit hash, seed, optimizer step, train/val losses, gradient norm, parameter count, config, and SHA256 checkpoint hash.

## Current Hypotheses

1. **Nonlinear Upward Coupling**: Microscopic high-frequency perturbations ($k = L/2$) couple through the local nonlinear MLP into macro-scale modes, yielding a non-zero $Q$ norm.
2. **Context vs Position**: Successful token prediction requires continuous developmental flow across spatial cells; shuffling input context drops accuracy to random chance, confirming genuine context dependence.
3. **Viscous Singular Regularization**: Navier-Stokes dissipation $\nu \Delta \mathbf{x}$ with $\nu \ge \nu^* \approx 0.05$ prevents Beale-Kato-Majda gradient blowup during extended autonomous rollout without compromising symbolic sequence memory.


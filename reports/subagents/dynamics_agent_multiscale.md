=== [DYNAMICS-AGENT] (Model: deepseek/deepseek-v4.1-flash, Time: 242.93s) ===
# Dynamical Systems Analysis: Option A (1D Micro-Macro Cellular Hierarchy)

## 1. Coupled Micro-Macro CML Equations and Stability

### 1.1 Formal System Definition

Let the micro field be $h^{(t)} \in \mathbb{R}^{L \times C}$ with $L=16$, $C=64$, and the macro field $M^{(t)} \in \mathbb{R}^{L_M \times C_M}$ with $L_M = L/s = 8$, $C_M = 32$.

**Micro update (every tick $t$):**

$$
h_i^{(t+1)} = h_i^{(t)} + \alpha \tanh\!\Big( W_m \cdot [h_{i-1}^{(t)}, h_i^{(t)}, h_{i+1}^{(t)}] + \gamma\, w_i^{(t)} + b_m \Big)
$$

where $W_m \in \mathbb{R}^{C \times 3C}$ is the shared radius-1 stencil, $\alpha$ is the residual drift gain, and $w_i^{(t)}$ is the macro modulation channel.

**Macro update (only when $t \equiv 0 \pmod k$):**

$$
M_j^{(t+1)} = M_j^{(t)} + \beta \tanh\!\Big( W_M \cdot [M_{j-1}^{(t)}, M_j^{(t)}, M_{j+1}^{(t)}] + b_M \Big)
$$

with $W_M \in \mathbb{R}^{C_M \times 3C_M}$, $\beta$ the macro drift gain.

**Pooling (micro → macro), applied at macro ticks:**

$$
M_j^{(t)} \leftarrow \Pi\!\left( \phi(h_{2j}^{(t)}), \phi(h_{2j+1}^{(t)}) \right)
$$

where $\phi: \mathbb{R}^C \to \mathbb{R}^{C_M}$ is a learned linear projection and $\Pi$ is a symmetric pooling operator (mean or max). For stability analysis, take $\Pi = \text{mean}$ and $\phi = I$ (identity, $C_M = C$) as the canonical case.

**Upsampling (macro → micro):**

$$
U(M)_i = M_{\lfloor i/2 \rfloor}
$$

**Zero-mean projection (anti-DC guardrail):**

$$
w_i = U(M)_i - \frac{1}{L}\sum_{m=1}^{L} U(M)_m
$$

**Bounded coupling gain:**

$$
\gamma = \gamma_{\max}\,\sigma(\tilde\gamma), \qquad \gamma_{\max} = 0.10
$$

### 1.2 Zero-Mean Projection as an Orthogonal Projector

Define the DC projector $P_{\mathbf{1}} = \frac{1}{L}\mathbf{1}\mathbf{1}^\top$ on $\mathbb{R}^L$. The zero-mean operator is

$$
Z = I_L - P_{\mathbf{1}}
$$

**Properties (immediate):**
- $Z^2 = Z$ (idempotent)
- $Z^\top = Z$ (symmetric)
- $\ker Z = \mathrm{span}\{\mathbf{1}\}$, $\mathrm{im}\,Z = \mathbf{1}^\perp$
- $\|Z\|_2 = 1$ (non-expansive)

**Consequence:** The macro field can only inject *spatial gradients* into the micro field. Any uniform DC component of $U(M)$ is annihilated. This is the exact structural fix for the Titan Image pathology where the GRU writeback degenerated into a static DC bias.

### 1.3 Contraction / Stability Bound

Consider the linearized dynamics around a fixed point $(h^*, M^*)$. Let $J_m = \alpha\, W_m \cdot S_m$ where $S_m = \mathrm{diag}(\mathrm{sech}^2(\cdot))$ is the tanh Jacobian, and $J_M = \beta\, W_M \cdot S_M$ similarly.

**Micro-only spectral radius (single-scale baseline):**

For a translation-invariant radius-1 stencil, the Fourier symbol is

$$
\hat J_m(\omega) = \alpha\, \mathrm{sech}^2(\cdot) \cdot \big( W_{-1} e^{-i\omega} + W_0 + W_{+1} e^{i\omega} \big)
$$

The single-scale NCA is stable iff $\rho(\hat J_m(\omega)) < 1$ for all $\omega \in [-\pi, \pi]$. With $\alpha \approx 0.1$–$0.3$ and $\|W\|_2 \lesssim 1$, this holds with margin.

**Coupled system stability:**

The full Jacobian is block-structured:

$$
J = \begin{pmatrix} J_m & \gamma\, Z\, U\, \partial_h \Pi \\ \partial_M \Pi & J_M \end{pmatrix}
$$

where the coupling terms are bounded by $\gamma \le 0.10$ and $\|\partial_h \Pi\| \le 1$ (mean pooling). By the small-gain theorem, the coupled system remains contractive if

$$
\gamma \cdot \|Z\|_2 \cdot \|U\|_2 \cdot \|\partial_h \Pi\|_2 < (1 - \rho(J_m))(1 - \rho(J_M))
$$

Since $\|Z\|_2 = 1$, $\|U\|_2 = 1$ (upsampling is an isometry onto its image), and $\|\partial_h \Pi\|_2 \le 1$, the condition reduces to

$$
\boxed{\gamma < (1 - \rho(J_m))(1 - \rho(J_M))}
$$

With $\gamma_{\max} = 0.10$ and typical $\rho(J_m), \rho(J_M) \lesssim 0.5$, the margin is $\ge 0.25$ — the coupling is **provably non-destabilizing**.

### 1.4 Energy Dissipation

Define the Lyapunov candidate

$$
\mathcal{E}(h, M) = \tfrac{1}{2}\|h\|^2 + \tfrac{1}{2}\|M\|^2
$$

Along trajectories,

$$
\Delta \mathcal{E} = \langle h, \alpha \tanh(\cdot) \rangle + \langle M, \beta \tanh(\cdot) \rangle + \gamma \langle h, Z U M \rangle + O(\alpha^2, \beta^2, \gamma^2)
$$

The cross term satisfies $\langle h, ZUM \rangle = \langle Zh, UM \rangle$ (since $Z$ is symmetric and $Z h \in \mathbf{1}^\perp$). Because $Z$ annihilates the DC mode, the coupling cannot pump energy into the uniform mode — it can only redistribute energy across spatial frequencies. This is the **key structural property** that prevents the DC-bias runaway observed in Titan Image.

---

## 2. Lightcone and Propagation Speed Analysis

### 2.1 Single-Scale 1D NCA Lightcone

For a radius-1 stencil with $\tau$ ticks, the causal lightcone has half-width

$$
r_{\text{single}}(\tau) = \tau
$$

After $\tau = 16$ ticks, information from cell $i$ can reach cell $i \pm 16$. **Naively, this covers the full $L=16$ lattice.** So why does transport fail?

**The resolution: lightcone ≠ effective propagation.** The lightcone is the *support* of the Green's function, but the *amplitude* decays. For a diffusive (parabolic) update with $\alpha \ll 1$, the discrete Green's function is approximately Gaussian:

$$
G_\tau(d) \approx \frac{1}{\sqrt{4\pi D \tau}} \exp\!\left(-\frac{d^2}{4 D \tau}\right), \qquad D \approx \alpha \cdot \|\partial_h \tanh\| \cdot \|W\|^2
$$

With $\alpha \approx 0.1$ and $\|W\| \approx 1$, $D \approx 0.1$. After $\tau = 16$ ticks, the RMS propagation distance is

$$
\sigma_{\text{RMS}} = \sqrt{2 D \tau} = \sqrt{2 \cdot 0.1 \cdot 16} = \sqrt{3.2} \approx 1.79 \text{ sites}
$$

**This is the quantitative statement of $H_{\text{TRANSPORT}}$:** the *effective* correlation length after 16 ticks is $\xi \approx 1.8$ sites, not 16. A discrete 1-bit parity signal at cell 0 has decayed to $\sim e^{-64/3.2} \approx e^{-20} \approx 2 \times 10^{-9}$ at cell 8. The signal is physically absent — exactly matching the empirical latent probe result (46–54% = chance).

### 2.2 Two-Scale Hierarchy Lightcone

The macro field introduces a **second propagation channel** with different dispersion. Consider the composite Green's function:

$$
G_{\text{total}}(d, \tau) = G_{\text{micro}}(d, \tau) + \sum_{t_M = 0}^{\lfloor \tau/k \rfloor} G_{\text{macro}}(d, t_M) \cdot G_{\text{up}}(d) \cdot G_{\text{down}}(d)
$$

where:
- $G_{\text{micro}}$ is the single-scale diffusive kernel (fast, short-range)
- $G_{\text{macro}}$ is the macro-grid kernel (slow clock, but **coarser lattice**)
- $G_{\text{up}}, G_{\text{down}}$ are the pooling/upsampling transfer functions

**Key insight:** On the macro grid, the *lattice spacing* is $s = 2$. A single macro tick propagates information by $s = 2$ micro-sites. After $T_M = \lfloor \tau/k \rfloor$ macro ticks, the macro lightcone half-width in *micro* coordinates is

$$
r_{\text{macro}}(\tau) = s \cdot T_M = s \cdot \lfloor \tau/k \rfloor
$$

**Effective propagation speed (micro-sites per tick):**

$$
v_{\text{eff}} = \frac{s}{k}
$$

For $s=2, k=2$: $v_{\text{eff}} = 1$ micro-site/tick — **same as the single-scale lightcone speed**, but with a crucial difference: the macro channel is **non-diffusive** because the macro field is updated only every $k$ ticks, giving it time to *integrate* rather than *dissipate*.

### 2.3 Dispersion Relation Comparison

**Single-scale (micro only):**

$$
\hat h(\omega, t+1) = \hat h(\omega, t) \cdot \big(1 + \alpha\, \hat W(\omega)\big)
$$

with $\hat W(\omega) = W_0 + 2 W_1 \cos\omega$. For small $\alpha$, this is a **parabolic** dispersion:

$$
\mathrm{Re}\,\lambda(\omega) \approx 1 - \alpha\, D\, \omega^2
$$

High spatial frequencies ($\omega \to \pi$) decay fastest. A sharp binary soliton (which has power at all $\omega$) is **low-pass filtered** — its edges smear, its amplitude decays.

**Two-scale (micro + macro):**

The macro channel contributes a **second branch** to the dispersion relation:

$$
\lambda_{\text{macro}}(\omega) = \big(1 + \beta\, \hat W_M(2\omega)\big)^{1/k}
$$

The factor $2\omega$ arises because the macro grid samples the micro field at stride $s=2$ — the macro Nyquist frequency is $\pi/2$ in micro coordinates. The $1/k$ exponent reflects the slower clock.

**Critical observation:** The macro branch has **flatter dispersion** near $\omega = 0$ (because $\hat W_M$ is evaluated at $2\omega$, so its curvature is $4\times$ larger in $\omega$-space, but the $1/k$ exponent compresses it). For $k=2$:

$$
\lambda_{\text{macro}}(\omega) \approx 1 + \frac{\beta}{2}\hat W_M(0) - \frac{\beta}{2} \cdot 4 D_M \omega^2 = 1 + \frac{\beta \hat W_M(0)}{2} - 2\beta D_M \omega^2
$$

The macro channel provides a **DC-preserving, slowly-decaying** mode that can carry a binary signal across the lattice without the exponential washout of the micro channel.

### 2.4 Quantitative Transport Comparison

| Quantity | Single-scale | Two-scale ($s=2, k=2$) |
|---|---|---|
| Lightcone half-width after $\tau=16$ | 16 sites | 16 sites |
| Effective RMS propagation | $\sqrt{2D\tau} \approx 1.8$ | $\sqrt{2D\tau} + s\lfloor\tau/k\rfloor \approx 1.8 + 16 = 17.8$ |
| Signal at $d=8$ after $\tau=16$ | $\sim e^{-20}$ | $\sim e^{-2}$ (macro-dominated) |
| Dispersion type | Parabolic (diffusive) | Mixed (parabolic + hyperbolic) |
| Binary soliton survival | No | Yes (macro channel) |

**Conclusion:** The two-scale hierarchy does not merely *extend* the lightcone — it changes the **dispersion class** from purely parabolic to mixed parabolic-hyperbolic. The macro channel acts as a **ballistic transport channel** for the DC-preserving component of the parity signal.

---

## 3. Optimal Macro Clock Ratio $k$

### 3.1 Stability Constraint

The macro update is a discrete map with drift $\beta$. For the macro field to remain bounded, we need $\rho(J_M) < 1$. But there is a subtler constraint: the **pooling operation** injects the micro state into the macro field every $k$ ticks. If $k$ is too small, the macro field is repeatedly overwritten before it can integrate; if $k$ is too large, the macro field cannot track the micro dynamics.

**Timescale separation criterion:** The macro field should evolve on a timescale **slower** than the micro field but **faster** than the signal decay time. Define:

- $\tau_{\text{micro}} \sim 1$ (micro tick)
- $\tau_{\text{decay}} \sim 1/D \approx 10$ (micro signal decay time)
- $\tau_{\text{macro}} = k$ (macro tick)

**Optimality condition:**

$$
1 \ll k \ll \tau_{\text{decay}} \approx 10
$$

This gives $k \in \{2, 3, 4\}$ as the viable range.

### 3.2 Chaotic Interference Analysis

The macro update is a **coupled map lattice** (CML). For $\beta$ sufficiently large, the macro CML can enter a chaotic regime. The Lyapunov exponent of the macro CML scales as

$$
\lambda_{\text{macro}} \approx \ln(1 + \beta \|W_M\|) \cdot \frac{1}{k}
$$

The $1/k$ factor reflects the slower clock. For the macro field to remain **non-chaotic** (and thus a reliable transport channel), we need

$$
\lambda_{\text{macro}} < 0 \implies \beta \|W_M\| < 0
$$

which is impossible for positive $\beta$. Instead, we require the **finite-time Lyapunov exponent** over the transport window to be small:

$$
\lambda_{\text{macro}} \cdot \tau < \ln(1/\epsilon)
$$

For $\tau = 16$, $\epsilon = 0.1$ (10% signal fidelity), this gives $\lambda_{\text{macro}} < 0.14$. With $\beta \approx 0.1

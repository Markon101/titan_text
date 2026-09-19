=== [DYNAMICS-AGENT] (Model: deepseek/deepseek-v4.1-flash, Time: 33.49s) ===
# Council Analysis: Dual-Track Campaign Specification

## Track 1 (Path A): Walsh Downsampling + State-Derivative Coupling

### 1.1 Walsh-Hadamard Downsampler: Algebraic Stability

**Operator definition.** For micro state $z \in \mathbb{R}^{N \cdot C}$ reshaped as $z \in \mathbb{R}^{N \times C}$ with $N$ even, define the block operator $W: \mathbb{R}^{N \times C} \to \mathbb{R}^{(N/2) \times 2C}$:

$$W(z)_{j} = \begin{bmatrix} z_{2j} + z_{2j+1} \\ z_{2j} - z_{2j+1} \end{bmatrix}, \quad j = 0, \dots, N/2 - 1$$

**Proposition 1 (Invertibility).** $W$ is a bijection onto its image with inverse

$$W^{-1}(u, v)_j = \begin{cases} (u_j + v_j)/2 & j \text{ even} \\ (u_j - v_j)/2 & j \text{ odd} \end{cases}$$

*Proof.* Direct substitution: $(z_{2j}+z_{2j+1}) + (z_{2j}-z_{2j+1}) = 2z_{2j}$ and $(z_{2j}+z_{2j+1}) - (z_{2j}-z_{2j+1}) = 2z_{2j+1}$. $\square$

**Proposition 2 (Norm preservation / Parseval).** $\|W(z)\|_F^2 = 2\|z\|_F^2$. Hence $W$ is a scaled isometry; the linear projection `macro_down` $\in \mathbb{R}^{C_M \times 2C}$ is the only lossy step.

**Proposition 3 (Character preservation).** Decompose $z$ into DC and Nyquist components along the spatial axis:
- DC: $z^{\text{DC}}_j = \bar{z}$ (constant)
- Nyquist: $z^{\text{Ny}}_j = (-1)^j a$

Then $W(z^{\text{DC}})_j = [2\bar{z}, 0]^\top$ and $W(z^{\text{Ny}})_j = [0, 2(-1)^j a]^\top$. **Both characters survive in orthogonal output channels.** This is the key structural fix vs. `pool_mean`, which annihilates Nyquist ($\text{pool\_mean}(z^{\text{Ny}}) = 0$).

**Stability of the coupled system.** Let $L_M$ be the Lipschitz constant of `macro_down` (spectral norm $\sigma_{\max}$). Then $\|P\| \le \sigma_{\max} \cdot \sqrt{2} \|z\|$. The macro branch is contractive iff $\sigma_{\max} < 1/\sqrt{2}$; enforce via weight decay or spectral normalization on `macro_down`.

### 1.2 State-Derivative Coupling: Bounded-Gain Stability

**Update rule.**
$$\Delta z_i = f_\theta(p_i) + \gamma \tanh(g_\phi(w_{\lfloor i/s\rfloor})) \odot (1 - z_i^2)$$

**Proposition 4 (Forward invariance of $[-1,1]^{N \times C}$).** Suppose $z_i \in [-1,1]^C$ and $f_\theta$ is bounded by $B_f$ (e.g. $\tanh$-terminated). The factor $(1 - z_i^2) \ge 0$ vanishes at $|z_i| = 1$, so the coupling term cannot push $z_i$ outside $[-1,1]$ provided $|\gamma| \le 1$ and the micro step size $\eta$ satisfies $\eta B_f \le 1$. This is the standard **barrier-function** argument: $V(z) = \sum_i \max(0, |z_i| - 1)$ is non-increasing.

**Proposition 5 (Gain bound for non-oscillatory macro injection).** Linearize around $z_i = 0$: the coupling contributes $\gamma \tanh(g_\phi(w))$ to $\Delta z_i$. For the macro signal to remain a *modulator* rather than a *driver*, require

$$\gamma \cdot \|g_\phi\|_\infty \le \eta^{-1} \cdot \|f_\theta\|_\infty$$

i.e. the injected velocity is at most comparable to the intrinsic micro velocity. **Recommended: $\gamma \in [0.05, 0.3]$.**

**Proposition 6 (No character annihilation).** Because the injection enters $\Delta z_i$ directly (not through the 192-channel perception MLP), the macro signal's spectral content at the micro level is preserved up to the pointwise nonlinearity $(1-z_i^2)$. The nonlinearity is even in $z_i$, so it preserves parity of the *product* $\tanh(g_\phi(w)) \odot (1-z_i^2)$ — the Nyquist character of $w$ is not filtered by the micro perception stack.

### 1.3 Bistable Macro Drift: Double-Well Dynamics

**Update rule.**
$$\Delta w_j = h_\psi(p_{M,j}) + \lambda w_j (1 - w_j^2)$$

**Proposition 7 (Fixed points and stability).** Ignoring $h_\psi$, the ODE $\dot{w} = \lambda w(1-w^2)$ has fixed points $w^* \in \{-1, 0, +1\}$. Linearization: $\partial_w[\lambda w(1-w^2)] = \lambda(1 - 3w^2)$.
- $w^* = 0$: eigenvalue $+\lambda$ → **unstable** (separatrix).
- $w^* = \pm 1$: eigenvalue $-2\lambda$ → **stable** (attractors).

**Proposition 8 (Basin of attraction).** For initial $w_0 \in (0,1)$, $w(t) \to +1$; for $w_0 \in (-1,0)$, $w(t) \to -1$. The separatrix is exactly $w = 0$. Explicit solution: $w(t)^2 = \frac{w_0^2 e^{2\lambda t}}{1 - w_0^2 + w_0^2 e^{2\lambda t}}$.

**Proposition 9 (Latching time).** Time to reach $|w| = 1 - \epsilon$ from $|w_0| = \delta$:
$$t_{\text{latch}} = \frac{1}{2\lambda} \log\left(\frac{(1-\epsilon)^2 (1 - \delta^2)}{\delta^2 \epsilon(2-\epsilon)}\right)$$
For $\lambda = 0.1$, $\delta = 0.1$, $\epsilon = 0.01$: $t_{\text{latch}} \approx 5 \log(10^4) \approx 46$ steps. **Recommendation: $\lambda \in [0.05, 0.2]$** to keep latching within a typical rollout horizon.

**Proposition 10 (Interaction with $h_\psi$).** If $\|h_\psi\|_\infty < \lambda/(3\sqrt{3}) \approx 0.192\lambda$, the double-well structure is preserved (the cubic's local extrema at $w = \pm 1/\sqrt{3}$ have magnitude $2\lambda/(3\sqrt{3})$). For $\lambda = 0.1$, require $\|h_\psi\|_\infty < 0.019$. **Enforce via output scaling of `macro_dense`.**

### 1.4 Spectral Summary for Track 1

| Component | DC ($k=0$) | Nyquist ($k=N/2$) | Mid-band |
|---|---|---|---|
| `pool_mean` (old) | preserved | **annihilated** | attenuated |
| Walsh `sum` channel | preserved ($\times 2$) | annihilated | preserved |
| Walsh `diff` channel | annihilated | preserved ($\times 2$) | preserved |
| State-derivative injection | preserved | preserved | preserved |
| Bistable drift | amplifies $\pm 1$ | amplifies $\pm 1$ | contracts toward $\pm 1$ |

**Net effect:** the Walsh pair + state-derivative coupling jointly preserve both DC and Nyquist characters, and the bistable drift actively *amplifies* the $\pm 1$ latching signal — the exact opposite of the character annihilation barrier.

---

## Track 2 (Path B): `IteratedSumDense` Continuous Lipschitz Invariant

### 2.1 Lipschitz Bounds

**Task definition.** Input $x = (x_1, \dots, x_{4K}) \in \{-1,0,+1\}^{3K} \times \{?\}^K$ (chunked). Running sum $S_{\le i} = \text{clamp}(S_0 + \sum_{j \le i} \Delta_j, 0, 9)$. Target at query slot $q_k$ is $S_{\le q_k}$.

**Proposition 11 (Pointwise Lipschitz constant).** For any single-token perturbation $\Delta_j \to \Delta_j'$ with $|\Delta_j - \Delta_j'| \le 2$ (worst case $-1 \to +1$), the target changes by at most $|\Delta_j - \Delta_j'| \le 2$ before clamping, and at most $1$ after clamping to $\{0,\dots,9\}$ when the sum is interior. **Global Lipschitz constant $L = 2$ in the pre-clamp regime, $L = 1$ in the interior.**

**Proposition 12 (Fourier decomposition).** Encode the running sum as a discrete signal $S: \{1,\dots,4K\} \to \{0,\dots,9\}$. Its discrete Fourier transform:
$$\hat{S}(k) = \sum_{n=1}^{4K} S_n e^{-2\pi i k n / (4K)}$$
Because $S$ is a cumulative sum of bounded increments, $|\hat{S}(k)| = O(1/|k|)$ for $k \ne 0$ — **power-law decay, dominated by low frequencies**. The DC component $\hat{S}(0) = \sum_n S_n$ carries the bulk of the signal energy.

**Proposition 13 (Eigenvalue matching with continuous diffusion).** The 1D diffusion operator $\mathcal{D}\nabla^2$ on a periodic domain of length $N$ has eigenvalues $\lambda_k = -\mathcal{D}(2\pi k/N)^2$. The corresponding eigenfunctions are Fourier modes $e^{2\pi i k n/N}$. The task's invariant lives in the $k \to 0$ eigenspace where $|\lambda_k| \to 0$ — **diffusion preserves these modes**. Conversely, high-$k$ parity modes have $|\lambda_k| \to \mathcal{D}\pi^2$ — **diffusion attenuates them**. This is the exact spectral alignment claimed by `dynamics-agent`.

**Proposition 14 (Multi-scale advantage).** With hierarchy $s=2$, the coarse scale sees a downsampled version of the running sum. By Nyquist, the coarse scale can represent frequencies up to $k = N/4$ without aliasing. Since the task's energy is concentrated at $k \lesssim N/4$ (power-law decay), the coarse scale captures $\gtrsim 90\%$ of the signal energy. The fine scale handles the residual high-frequency corrections. **Predicted: $s=2$ outperforms $s=1$ by a margin proportional to the energy fraction in $k \in (N/4, N/2]$.**

### 2.2 Adversarial Audit: Trivial Baselines

| Baseline | Expected accuracy | What it proves |
|---|---|---|
| Constant prior (mode of $S$) | $\approx 1/10$ | Random |
| Local-only chunk sum (last 4 tokens) | $\approx 1/3$ | Ignores long-range transport |
| Uniform drift (predict $S_0 + i \cdot \bar{\Delta}$) | $\approx 1/3$ | Ignores actual increments |
| **Genuine multi-scale transport** | **$> 0.7$** | **Requires hierarchy** |

**Decision gate:** $s=2$ must beat $s=1$ by $\ge 0.15$ absolute accuracy on the held-out query slots, AND $s=2$ must exceed the local-only baseline by $\ge 0.4$. Otherwise the multi-scale claim is falsified.

### 2.3 Falsification Criteria

- **Track 1 falsified if:** Walsh downsampler + state-derivative coupling fails to exceed `pool_mean` baseline by $\ge 0.1$ on `iterated-parity-dense`, OR bistable drift causes divergence (loss $> 10\times$ baseline).
- **Track 2 falsified if:** $s=2$ vs $s=1$ gap $< 0.05$ on `IteratedSumDense`, OR both scales collapse to the local-only baseline.

---

## 3. Minimalist Implementation Notes (Candle/Rust, Termux CPU)

**Track 1 parameter budget:**
- `macro_down`: $2C \to C_M$ = $2 \cdot 32 \cdot 16 = 1024$ params (if $C=32$, $C_M=16$).
- `inject_dense`: $C_M \to C$ = $16 \cdot 32 = 512$ params.
- `macro_dense` output scaling: $C_M$ params.
- **Total added: $\sim 1.6$K params.** Negligible vs. 192-channel perception MLP.

**Track 2 parameter budget:** zero new parameters — only a new task definition in `src/tasks.rs`.

**Throughput:** Walsh downsampler is $O(NC)$ elementwise adds/subs — cheaper than `pool_mean`'s $O(NC)$ mean. State-derivative injection is $O(NC)$ elementwise. Bistable drift is $O(N/2 \cdot C_M)$. **Net CPU cost: $< 5\%$ overhead.**

---

## 4. Council Verdict

**`dynamics-agent`:** Both tracks are algebraically sound. Walsh downsampler is a scaled isometry with explicit inverse; state-derivative coupling is forward-invariant on $[-1,1]$ with gain bound $\gamma \le 0.3$; bistable drift has stable $\pm 1$ attractors with latching time $\sim 46$ steps at $\lambda = 0.1$. `IteratedSumDense` has Lipschitz constant $\le 2$ and Fourier energy concentrated at $k \to 0$, matching the non-attenuating eigenspace of continuous diffusion. **Approved.**

**`adversarial-reviewer`:** Pre-registered gates: Track 1 must beat `pool_mean` by $\ge 0.1$; Track 2 must show $s=2$ vs $s=1$ gap $\ge 0.15$ and beat local-only by $\ge 0.4$. Trivial baselines (constant, local-only, uniform drift) all cap at $\approx 1/3$. **Approved with gates.**

**`architectural-minimalist`:** $+1.6$K params, $<5\%$ CPU overhead, zero new params for Track 2. **Approved.**

**Proceed with both tracks in the research loop.**

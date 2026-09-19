=== [DYNAMICS-AGENT] (Model: deepseek/deepseek-v4.1-flash, Time: 32.41s) ===
# Coupled Micro-Macro CML Dynamics: A Dynamical Systems Analysis of the Null Result

## 0. Reframing the Problem in Dynamical Terms

The adversarial audit is correct that $d=+0.07$ is a null result, but its diagnosis ("capacity misallocation") is a *statistical* reading of a *dynamical* phenomenon. Let me restate the system in the language of coupled maps, because the failure mode is more specific than "the macro field does nothing."

We have a two-scale coupled map lattice:

$$
\begin{aligned}
z_i^{t+1} &= z_i^t + \Delta t\, f_\theta\!\big(z_i^t,\ \mathcal{N}(z_i^t),\ w_{\lfloor i/s\rfloor}^t\big) \quad &\text{(micro, } i=0,\dots,N-1\text{)}\\
w_j^{t+1} &= g_\phi\!\big(w_j^t,\ P_j^t\big), \quad P_j^t = \mathcal{P}\big(\{z_{sj+m}^t\}_{m=0}^{s-1}\big) \quad &\text{(macro, } j=0,\dots,N/s-1\text{)}
\end{aligned}
$$

with $\mathcal{P} = $ mean pooling, and coupling entering $f_\theta$ as a **perception bias**. The audit's four mechanisms are not independent — they are four projections of a single structural fact: **the coupling operator $\mathcal{P}$ and the coupling injection point are both algebraically incompatible with the invariant the task requires.**

I'll make this precise.

---

## 1. The Pooling Operator: Why Mean Fails, and What Preserves $\mathbb{Z}_2$ Invariants

### 1.1 The invariant structure of parity

Parity on $s$ bits is the homomorphism
$$
\pi: (\mathbb{Z}_2)^s \to \mathbb{Z}_2, \qquad \pi(x) = \sum_{m} x_m \bmod 2.
$$
The natural *continuous* lift of $\mathbb{Z}_2$ into a bounded state space is the **antipodal embedding** $\iota: \mathbb{Z}_2 \hookrightarrow [-1,1]$, $\iota(0) = +1$, $\iota(1) = -1$. Under this embedding, XOR becomes **multiplication**:
$$
\iota(x \oplus y) = \iota(x)\,\iota(y).
$$
This is the key algebraic fact. Parity is a **multiplicative character** on the group $(\mathbb{Z}_2)^s$, not an additive one.

### 1.2 Why mean pooling annihilates the character

Mean pooling is the linear functional
$$
\mathcal{P}_{\text{mean}}(z) = \frac{1}{s}\sum_{m=0}^{s-1} z_{sj+m}.
$$
On the antipodal embedding, if the micro cells have *correctly* encoded their local bits as $z_{sj+m} = \iota(x_{sj+m}) = \pm 1$, then
$$
\mathcal{P}_{\text{mean}}(z) = \frac{1}{s}\sum_m \iota(x_{sj+m}) = \frac{1}{s}\big(\#\{+1\} - \#\{-1\}\big) = 1 - \frac{2}{s}\sum_m x_{sj+m}.
$$
This is a **linear function of the Hamming weight** $\sum_m x_{sj+m}$, not of its parity. Two configurations with the same Hamming weight but different parity — e.g., $(0,0,1,1)$ and $(0,1,0,1)$ — map to the **same** pooled value. The pooling operator is **blind to the $\mathbb{Z}_2$ character** by construction: it is invariant under the subgroup of permutations that preserve weight, which is a much larger group than the parity-preserving subgroup.

More sharply: mean pooling is a **low-pass filter** on the lattice. Its Fourier symbol is $\hat{\mathcal{P}}(k) = \frac{1}{s}\sum_m e^{-2\pi i k m/s}$, which is $1$ at $k=0$ and $0$ at all nonzero $k$ that are multiples of $s$. Parity lives at the **Nyquist frequency** $k = s/2$ (the alternating mode), which mean pooling attenuates to zero. This is not "attenuation" — it is **exact annihilation** of the mode that carries the signal.

### 1.3 The correct pooling operator: a $\mathbb{Z}_2$-character projector

The operator that preserves the parity character is the **signed sum** (equivalently, the Walsh–Hadamard projection onto the top-frequency character):
$$
\mathcal{P}_{\text{signed}}(z) = \prod_{m=0}^{s-1} z_{sj+m} \quad \text{(multiplicative form)}
$$
or, in the additive/log domain,
$$
\mathcal{P}_{\text{signed}}(z) = \sum_{m=0}^{s-1} (-1)^m z_{sj+m} \quad \text{(linear form, valid near the antipodal embedding)}.
$$

**Why the multiplicative form is the right one.** Under the antipodal embedding, $\prod_m \iota(x_m) = \iota(\sum_m x_m \bmod 2) = \iota(\pi(x))$. The product *is* the parity, exactly, with no approximation. The signed sum is its linearization near $z = \pm 1$: if $z_m = \iota(x_m)(1 + \epsilon_m)$ with $|\epsilon_m| \ll 1$, then
$$
\sum_m (-1)^m z_m = \sum_m \iota(x_m \oplus m)(1+\epsilon_m) = s\,\iota(\pi(x)) + O(\epsilon),
$$
where I've used $(-1)^m = \iota(m \bmod 2)$. So the signed sum recovers the parity character to first order in the deviation from the antipodal manifold.

**The general principle.** The pooling operator must be a **projector onto the irreducible representation of the symmetry group that the task's invariant lives in.** For parity, that's the sign representation of $(\mathbb{Z}_2)^s$, realized as the top Walsh–Hadamard character. For a task with a different invariant (e.g., sum-mod-3, majority), the correct projector is different. There is no universal pooling operator; there is only the operator matched to the task's algebraic structure.

**Falsification test (sharpened from the audit):** Replace $\mathcal{P}_{\text{mean}}$ with $\mathcal{P}_{\text{signed}}$ and re-run. If interior slots remain at chance, the pooling operator is not the bottleneck — the coupling injection point is (§2). If they improve, the audit's Mechanism 1 is confirmed with a *specific* replacement, not a vague "learned projection."

---

## 2. State-Derivative Coupling vs. Perception Concatenation

### 2.1 The current coupling is a perception bias — and that's the wrong intervention point

The current architecture injects the macro signal as
$$
f_\theta(z_i, \mathcal{N}(z_i), w_j) = f_\theta^{(0)}\big([z_i; \mathcal{N}(z_i); w_j]\big),
$$
i.e., $w_j$ is **concatenated onto the perception vector**. In dynamical terms, this makes $w_j$ a **parameter of the vector field**, not a **state variable of the micro dynamics**. The micro cell's *trajectory* is unchanged in structure; only its *instantaneous velocity* is modulated.

For an accumulator, this is the wrong topology. An accumulator needs the macro signal to enter the **state update**, not the **perception**:
$$
z_i^{t+1} = z_i^t + \Delta t \cdot \underbrace{\Big[f_\theta^{(0)}\big(z_i^t, \mathcal{N}(z_i^t)\big) + \gamma\, w_j^t \cdot h_\theta\big(z_i^t\big)\Big]}_{\text{state derivative}},
$$
where $h_\theta$ is a **gain field** (e.g., $h_\theta(z) = 1 - z^2$ for a $\tanh$-bounded state, or a learned scalar). The distinction is not cosmetic:

- **Perception concatenation** asks: "given what you see, what should you do?" The macro signal is *evidence*.
- **State-derivative coupling** asks: "given where you are, how fast should you move?" The macro signal is a *control*.

For a sequential composition task, the macro field must act as a **control signal that gates the accumulation rate**, not as an additional sensory channel. The audit's Mechanism 2 is correct that the coupling is "structurally wrong," but the fix is not just "increase $\gamma$" — it is to **change the injection point from the argument of $f_\theta$ to the coefficient of the state derivative.**

### 2.2 The correct coupled-map formulation

Let me write the coupled system in the form that respects the micro-macro separation of timescales. Let $\epsilon = \Delta t_{\text{macro}}/\Delta t_{\text{mic}}$ be the timescale ratio (in the current setup, $\epsilon = 2$). The proper two-scale CML is:

$$
\begin{aligned}
\text{(micro)} \quad z_i^{t+1} &= z_i^t + \Delta t_{\text{mic}}\Big[f_\theta\big(z_i^t, \mathcal{N}(z_i^t)\big) + \gamma\, \sigma(w_{j(i)}^t)\, h_\theta(z_i^t)\Big],\\
\text{(macro)} \quad w_j^{t+1} &= w_j^t + \Delta t_{\text{macro}}\Big[g_\phi\big(w_j^t\big) + \beta\, \mathcal{P}_{\text{signed}}\big(\{z_{sj+m}^t\}\big)\Big],
\end{aligned}
$$
where $\sigma$ is a bounded squashing (e.g., $\tanh$) and $h_\theta$ is the state-dependent gain. The macro field $w_j$ now enters the micro dynamics as a **multiplicative gain on the state derivative**, which is the correct topology for a control signal.

**Why this matters for the null result.** In the current architecture, the macro field's only route to influence the micro state is through the perception MLP, which is a **high-dimensional, nonlinear, overparameterized map**. The macro signal is one of many inputs; gradient descent will preferentially use the *easier* inputs (the micro state and its neighbors) and treat $w_j$ as noise. This is a **gradient competition** effect: the macro pathway is not "weak" in magnitude, it is **weak in gradient signal-to-noise** because it is competing with the micro state for influence over the same output.

By moving the coupling to the state derivative, the macro signal gets a **dedicated, low-dimensional channel** into the micro dynamics. It no longer competes with the perception MLP; it modulates the *rate* of state change, which is a scalar quantity with a direct, interpretable effect.

### 2.3 The audit's "capacity misallocation" reinterpreted

The audit observed that Arm H boosts Slot 0 and degrades Slot 2. In the coupled-map language, this is a **bifurcation**: the macro coupling has shifted the effective vector field at Slot 0 (where the local feature is easy) into a regime where the micro cell converges faster, while at Slot 2 (where the local feature is hard) it has shifted the field into a regime where the micro cell's attractor is *less* accessible.

This is not "capacity competition" in the ML sense. It is a **parameter-induced bifurcation** of the micro CML. The macro field is acting as a spatially varying bifurcation parameter, and the bifurcation is *not aligned with the task's invariant*. The fix is not to remove the macro field; it is to **align the bifurcation with the parity character** by using the correct pooling operator (§1) and the correct injection point (§2.1).

---

## 3. Continuous Diffusion vs. Discrete Parity: The Algebraic Mismatch

### 3.1 The mismatch is a spectral gap, not a philosophical one

The audit's Mechanism 4 is the deepest, but it states the mismatch qualitatively. Let me make it quantitative.

A continuous diffusion operator $\mathcal{L}$ on a 1D lattice has eigenvalues $\lambda_k = -4\sin^2(\pi k/N)$ (for the discrete Laplacian). The **spectral gap** is $\lambda_1 \sim -4\pi^2/N^2$, and the **highest frequency** is $\lambda_{N/2} \sim -4$. The diffusion operator is a **low-pass filter**: it damps high-$k$ modes fastest.

Parity on $N$ bits is the **top-frequency mode** $k = N/2$. Under diffusion, this mode decays at rate $|\lambda_{N/2}| \sim 4$, which is the **fastest** decay rate in the spectrum. So a continuous diffusion dynamics will **destroy parity information faster than any other mode**. This is not a bug; it is the defining property of diffusion.

**The implication for the hierarchy.** The micro NCA can learn local parity because local parity (3 bits) is a **shallow** function — it can be computed by a feedforward MLP in the perception step, without requiring the diffusion dynamics to preserve the parity mode. But cumulative parity (16 bits) requires the parity mode to **survive across the entire lattice**, which diffusion actively prevents.

**The correct task structure.** To prove genuine emergent synthesis in a 1D continuous NCA, the task must have an invariant that lives in a **low-frequency mode** of the diffusion operator — i.e., a mode that diffusion *preserves* rather than destroys. Candidates:

1. **Iterated running sum (bounded):** $s_t = \tanh(\alpha s_{t-1} + x_t)$. The invariant is the *magnitude* of the sum, which lives at $k=0$ (DC mode). Diffusion preserves DC exactly.
2. **Iterated majority:** $s_t = \text{sign}(\sum_{m} x_{t-m})$. The invariant is the *sign* of a local average, which lives at low $k$. Diffusion preserves low-$k$ modes.
3. **Iterated threshold (smooth):** $s_t = \tanh(\beta(s_{t-1} - \theta))$. The invariant is a *fixed point* of a smooth map, which is a low-frequency attractor.

All three have the same **sequential depth** as iterated parity (16 steps), but their invariants live in the **surviving modes** of the diffusion operator. If the hierarchy succeeds on these and fails on parity, the audit's Mechanism 4 is confirmed with a specific, testable prediction.

### 3.2 The deeper principle: match the task's symmetry to the substrate's symmetry

The failure on parity is a special case of a general principle:

> **A dynamical system can only compute invariants that are compatible with its symmetry group.**

The continuous diffusion CML has symmetry group $(\mathbb{R}, +)$ (translation in state) and $(\mathbb{Z}_N, +)$ (translation in space). Its invariants are **continuous, translation-covariant** functions. Parity is a **discrete, translation-invariant** function — it is invariant under spatial translation but *not* continuous in the state. The mismatch is between the **continuous state symmetry** of the substrate and the **discrete state symmetry** of the task.

The correct task for a continuous CML is one whose invariant is a **continuous function of the state** — i.e., a function that is Lipschitz with respect to the state metric. Parity is not Lipschitz: flipping one bit changes the output by 2, regardless of how small the state perturbation is. A running sum or a majority is Lipschitz: small state perturbations produce small output perturbations.

**This is the falsifiable core of the audit's Mechanism 4.** It is not "parity is hard"; it is "parity is *non-Lipschitz*, and continuous diffusion cannot represent non-Lipschitz invariants without a discrete state variable."

### 3.3 What "genuine emergent synthesis" requires

The user's mandate is "genuine emergent synthesis, no weak mimicry, no central GRU bypass." In dynamical terms, this means:

1. **The macro field must be a genuine dynamical variable**, not a readout. It must have its own attractors, its own timescales, and its own response to the micro state.
2. **The coupling must be bidirectional and load-bearing.** Ablating the macro pathway must change the micro dynamics qualitatively (a bifurcation), not just quantitatively (a small accuracy drop).
3. **The computation must be distributed.** No single cell or single scale should carry the invariant; it should be a **collective mode** of the coupled system.

The current architecture fails (2) and (3): the macro field is a readout (it doesn't feed back into the micro dynamics in a load-bearing way), and the computation is localized to Slot 0 (the easiest subproblem). The fix is to make the macro field a **genuine dynamical variable** with its own attractor structure, coupled to the micro field through the state derivative.

**A concrete proposal.** Let the macro field be a **slow CML** with its own local coupling:
$$
w_j^{t+1} = w_j^t + \Delta t_{\text{macro}}\Big[-\partial_w V(w_j^t) + D_w (w_{j+1}^t - 2w_j^t + w_{j-1}^t) + \beta\, \mathcal{P}_{\text{signed}}(z)\Big],
$$
where $V$ is a double-well potential (so the macro field has **bistable attractors** — the discrete phase the task requires) and $D_w$ is a macro diffusion constant. The macro field now has its own **discrete phase structure** (the two wells), which can represent the parity bit, while the micro field provides the **continuous substrate** for the local computation. This is a **hybrid continuous-discrete** system, but the discreteness is *emergent* from the macro potential, not imposed by a discrete state variable.

This is the correct reading of the user's mandate: not "no discrete state," but "no *externally imposed* discrete state." The macro field's discreteness should emerge from its own dynamics, coupled to the micro field.

---

## 4. Synthesis: The Null Result as a Bifurcation Signature

Pulling the threads together:

1. **The pooling operator $\mathcal{P}_{\text{mean}}$ annihilates the parity character** (top Walsh–Hadamard mode) by construction. The correct operator is the signed sum

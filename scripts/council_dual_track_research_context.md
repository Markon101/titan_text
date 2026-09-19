# High-Context Council Consultation: Dual-Track Campaign Specification (Path A & Path B)

## 1. Executive Context & Objective
Following the completion of the 300-epoch Micro-Macro Hierarchy Campaign (Arm H vs Arm C) which proved the null result ($d = +0.07$) and established the **Spectral Character & Character Annihilation Barrier**, the user has explicitly authorized:
> *"continue agentically and do both. continue in a reasearch loop."*

We are executing two complementary, rigorous tracks:
- **Track 1 (Path A: Character-Matched Cellular Discrete Latching)**:
  Overcome the character annihilation and perception competition bottlenecks on `iterated-parity-dense` by:
  1. Invertible Walsh-Hadamard downsampling ($P_j = [z_{2j} + z_{2j+1}, z_{2j} - z_{2j+1}]$ projected to $C_M$), preserving both DC Hamming weight and AC Nyquist parity character.
  2. State-derivative macro coupling: injecting macro modulation directly into the micro state derivative $\Delta z$ rather than diluting it inside the 192-channel perception MLP.
  3. Emergent bistable macro dynamics: continuous double-well potential drift term $+\lambda w (1 - w^2)$ creating $\pm 1$ latching without external non-differentiable discrete hacks.
- **Track 2 (Path B: Continuous Lipschitz Invariant Transport)**:
  Test the foundational mathematical theorem derived by `dynamics-agent`:
  > Continuous diffusion operators $\mathcal{D}\nabla^2$ naturally preserve low-frequency ($k \to 0$) continuous modes while attenuating high-frequency ($k \to N/2$) parity modes.
  Implement a sequential task whose invariant is Lipschitz continuous in state space (e.g. `IteratedSumDense` / `ChunkedSum` where inputs are $\pm 1$ or digits and the target at query slots is the cumulative bounded running sum or bracket depth), and test whether the multi-scale hierarchy ($s=2$) decisively outperforms the single-scale control ($s=1$) on low-frequency transport!

---

## 2. Technical Architecture Specifications

### Track 1: Path A (Walsh Downsampling + State-Derivative Coupling)
In `src/nca.rs`:
1. **Downsampling**:
   Instead of `pool_mean`, compute:
   $$P_{j, \text{sum}} = z_{2j} + z_{2j+1}$$
   $$P_{j, \text{diff}} = z_{2j} - z_{2j+1}$$
   Concatenate $[P_{\text{sum}}, P_{\text{diff}}] \in \mathbb{R}^{2C}$ and project via a linear layer `macro_down` $\mathbb{R}^{2C} \to \mathbb{R}^{C_M}$.
   This operator is full-rank and invertible: zero character annihilation.
2. **State-Derivative Coupling**:
   Micro perception $p_i$ is computed purely on micro state and its radius-1 neighbors: $p_i = [z_{i-1}, z_i, z_{i+1}] \in \mathbb{R}^{3C}$.
   Micro update:
   $$\Delta z_i = \text{micro\_dense}(p_i) + \gamma \cdot \tanh(\text{inject\_dense}(w_{\lfloor i/s \rfloor})) \odot (1 - z_i^2)$$
   where $\gamma$ is a bounded gain parameter.
   This guarantees a dedicated, un-diluted control channel directly into the micro cell's velocity field!
3. **Bistable Macro Drift**:
   Macro update includes a double-well restoring force:
   $$\Delta w_j = \text{macro\_dense}(p_{M, j}) + \lambda \cdot w_j \odot (1 - w_j^2)$$
   with $\lambda \in [0.0, 0.2]$.

### Track 2: Path B (Continuous Lipschitz Invariant: `IteratedSumDense`)
In `src/tasks.rs`:
Define `IteratedSumDense`:
- Sequence divided into chunks of 4 tokens: 3 step tokens + 1 query token '?'.
- Step tokens are sampled from $\{'-', '0', '+'\}$ (representing increments $\Delta \in \{-1, 0, +1\}$).
- Cumulative running sum $S_{\le i} = \text{clamp}(S_0 + \sum \Delta, 0, 9)$ (mapped to digits '0'..'9').
- At query slot $q_k$, target is $S_{\le k}$.
- **Mathematical Property**: Flipping any single input step changes the target by at most 1 (Lipschitz constant $\le 1$). This invariant lives strictly in the continuous, low-frequency regime ($k \to 0$).

---

## 3. Council Mandate

1. **`dynamics-agent`**:
   - Verify the algebraic stability and convergence of the Walsh downsampler and state-derivative coupling.
   - Prove the Lipschitz bounds of `IteratedSumDense` and confirm that its Fourier decomposition matches the non-attenuating eigenspaces of 1D continuous diffusion.
2. **`adversarial-reviewer`**:
   - Formulate pre-registered decision gates and falsification criteria for both Track 1 (Path A) and Track 2 (Path B).
   - Audit `IteratedSumDense` for trivial baselines (e.g. constant prior, local-only chunk sum, uniform drift). Define what performance level proves genuine multi-scale transport.
3. **`architectural-minimalist`**:
   - Ensure the Candle/Rust implementation adds minimal parameters and preserves CPU throughput on Termux.

# Operational Criteria for Genuine Emergent Synthesis vs. Weak Mimicry

**Laboratory for Recurrent Neural-Cellular Latent Dynamics · Titan Text**  
*Document Version: 1.0 · Date: 2026-09-19 · Repository: `titan_text`*  
*Authors: Antigravity Lead Agent & DeepSeek 4.1 Flash High-Context Council*

---

## 1. The Core Scientific Hazard: Mimicry of Emergence

In neural-cellular sequence modeling, a critical epistemological hazard is **weak mimicry**: modifying an architecture such that benchmark accuracy increases, but where the computation is performed by a centralized shortcut or standard autoregressive component rather than arising from distributed local cellular interactions.

When a centralized sequence model (such as a Gated Recurrent Unit, LSTM, or all-to-all self-attention block) is hybridized with a Cellular Automaton (NCA), gradient descent naturally routes task information through the unconstrained, centralized path. The cellular field is reduced to an idle decorative carrier or static canvas. Calling the resulting system "emergent cellular computation" is a category error.

To prevent self-deception and ensure scientific rigor across all Titan modalities (Text, Image, Audio), any claim of **emergent synthesis** must be validated against the five operational criteria below.

---

## 2. The Five Operational Emergence Criteria

A model configuration exhibits **genuine emergent synthesis** on task $\mathcal{T}$ if and only if it satisfies all five criteria:

### Criterion 1: Load-Bearing Cellular Necessity (Ablation Asymmetry)
- **Test**: At test time, ablate the cellular dynamics by clamping the field update to zero ($\Delta h \equiv 0$) or replacing the cellular trajectory with a constant/mean field, while leaving any auxiliary, decoder, or interface components active.
- **Requirement**: Cellular ablation must destroy task performance:
  $$\Delta_{\text{CA}} = \text{Acc}_{\text{full}} - \text{Acc}_{\text{CA\_ablated}} > 0.5 \cdot (\text{Acc}_{\text{full}} - \text{Acc}_{\text{chance}})$$
- **Falsification Condition**: If ablating the cellular automaton preserves $\ge 50\%$ of above-chance performance, the cellular automaton is not the primary computational engine; the task has been solved by an auxiliary shortcut (Weak Mimicry).

### Criterion 2: Strict Locality Preservation
- **Test**: Audit the receptive field and computation graph of the cellular update operator $\Delta h_i^{(t)}$.
- **Requirement**: The cellular update must depend exclusively on a bounded local stencil $\mathcal{N}_r(i) = \{j \mid |i - j| \le r\}$ where $r \ll L$ (specifically $r=1$ in Titan Text).
- **Prohibited**:
  - Global token-to-token all-to-all attention inside the recurrent step.
  - Dense all-to-all linear mixing layers across the sequence dimension $L$.
  - Unconstrained global broadcast channels that inject instance-specific token information everywhere simultaneously.

### Criterion 3: Absence of Centralized Input Bypass
- **Test**: Trace gradient reachability and information flow from input tokens $x_{0 \dots L-1}$ to the readout head $\hat{y}_k$.
- **Requirement**: No centralized subsystem (such as an RNN, Transformer, or global MLP) may receive the full input sequence and project directly to query positions, bypassing the cellular trajectory.
- **Readout vs. Field Separation**: Readout/decoder heads may pool the *terminal* cellular state $h^{(\tau)}$ to emit predictions, but cannot bypass the dynamical trajectory of the field during latent time $t \in [0, \tau]$.

### Criterion 4: Length Extrapolation & Scale Invariance
- **Test**: Train the model on sequence length $L$ (e.g. $L=16$). Evaluate without retraining on length $L' > L$ (e.g. $L=32, 64$).
- **Requirement**: Genuine local cellular rules are spatially invariant and can execute on larger lattices. While accuracy may degrade due to accumulated noise or boundary effects, the model must maintain above-chance structured computation without dimension mismatch panics.
- **Contrast**: Fixed-width centralized models or models reliant on absolute sequence positions fail immediately on $L' > L$.

### Criterion 5: Dynamic Causal Influence Propagation
- **Test**: Single-cell perturbation mapping $\| \frac{\partial \text{logits}(q_k)}{\partial x_j} \|$ and single-bit flip probability across latent ticks $\tau$.
- **Requirement**: Causal influence from position $j$ to distant query slot $q_k$ must grow monotonically with latent time $\tau$ consistent with the stencil speed of light $\tau \ge |q_k - j| / r$.
- **Falsification Condition**: If influence jumps instantaneously across the lattice at $\tau=1$, or if influence remains permanently frozen in a local radius despite large $\tau$, genuine distributed transport has failed.

---

## 3. Taxonomy of Modality Mechanisms

| Mechanism | Modality Origin | Scientific Status | Risk / Failure Mode | Emergence Gate |
|---|---|---|---|:---:|
| **Centralized GRU reading global state** | Audio (512-unit), Image (128-unit) | **REJECTED FOR TEXT FIELD** | Dominates gradient; turns CA into passive canvas; fails Criteria 1 & 3 | **MIMICRY** |
| **Learned global-to-local writeback** | Image (`RecurrentInterface`) | **REJECTED FOR TEXT** | 100% saturation at gain limit; degenerates into static DC bias | **PATHOLOGY** |
| **1D Micro-Macro Cellular Hierarchy** | Adapted from Image/Audio dual-grids | **APPROVED** | Slower macro clock on $L/2$ lattice; preserves strict local stencil at each scale | **GENUINE** |
| **Zero-Mean Spatial Modulation** | Novel synthesis | **APPROVED** | Enforces $\sum_i w_i = 0$; makes DC spatial bias mathematically impossible | **GUARDRAIL** |
| **Decoder-Side MorphicStack** | Audio (MorphicStack), Image | **APPROVED FOR DECODER** | Zero-initialized contractions ($C_i = 0$); capacity grows only as needed | **PERMITTED** |
| **Reference Withdrawal / Emergent Head** | Image ($z = g + \alpha e$) | **APPROVED** | Separates local chunk scaffold from long-range emergent transport | **CURRICULUM** |

---

## 4. Empirical Protocol for Any Emergence Claim

Before any report or paper draft asserts "emergence" in Titan Text:
1. Run the **CA Ablation Benchmark** ($\Delta_{\text{CA}}$ vs $\Delta_{\text{full}}$).
2. Run the **Length Extrapolation Benchmark** ($L=16 \to L=32$).
3. Run the **Empirical Causal Influence Matrix** (verifying multi-hop transport).
4. Run the **Zero-Tick ($\tau=0$) & Shuffle Controls** (confirming instance-specific recurrent necessity).

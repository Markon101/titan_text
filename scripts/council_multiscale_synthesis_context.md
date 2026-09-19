# High-Context Research State Packet: Multi-Scale Cellular Hierarchy Synthesis

**Laboratory for Recurrent Neural-Cellular Latent Dynamics · Titan Text**  
**Date**: 2026-09-19  
**Platform**: Snapdragon 8 Elite (ARM64 Linux / Termux)  
**Target Architecture**: 1D Neural Cellular Automaton (Candle / Rust)

---

## 1. Executive Scientific Mission & Context

Titan Text investigates whether distributed, strictly local Neural Cellular Automata (NCAs) can perform algorithmic sequence reasoning and emergent linguistic synthesis without relying on global token-to-token all-to-all attention or centralized autoregressive RNNs/Transformers.

Our primary benchmark is **Iterated Prefix Parity ($L=16, \tau=16$)**:
- Input: Sequence of 16 tokens partitioned into 4-token chunks.
- Each chunk contains 3 data bits ($x_0, x_1, x_2 \in \{0, 1\}$) followed by a query token `?` at position 3, 7, 11, 15.
- Target at query slot $k$: The cumulative prefix parity $P_{\le k} = \left(\sum_{j=0}^{4k+2} x_j\right) \bmod 2$.
- Chance accuracy: 50.0%.
- Local chunk parity baseline: If a cell only knows its own chunk's parity, it can achieve ~50% accuracy on cumulative parity. To exceed 50%, the network MUST transport 1-bit parity information across chunks (up to 12 hops from chunk 0 to slot 3).

---

## 2. Completed Experimental Campaign & Empirical Falsifications

Over the preceding research phases, we executed systematic 1,000-epoch controlled learning campaigns across $N=5$ seeds (`42, 101, 202, 303, 404`) with extensive falsification interventions (zero-tick $\tau=0$, state lesion, batch-state shuffle, inverted gain, coordinate counterfactuals, and closed-form linear representation probes).

The results established a clear spatial structure:
- **Slot 0 (pos 3)**: Solved robustly (**93.1% - 100.0%** intact accuracy; $G_{\text{identity}} = +20.0\%$).
- **Slot 3 (pos 15)**: Moderately learned (**66.9% - 73.1%** intact accuracy; $G_{\text{identity}} = +16.9\%$) via periodic boundary wrap-around from chunk 0 (distance = 4 hops backwards).
- **Interior Slots 1 & 2 (pos 7 & 11)**: Strictly trapped at chance (**44.4% - 54.4%** intact accuracy; $G_{\text{identity}} \approx 0.0\%$).

### Three Competing Hypotheses Were Systematically Tested & Decisively Falsified:

1. **$H_{\text{POS}}$ (Spatial Coordinate / Readout Confusion)**:
   - *Hypothesis*: The CA is translation-invariant and cannot distinguish query slot 1 from slot 2.
   - *Intervention*: Injected explicit static 1D spatial coordinate channel $p_i = \frac{2i}{L-1} - 1 \in [-1, 1]$.
   - *Result*: Falsified. Intact accuracy delta was statistically indistinguishable from seed noise ($G_{\text{coord}} = +1.41\%, p = 0.528$). Coordinate counterfactuals (reversed, zeroed, constant, shuffled) caused zero degradation ($\le 0.6\%$).
2. **$H_{\text{READOUT}}$ (Latent Representation Present but Unread)**:
   - *Hypothesis*: The cellular field successfully transported prefix parity to interior cells, but the linear readout head failed to extract it.
   - *Intervention*: Trained analytical closed-form Ridge and non-linear 2-layer MLP probes on latent states $h_7^{(\tau)}$ and $h_{11}^{(\tau)}$.
   - *Result*: Falsified. Latent probes achieved 46.1% - 53.9% accuracy (pure chance). The prefix parity signal is physically absent from interior cells.
3. **$H_{\text{OPT}}$ (Optimization / Credit Assignment Saddle)**:
   - *Hypothesis*: Sparse query loss at positions 3, 7, 11, 15 creates a severe credit-assignment bottleneck across 4 intermediate hops; BPTT gradients through continuous $\tanh$ decay before teaching intermediate cells to act as repeaters.
   - *Intervention* (**Escalation B: Dense Auxiliary Supervision**): Provided explicit running prefix parity labels $P_{\le i}$ to every single cell $i \in [0, 15]$ with auxiliary loss $\mathcal{L} = \mathcal{L}_{\text{sparse}} + \lambda_{\text{aux}} \mathcal{L}_{\text{dense}}$ ($\lambda_{\text{aux}} = 0.5$) for 1,000 full epochs across $N=5$ seeds.
   - *Result*: Decisively Falsified.
     - Slot 1 intact accuracy: **44.38% ± 1.88%** (falsification threshold $\le 55.0\%$).
     - Slot 2 intact accuracy: **50.00% ± 0.00%**.
     - $G_{\text{identity}}$ at Slot 1: **+1.25%**; Slot 2: **+0.00%**.
     - Single-cell causal perturbation: Chunk 0 prefix bits have **0.0% flip probability** on Slot 2 logits; NumSens $\le 0.051$.

### The Incumbent Corroborated Hypothesis: $H_{\text{TRANSPORT}}$
The bottleneck is fundamentally physical/dynamical:
In a 1D continuous-state radius-1 Cellular Automaton with $\tanh$ activations and residual drift ($\Delta h = \alpha \tanh(\dots)$), the intrinsic spatial correlation length is $\xi \approx 1-2$ lattice sites. Continuous diffusion acts as a spatial low-pass filter. Propagating a discrete 1-bit parity signal across 8 to 12 hops requires maintaining sharp non-linear soliton-like binary states across multiple continuous updates without dissipation. The single-scale 1D NCA cannot bridge this transport gap.

---

## 3. Cross-Modality Audit & The Hazard of Weak Mimicry

We audited the architectural memory mechanisms in **Titan Image** and **Titan Audio**:
1. **Titan Image** (`src/interface.rs`, `analysis/EMERGENCE_AUDIT_2026-09-07.md`):
   - Coupled an 8x8 pooled token grid into a 128-unit `GRUCell` with learned writeback to 64x64 micro and 32x32 macro NCA fields.
   - **Pathology Discovered**: 100% of micro and macro writeback elements saturated at 99%+ of the gain limit (0.18). After perturbing the field, the writeback change was near zero ($1.49 \times 10^{-7}$). The GRU writeback had degenerated into a static DC spatial bias that overwhelmed the local NCA.
2. **Titan Audio** (`src/main.rs`, `math.md`):
   - Uses a 512-unit `GRUCell`, but it is strictly **decoder-side** (DDSP spectral residual synthesis). It does NOT write into or drive the cellular field.
3. **The Trap of Weak Mimicry**:
   - If we add a centralized GRU or global self-attention block to Titan Text, the GRU will trivially solve prefix parity in a few dozen updates. The cellular automaton will become an idle decorative canvas ($\Delta_{\text{CA}} \approx 0$).
   - This violates **Emergence Criteria 1 & 3** (`docs/EMERGENCE_CRITERIA.md`). It is fake emergence.

---

## 4. The Proposed Architecture: Option A (1D Micro-Macro Cellular Hierarchy)

To solve $H_{\text{TRANSPORT}}$ while strictly preserving local cellular emergence, we borrow the multi-scale grid structure from Image and Audio with rigorous anti-saturation guardrails:

### Structural Blueprint:
1. **Micro Field**:
   - Lattice: $h \in \mathbb{R}^{B \times L \times C}$ ($L=16, C=64$).
   - Evolving at every latent tick $t \in [1, \tau]$.
   - Radius-1 local stencil: perceives $[h_{i-1}, h_i, h_{i+1}]$.
2. **Macro Field**:
   - Lattice: $M \in \mathbb{R}^{B \times (L/s) \times C_M}$ ($s=2 \implies L_M = 8$, $C_M = 32$).
   - Radius-1 local stencil on the macro grid: perceives $[M_{j-1}, M_j, M_{j+1}]$.
   - Slower Macro Clock: Updates only on ticks $t \equiv 0 \pmod k$ (e.g. $k=2$ or $4$).
3. **Micro-to-Macro Pooling ($D$)**:
   - Local non-overlapping or strided pooling over cells $\{2j, 2j+1\}$:
     $$M_j^{(t)} \leftarrow \text{Pool}(h_{2j}, h_{2j+1})$$
4. **Macro-to-Micro Modulation ($U$) with Anti-Saturation Guardrails**:
   - Upsampling: $U(M)_i = M_{\lfloor i/2 \rfloor}$.
   - **Zero-Mean Spatial Modulation** (eliminating the DC cheat bias from Image):
     $$w_i = U(M)_i - \frac{1}{L} \sum_{m=1}^L U(M)_m$$
     Enforcing $\sum_i w_i \equiv 0$ guarantees that the macro field can only inject spatial relative gradients, never a static uniform DC bias!
   - **Hard-Bounded Coupling Gain**:
     $$\gamma = \gamma_{\max} \cdot \sigma(\tilde{\gamma}), \quad \gamma_{\max} = 0.10$$
     Micro perception receives $\gamma \cdot w_i$ as an auxiliary local channel.
5. **Degenerate Control**:
   - Pre-registered control: $s=1$ (stride 1, no spatial coarsening, $L_M = 16$).
   - If $s=1$ fails while $s=2$ succeeds, it proves that **spatial coarsening and receptive field expansion** is the load-bearing causal mechanism, not merely having two fields or extra parameters.

---

## 5. Objectives for DeepSeek Council Inquiries

We consult three specialized DeepSeek 4.1 Flash subagents:
1. **`dynamics-agent`**: Mathematical analysis of the coupled Micro-Macro Coupled Map Lattice (CML). Proof of contraction/stability, optimal macro timescale ratio $k$, and phase transition dynamics for discrete parity solitons.
2. **`adversarial-reviewer`**: Harsh scrutiny of Option A against the 5 Emergence Criteria. Identify any potential cheat bypasses, subtle leakage, or saturation failure modes. Formulate the minimal falsification tests.
3. **`architectural-minimalist`**: Specify the minimal, zero-bloat Candle/Rust implementation. Keep parameter count low and execution fast on the mobile ARM Snapdragon 8 Elite CPU. Formulate the exact CLI flags and config structures.

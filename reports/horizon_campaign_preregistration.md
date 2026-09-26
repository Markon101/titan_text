# Preregistration Protocol: Causal Recurrence Horizon Campaign (CHCD)

**Laboratory for Recurrent Neural-Cellular Latent Dynamics · Titan Text**  
*Document Version: 1.0 · Date: 2026-09-20 · Repository: `titan_text`*  
*Principal Investigator: Gemini Lead Agent · Council: DeepSeek 4.1 Flash*

---

## 1. Core Mechanistic Question
Does the persistent failure of 1D continuous Neural Cellular Automata (NCAs) to solve bulk interior slots on $L=16$ (Slot 1 at cell 7, Slot 2 at cell 11, Slot 3 at cell 15) stem from **temporal unroll truncation** (insufficient developmental ticks $\tau$ to accommodate the multi-tick continuous settling time required per spatial hop), or from an **intrinsic representational / transport barrier** (continuous diffusion destroying the $\mathbb{Z}_2$ character or channel multiplexing interference)?

---

## 2. Competing Hypotheses & Predictions

### Branch H1: Temporal Settling Latency (The Continuous Dynamic Delay)
- **Mechanism**: Each local chunk requires $\tau_{\text{settle}} \approx 10-14$ continuous integration ticks for cell hidden states to rotate to the correct parity manifold. At $\tau = 16$, the carry from Chunk 0 arrives at Chunk 1 at the very end of development ($t \approx 14-16$), leaving zero time for Chunk 1 to integrate or propagate further.
- **Falsifiable Prediction**: Extending training developmental depth to $\tau = 48$ ticks ($12$ ticks per chunk) will enable sequential carry propagation: Slot 1 will emerge above chance ($>65\%$) at $t \approx 16-24$, followed by Slot 2 at $t \approx 30-36$.

### Branch H2: Channel Multiplexing Interference (Shared Kernel Limitation)
- **Mechanism**: Intermediate carrier cells $i \in \{4, 5, 6\}$ cannot simultaneously preserve their local input token $b_i$ and relay the upstream carry state $x_{i-1}$ using a single translation-equivariant MLP kernel.
- **Falsifiable Prediction**: Increasing $\tau$ to 48 will NOT enable downstream slots; Slot 1 and Slot 2 will remain pinned near 50% chance across all ticks $0..48$.

### Branch H4: Continuous Character Annihilation (Spectral Low-Pass Barrier)
- **Mechanism**: Continuous $\tanh$ updates act as spatial smoothing filters that exponentially attenuate high-frequency antipodal parity signs.
- **Falsifiable Prediction**: As $\tau$ increases to 48, downstream carry decodability decays toward zero, and the identity gap $G_{\text{identity}}$ remains $\le 5\%$.

---

## 3. Experimental Variables & Controls

- **Manipulated Variable**: Developmental unroll horizon $\tau$ during training:
  - Control Arm: $\tau_{\text{train}} = 16$ (Exp 5 CCA baseline, 100 epochs).
  - Experimental Arm: $\tau_{\text{train}} = 48$ (100 epochs).
- **Held-Constant Variables**:
  - Architecture: 1D Causal Cellular Automaton (`--causal-stencil`, `--zero-boundary`).
  - Channels: $C = 64$.
  - Activation: $\tanh$.
  - Step size: $\alpha = 0.5$.
  - Damping / Viscosity: $\nu = 0.0$.
  - State norm: None.
  - Task: `iterated-parity-dense` ($L=16$).
  - Optimizer: AdamW, $\text{lr} = 0.003$.
  - Epochs: 100.
  - Evaluation batch size: $B = 32$.
  - Seeds: $N = 5$ independent seeds (`[42, 101, 202, 303, 404]`).

---

## 4. Evaluation Metrics & Success Criteria

### Primary Metric:
- **Downstream Slot Accuracy**: Mean accuracy on Slot 1 ($q_1 = 7$) and Slot 2 ($q_2 = 11$) at terminal evaluation budget $\tau = 48$.

### Secondary Metrics:
- **Identity Gap ($G_{\text{identity}}$)**: Intact accuracy minus batch-shuffled state accuracy ($G_{\text{identity}} = \text{acc}_{\text{intact}} - \text{acc}_{\text{shuffled}}$) across ticks.
- **Tick-Wise Accuracy Progression**: Per-slot accuracy curves over budgets $\tau \in [0, 4, 8, 12, 16, 24, 32, 40, 48]$.
- **Latent Representation Decodability**: Post-hoc linear and MLP ridge classification accuracy on frozen cell states for:
  1. Local 3-bit chunk parity.
  2. Cumulative prefix parity.
  3. Chunk 0 prefix parity.

### Pre-Registered Decision Thresholds:
- **H1 Corroboration**: Slot 1 accuracy reaches $\ge 65.0\%$ with $G_{\text{identity}} \ge 10.0\%$ at $\tau \ge 24$.
- **H1 Decisive Falsification**: Slot 1 and Slot 2 remain $\le 55.0\%$ (within noise of 50% random chance) at all budgets $\tau \in [16, 24, 32, 40, 48]$ across the 5 seeds.

---

## 5. Potential Confounds & Safeguards

1. **Memory & OOM Safeguard**: Run with `--batch-size 32` and `RAYON_NUM_THREADS=1` to guarantee strict CPU/RAM safety on Termux.
2. **Gradient Vanishing over 48 Ticks**: The causal DAG with residual accumulation has a critical path of $15 + 48 = 63$ steps. We will log the loss and gradient norms to verify that gradients reach input embeddings.
3. **Overfitting to Small Split**: The hash split verification ensures disjoint train/val sets with $2^{12} = 4096$ possible inputs at $L=16$.

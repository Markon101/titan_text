# High-Context Council Consultation: Final Synthesis of the L=16 Spatial Trap & Falsification of H_POS and H_READOUT

## 1. Executive Research Objective & Problem Reconstruction
We are investigating why the 1D Neural Cellular Automaton (NCA) in Titan Text exhibits a severe spatial failure mode when scaling from sequence length $L=8$ (2 query slots) to $L=16$ (4 query slots) on the Iterated Prefix Parity task (`iterated_parity_l16`).

Specifically, in the completed 1,000-epoch controlled IPPR campaigns:
- On $L=8$ (2 query slots: $q_0=3, q_1=7$):
  - Both query slots learn successfully ($>70-80\%$).
  - A large identity-dependent recurrence gap emerges ($G_{\text{identity}} = \text{Acc}_{\text{intact}} - \text{Acc}_{\text{batch\_shuffle}} > 20\%$).
- On $L=16$ (4 query slots: $q_0=3, q_1=7, q_2=11, q_3=15$):
  - Slot 0 ($q_0=3$) learns robustly ($70-80\%$).
  - Slots 1 and 2 ($q_1=7, q_2=11$) remain pinned at chance ($48-54\%$) across all 1,000 epochs.
  - Slot 3 ($q_3=15$) hovers marginally above chance ($55-58\%$).
  - Extended training (up to 1,000 epochs), transfer curriculum from $L=8$, and weight initialization variations all failed to resolve the interior collapse.

The scientific question: **What computation is Titan Text actually performing, and why do the interior query slots fail?**

---

## 2. Architecture & Recurrent Dynamics
- **Input / State Space**:
  - Sequence of discrete tokens $x \in V^L$, where vocabulary $V = \{0, 1, ?\}$.
  - Embedding via pointwise $1\times 1$ linear projection into morphogenic field state $h_0 \in \mathbb{R}^{L \times C}$ ($C=64$ channels).
- **Recurrent Latent Ticks**:
  - Between token arrivals (or during sequence processing), the state evolves over $\tau$ recurrent latent ticks ($t = 0 \dots \tau-1$).
  - Perception stencil: strictly local radius-1 convolution:
    $$z_i^{(t)} = [h_{i-1}^{(t)}, h_i^{(t)}, h_{i+1}^{(t)}] \in \mathbb{R}^{3C} = \mathbb{R}^{192}$$
    (With coordinate channel enabled: $z_i^{(t)} \in \mathbb{R}^{3C+1} = \mathbb{R}^{193}$).
  - Update MLP:
    $$\Delta h_i^{(t)} = W_2 \tanh(W_1 z_i^{(t)} + b_1) + b_2$$
    where $W_1 \in \mathbb{R}^{128 \times 192}$, $W_2 \in \mathbb{R}^{64 \times 128}$.
  - State update with viscous dissipation:
    $$h_i^{(t+1)} = (1 - \nu) h_i^{(t)} + \alpha \Delta h_i^{(t)}$$
    where $\alpha = 0.5$ (alive gate / update rate), $\nu$ is viscous damping.
  - Boundary condition: Strictly zero-padding at physical boundaries ($h_{-1} = 0, h_L = 0$).
- **Readout**:
  - Pointwise $1\times 1$ linear projection: $\hat{y}_i = W_{\text{out}} h_i^{(\tau)} + b_{\text{out}} \in \mathbb{R}^{|V|}$.
  - Supervised only at query positions $q_k = 4k+3$ via cross-entropy loss against cumulative prefix parity $P_k = \bigoplus_{j=0}^{4k+2} b_j$.

---

## 3. The Four Competing Mechanistic Hypotheses

1. **$H_{\text{POS}}$ (Positional Symmetry Deficit)**:
   - Translation-invariant convolution without coordinate encoding prevents cells from distinguishing their absolute position in the sequence, preventing the symmetry breaking needed to establish directed left-to-right transport.
2. **$H_{\text{READOUT}}$ (Readout Failure / Masking)**:
   - The recurrent NCA successfully propagates and integrates prefix parity into the 64-dimensional latent state $h_i^{(\tau)}$ of interior cells, but the linear readout projection $W_{\text{out}}$ fails to extract or decode it due to gradient vanishing or multi-task interference.
3. **$H_{\text{TRANSPORT}}$ (Diffusion Attenuation / Signal Collapse)**:
   - Parity information is attenuated or destroyed across continuous $\tanh$ diffusion hops without discrete latching or macro feedback.
4. **$H_{\text{OPT}}$ (Local Receptive Field Optimization Trap)**:
   - The loss landscape allows greedy independent discovery of local 3-bit chunk parity $p_k = b_{4k} \oplus b_{4k+1} \oplus b_{4k+2}$, yielding a 1-bit signal that provides partial loss reduction. Extending the causal stencil across chunk boundaries requires coordinated multi-cell changes across 4-8 cells that produce zero immediate gradient benefit under sparse query supervision.

---

## 4. Empirical Evidence Accumulated

### A. Pre-Registered Coordinate Campaign (Testing $H_{\text{POS}}$)
- Evaluated $N=5$ seeds over 1,000 epochs with coordinate channel $p_i = \frac{2i}{L-1} - 1 \in [-1, 1]$ appended to perception ($+128$ weights, $+0.29\%$).
- Checkpoints evaluated under full counterfactual battery:
  - `coord_intact`: $57.34\% \pm 3.34\%$
  - `coord_zeroed`: $55.94\% \pm 3.42\%$ ($G_{\text{coord\_zeroed}} = +1.41\% \pm 0.46\%$)
  - `coord_shuffled`: $56.41\% \pm 3.65\%$ ($G_{\text{coord\_shuffled}} = +0.94\% \pm 0.76\%$)
  - `coord_reversed`: $54.38\% \pm 3.21\%$ ($G_{\text{coord\_reversed}} = +2.97\% \pm 0.87\%$)
  - `coord_constant`: $54.84\% \pm 3.15\%$ ($G_{\text{coord\_constant}} = +2.50\% \pm 0.57\%$)
- Per-slot breakdown:
  - Slot 0 ($q_0=3$): Intact $68.75\%$, $G_{\text{coord\_zeroed}} = +6.25\%$, $G_{\text{identity}} = +16.25\%$
  - Slot 1 ($q_1=7$): Intact $48.75\%$, $G_{\text{coord\_zeroed}} = 0.00\%$, $G_{\text{identity}} = +1.25\%$
  - Slot 2 ($q_2=11$): Intact $54.38\%$, $G_{\text{coord\_zeroed}} = -0.63\%$, $G_{\text{identity}} = +2.50\%$
  - Slot 3 ($q_3=15$): Intact $57.50\%$, $G_{\text{coord\_zeroed}} = 0.00\%$, $G_{\text{identity}} = +7.50\%$
- **Decision Rule**: Pre-registered protocol required $G_{\text{coord\_zeroed}} \ge +8.0\%$ and interior slots $\ge 60.0\%$. Both failed decisively.

### B. Empirical Causal Influence & Sensitivity Mapping
- Perturbation of individual input bits $j \in [0..15]$ with measure of gradient sensitivity $\| \frac{\partial \text{logits}(q_k)}{\partial x_j} \|$ and single-bit flip probability:
  - **Slot 0 ($q_0=3$)**: Highly sensitive to bits 0, 1, 2 (FlipPr: 31.2%, 39.1%, 45.3%).
  - **Slot 1 ($q_1=7$)**: Strictly sensitive to bits 4, 5, 6 (FlipPr: 37.5%, 57.8%, 37.5%) and future bits 8..10 (FlipPr: 31.2%, 21.9%).
    - **Sensitivity to prefix bits 0..2**: Numerical sensitivity $\le 0.229$, FlipPr $= 0.0\%$.
  - **Slot 2 ($q_2=11$)**: Strictly sensitive to bits 8, 9, 10 (FlipPr: 46.9%, 50.0%) and future bits 12..14 (FlipPr: 39.1%, 25.0%).
    - **Sensitivity to prefix bits 0..6**: Numerical sensitivity $\le 0.188$, FlipPr $= 0.0\%$.
- **Finding**: Despite theoretical reachability spanning the entire sequence at $\tau=16$, empirical causal influence forms a tight, bounded 7-cell receptive field island $[q_k-3, q_k+3]$. Prefix bits exert zero causal influence on downstream slots!

### C. Latent Representation Probing (Testing $H_{\text{READOUT}}$ vs $H_{\text{TRANSPORT}}$)
Using closed-form analytical Ridge regression (Gaussian elimination) and non-linear Random Feature / ELM MLP probes on hidden state representations $h_{q_k}^{(\tau)}$:
1. **Local Chunk Parity Decoding Accuracy**:
   - Baseline Arm 1: Slot 0: 100.0%, Slot 1: 100.0%, Slot 2: 97.7%, Slot 3: 100.0% (Linear)
   - Coordinate Arm 2: Slot 0: 100.0%, Slot 1: 100.0%, Slot 2: 93.0%, Slot 3: 100.0% (Linear)
   - Non-linear MLP probe yields identical $95-100\%$ accuracy across all slots.
2. **Chunk 0 Prefix Parity Decoding Accuracy (at cells 3, 7, 11, 15)**:
   - Baseline Arm 1:
     - Linear: [100.0%, 44.5%, 54.7%, 52.3%]
     - MLP:    [100.0%, 48.4%, 53.9%, 50.8%]
   - Coordinate Arm 2:
     - Linear: [100.0%, 47.7%, 50.0%, 50.8%]
     - MLP:    [100.0%, 46.1%, 51.6%, 51.6%]
3. **Cumulative Parity Decoding Accuracy**:
   - Linear: [100.0%, 49.2%, 50.0%, 54.7%]
   - MLP:    [100.0%, 47.7%, 50.0%, 57.0%]

---

## 5. Specific Questions for the Council

1. **Falsification Status**:
   - Can we definitively state that $H_{\text{POS}}$ is falsified?
   - Can we definitively state that $H_{\text{READOUT}}$ is falsified, proving that prefix parity does not exist in the latent state of cells 7, 11, 15?
2. **The "Local Receptive Field Trap"**:
   - How does the model achieve $100\%$ local parity representation at every query slot while learning zero horizontal communication? Why does gradient descent settle here?
3. **The Recommended Next Escalation**:
   - Should we pursue **Escalation B: Auxiliary Dense Readout Supervision** (supervising intermediate carrier cells to determine if credit assignment unlocks transport) or **Escalation A: Dual-Timescale Macro Feedback** (providing macroscopic spatial pooling)?
   - What is the most scientifically informative next step?

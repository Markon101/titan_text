# High-Context Council Consultation: Final Empirical Evaluation of Escalation B (Auxiliary Intermediate Supervision)

## 1. Executive Summary & Experimental Execution

Following the user's explicit authorization of **Option B (Escalation B: Auxiliary Intermediate Readout Supervision)**, we have implemented and executed the pre-registered 1,000-epoch campaign on sequence length $L=16$ with task `iterated-parity-dense` (`campaign_arm_b_dense`, $N=5$ seeds, `--zero-boundary`).

### The Pre-Registered Hypothesis & Decision Rule
- **Hypothesis $H_{\text{OPT}}$ (Credit Assignment / Local Saddle Barrier)**:
  Asserts that interior slots 1 and 2 collapsed at $50\%$ chance solely because intermediate carrier cells received zero direct loss gradients under sparse query supervision at $\{3, 7, 11, 15\}$.
  - *Prediction*: Providing dense auxiliary supervision to intermediate carrier cells on running prefix parity $P_{\le i}$ with loss weight $\lambda_{\text{aux}} = 0.5$ will provide direct gradient flow along the transport direction, escaping the local receptive field saddle and elevating interior slot accuracy to $\ge 65.0\%$ with prefix bit FlipPr $\ge 15.0\%$.
- **Hypothesis $H_{\text{TRANSPORT}}$ (Intrinsic Dynamical Transport Collapse)**:
  Asserts that continuous radius-1 NCA diffusion with local $\tanh$ updates physically cannot sustain 1-bit discrete latching across $\ge 4-8$ cells without macroscopic spatial coupling, regardless of gradient guidance.
  - *Falsification Criterion for $H_{\text{OPT}}$ / Confirmation of $H_{\text{TRANSPORT}}$*: If interior slots remain $\le 55.0\%$ across 1,000 epochs despite dense supervision.

---

## 2. Empirical Findings from the 1,000-Epoch Campaign (`campaign_arm_b_dense.json`)

1. **Intact Query Slot Accuracies Across Checkpoints (Evaluated on $N=5$ seeds strictly at query slots $\{3, 7, 11, 15\}$)**:
   - Epoch 100: Intact Acc: $56.56\% \pm 6.64\%$ [Slot 0: 71.9%, Slot 1: 45.6%, Slot 2: 48.8%, Slot 3: 60.0%]
   - Epoch 300: Intact Acc: $56.72\% \pm 5.32\%$ [Slot 0: 73.8%, Slot 1: 45.6%, Slot 2: 47.5%, Slot 3: 60.0%]
   - Epoch 500: Intact Acc: $55.78\% \pm 5.97\%$ [Slot 0: 74.4%, Slot 1: 41.9%, Slot 2: 46.9%, Slot 3: 60.0%]
   - Epoch 700: Intact Acc: $56.09\% \pm 6.97\%$ [Slot 0: 72.5%, Slot 1: 46.2%, Slot 2: 48.8%, Slot 3: 56.9%]
   - Epoch 1000: Intact Acc: $55.78\% \pm 7.53\%$ [Slot 0: 71.9%, Slot 1: 44.4%, Slot 2: 50.0%, Slot 3: 56.9%]

2. **Identity-Dependent Recurrence Gap $G_{\text{identity}} = \text{Acc}_{\text{intact}} - \text{Acc}_{\text{batch\_shuffle}}$**:
   - Epoch 1000: Overall $G_{\text{identity}} = +7.19\% \pm 4.21\%$
   - Per-slot $G_{\text{identity}}$: Slot 0: **+20.0%**, Slot 1: **+1.2%**, Slot 2: **+0.0%**, Slot 3: **+7.5%**.

3. **Latent Hidden State Probing ($\tau=16$) at Epoch 1000**:
   - **Local Chunk Parity Accuracy**:
     - Linear Ridge: [98.4%, 96.9%, 98.4%, 100.0%]
     - Non-linear MLP: [100.0%, 96.9%, 96.9%, 100.0%]
     *(Local 3-bit parity is computed with near-perfect ~97-100% precision at every cell)*.
   - **Chunk 0 Prefix Parity at Interior Cells $\{3, 7, 11, 15\}$**:
     - Linear Ridge: [98.4%, 45.3%, 63.3%, 56.2%]
     - Non-linear MLP: [100.0%, 49.2%, 53.1%, 53.9%]
     *(Chunk 0 prefix parity remains pinned near random chance at interior cells)*.
   - **Cumulative Prefix Parity at Interior Cells**:
     - Linear Ridge: [98.4%, 48.4%, 44.5%, 49.2%]
     - Non-linear MLP: [100.0%, 51.6%, 49.2%, 55.5%]
     *(Cumulative parity is identically random chance ~50% at interior cells)*.

4. **Empirical Causal Influence Mapping at Epoch 1000**:
   - For Query Slot 1 (pos 7):
     - Input bit 0 (pos 0): NumSens = 0.777, FlipPr = **1.6%**
     - Input bit 1 (pos 1): NumSens = 2.702, FlipPr = **1.6%**
     - Input bit 2 (pos 2): NumSens = 8.634, FlipPr = **12.5%**
     - Input bits 4..6 (local chunk 1): NumSens = 23.5 - 44.8, FlipPr = **28.1% - 57.8%**
   - For Query Slot 2 (pos 11):
     - Input bits 0..2 (chunk 0): NumSens $\le 0.051$, FlipPr = **0.0%**
     - Input bits 4..6 (chunk 1): NumSens = 0.63 - 10.7, FlipPr = **1.6% - 18.8%**
     - Input bits 8..10 (local chunk 2): NumSens = 24.3 - 45.5, FlipPr = **35.9% - 59.4%**

---

## 3. Specific Questions for the Falsification Arbiter

1. **Formal Falsification Verdict**:
   - Does this empirical evidence definitively falsify $H_{\text{OPT}}$ (that credit assignment / sparse supervision alone was the root cause)?
   - Can we now state with scientific certainty that the 1D single-scale continuous NCA suffers from an intrinsic physical transport collapse ($H_{\text{TRANSPORT}}$)?
2. **Mechanistic Explanation of the Dynamical Barrier**:
   - Why does continuous $\tanh$ diffusion across radius-1 stencils fail to propagate 1-bit discrete states over 8–16 hops even when every intermediate cell is explicitly supervised with cross-entropy loss on the running parity?
3. **The Definitive Path Forward**:
   - Now that $H_{\text{POS}}$, $H_{\text{READOUT}}$, and $H_{\text{OPT}}$ have all been systematically and empirically falsified, why is the **1D Micro-Macro Cellular Hierarchy** (Option A) the necessary and sufficient architectural solution?

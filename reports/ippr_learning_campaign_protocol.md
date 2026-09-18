# Controlled IPPR Learning Campaign Protocol
**Titan Text · Recurrent Neural-Cellular Sequence Laboratory**  
*Date: 2026-09-18*

---

## 1. Executive Framing & Pre-Registered Objective

Our previous $N=5$ statistical evaluation demonstrated that while latent recurrence is causally necessary for any non-zero prediction in the $v_0$ checkpoint ($p < 10^{-6}, d = 15.56$), the **identity-dependent recurrence gap** between intact and batch-shuffled recurrence is statistically indistinguishable from seed noise ($G_{\text{identity}} = +1.88\% \pm 4.34\%, p = 0.388$).

**Conservative Interpretation**:
> Recurrence is necessary for the checkpoint's learned output behavior, but sample-specific recurrent-state identity has not been shown necessary for its weak task performance. The current checkpoint operates as a generic non-linear conditioner and local feature extractor rather than an instance-specific sequential memory register.

**Campaign Goal**:
Execute a controlled, extended learning campaign (5–10× current duration) to empirically determine whether Titan Text can undergo a phase transition from generic recurrence-dependent conditioning to genuine instance-specific recurrent computation.

---

## 2. Synthesis of DeepSeek Subagent Reviews

Before launching, three independent DeepSeek 4.1 Flash subagents audited the proposed protocol:

### A. Task Auditor (`task-auditor`)
1. **Light Cone & Positional Specialization**: In IPPR, chunk $c$ has query token at $4c+3$. Under fixed $L$, query positions are $\{3, 7, 11, \dots\}$. Position 3 is never de-specialized, explaining why slot 0 showed $57.5\%$ accuracy in both intact and shuffled recurrence in the initial checkpoint.
2. **Periodic Boundary Leakage**: Circular spatial roll in 1D NCA allows information to wrap between $0$ and $L-1$. The protocol must track whether the final query slot leverages wrap-around shortcuts.
3. **Curriculum Risk**: An easy $L=8$ stage risks teaching a static 2-head lookup that must be unlearned. The campaign must run a matched from-scratch baseline alongside the curriculum.

### B. Experiment Designer (`experiment-designer`)
1. **Light-Cone Bound**: In 1D NCA with radius-1 convolution, information travels $\le 1$ cell per latent tick $\tau$.
   - Chunk 0 ($k=0$, index 3): distance 3 $\implies$ reachable in $\tau \ge 3$.
   - Chunk 1 ($k=1$, index 7): distance 7 $\implies$ reachable in $\tau \ge 7$.
   - Chunk 2 ($k=2$, index 11): distance 11 $\implies$ reachable in $\tau \ge 11$.
   - Chunk 3 ($k=3$, index 15): distance 15 $\implies$ reachable in $\tau \ge 15$.
   *Crucial finding*: The previous evaluation at $\tau=4$ was **physically outside the light cone** for slots 1, 2, and 3. Testing at $\tau=4$ guaranteed chance performance on distant slots regardless of learning!
2. **Latent-Tick Training Schedule**: Training must provide sufficient horizon ($\tau \ge 16$) so that all positions fall within the causal light cone.

### C. Falsification Arbiter (`falsification-arbiter`)
1. **Two Competing Hypotheses**:
   - *Claim A (Dead Run / Representational Bottleneck)*: $G_{\text{identity}}$ remains flat ($< 3\%$) as training continues because the architecture cannot maintain multi-slot registers.
   - *Claim B (Pre-Critical Phase Transition)*: IPPR requires crossing a critical threshold, after which $G_{\text{identity}}$ jumps steeply and concentrates in later slots.
2. **Decisive Metric**: Identity-dependent recurrence gap:
   $$G_{\text{identity}} = \text{Acc}_{\text{intact}} - \text{Acc}_{\text{batch\_shuffle}}$$
   both overall and per query position $k \in \{0, 1, 2, 3\}$.

---

## 3. Pre-Registered Stopping & Escalation Criteria

To prevent indefinite training if the task is not being learned, the campaign enforces strict stopping and escalation gates:

### Early Stopping Criteria (Halt Run)
Training will be terminated early if any of the following occur:
- **S1 (Flat Identity Gap)**: $G_{\text{identity}} < 3.0\%$ raw and the slope of $G_{\text{identity}}$ over the last 3 checkpoints is $< 0.5\%$ per checkpoint, sustained across **4 consecutive checkpoints** after step 400.
- **S2 (Loss Plateau Without Gap Growth)**: Validation loss relative improvement is $< 1.0\%$ over 4 checkpoints while $G_{\text{identity}} < 3.0\%$.
- **S3 (Slot Profile Collapse)**: $G_{\text{identity}}$ remains uniformly $< 3.0\%$ across all individual query slots past the midpoint of the campaign.

### Escalation Criteria (Declare Transition & Extend)
A phase transition to genuine instance-specific recurrence is declared when **all** of the following hold:
- **E1 (Decisive Gap)**: $G_{\text{identity}} \ge 15.0\%$ overall.
- **E2 (Ablation Separation)**: Intact accuracy exceeds both Zero-Tick and State Lesion by $\ge 40.0\%$, and exceeds Batch-State Shuffle by $\ge 12.0\%$.
- **E3 (Distant Slot Solvability)**: Later query slots (slots 2 and 3) exceed $65.0\%$ accuracy, and $G_{\text{identity}, k}$ increases with dependency distance $k$.
- **E4 (Persistence)**: Conditions E1–E3 hold across **2 consecutive intermediate checkpoints**.

---

## 4. Experimental Arms

1. **Arm 1: Matched From-Scratch ($L=16$)**
   - Sequence length $L=16$ (4 chunks).
   - Recurrence horizon: `dev_steps = 16` (full light-cone reachability across all 16 cells).
   - Training duration: 1,000 epochs (5× current duration), saving intermediate checkpoints every 100 epochs.
2. **Arm 2: Staged Curriculum ($L=8 \to L=16$)**
   - Stage 1: $L=8$ (2 chunks), `dev_steps = 8`, trained for 500 epochs.
   - Stage 2: Transfer weights to $L=16$ (4 chunks), `dev_steps = 16`, trained for 500 epochs.
   - Matched total training: 1,000 epochs.

---

## 5. Evaluation Battery at Each Intermediate Checkpoint

At every 100-epoch checkpoint, evaluate:
1. **Intact Latent-Tick Sweep**: $\tau \in \{0, 1, 2, 4, 8, 12, 16\}$ to measure compute-gain regimes and check for trajectory divergence.
2. **Zero-Tick Baseline ($\tau=0$)**: No recurrence updates.
3. **Total State Lesion ($\tau=16, \text{disable\_recurrent}=\text{true}$)**: Recurrent updates zeroed.
4. **State Inversion Gain ($\tau=16, \text{recurrence\_gain}=-1.0$)**: Causal necessity inversion swap.
5. **Batch-State Shuffle ($\tau=16, \text{shuffle\_batch}=\text{true}$)**: Temporal identity destroyed via cyclic permutation.
6. **Sham Perturbation ($\tau=16, \sigma=0.05$)**: Matched small additive Gaussian state noise to separate identity destruction from generic perturbation sensitivity.
7. **Per-Query-Slot Decomposition**: Accuracy reported individually for Slot 0 (dist 3), Slot 1 (dist 7), Slot 2 (dist 11), and Slot 3 (dist 15).
8. **Recurrence Gaps**:
   $$G_{\text{identity}} = \text{Acc}_{\text{intact}} - \text{Acc}_{\text{batch\_shuffle}}$$
   $$G_{\text{sham}} = \text{Acc}_{\text{intact}} - \text{Acc}_{\text{sham\_noise}}$$
   $$G_{\text{lesion}} = \text{Acc}_{\text{intact}} - \text{Acc}_{\text{lesion\_state}}$$

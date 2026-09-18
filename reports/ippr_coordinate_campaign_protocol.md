# Pre-Registered Protocol: IPPR Coordinate-Channel Campaign & Boundary-Relocation Experiment
**Laboratory for Recurrent Neural-Cellular Latent Dynamics · Titan Text**  
*Protocol Version: 1.0 · Date: 2026-09-18 · Repository: `titan_text`*

---

## 1. Context & Audited Causal Geometry

### The Resolved Light-Cone Anomaly
The source-level causal geometry audit (`docs/ippr_causal_geometry_audit.md`) proved:
1. The radius-1 stencil strictly obeys physical reachability $|i - j| \le \tau$. At $\tau=2$, numerical sensitivity from cell 0 to cell 7 is strictly $0.0000$ with $0.0\%$ flip probability.
2. The apparent $71.3\%$ accuracy at $\tau=2$ on $L=8$ was driven by local 2-bit parity ($b_5 \oplus b_6$), which exhibits finite-sample correlation on the 11 validation sequences of $L=8$.
3. On $L=16$, the model suffers from a **Local Receptive Field Trap**: each query slot $q_k = 4k+3$ establishes a symmetric local sensitivity island spanning $[q_k - 3, q_k + 3]$. Slot 0 ($q_0 = 3$) captures its prefix bits $[0..2]$ and achieves $77.5\%$. Slot 1 ($q_1 = 7$) and Slot 2 ($q_2 = 11$) have virtually zero sensitivity to prefix bits ($\le 0.012$), causing permanent collapse to $50\%$ chance.

---

## 2. Competing Mechanistic Hypotheses

1. **$H_{\text{POS}}$ (Positional / Symmetry Deficit)**:
   - Translation-invariant convolution without coordinate encoding makes interior cells indistinguishable, preventing them from breaking symmetry to establish directed left-to-right state transport.
   - *Prediction*: Providing an explicit coordinate channel $p_i = \frac{2i}{L-1} - 1$ will break symmetry and allow interior query slots (1 and 2) to establish transport and rise significantly above chance ($G_{\text{slot}} \ge +10\%$).

2. **$H_{\text{TRANSPORT}}$ (Signal Attenuation / Diffusion Dilution)**:
   - Continuous $\tanh$ recurrence over 11–15 hops dilutes the 1-bit parity signal below the numerical noise floor without discrete latching or macro feedback.
   - *Prediction*: Coordinate channels will not rescue interior slots ($G_{\text{slot}} \approx 0\%$), because the bottleneck is signal survival across multi-hop diffusion.

3. **$H_{\text{READOUT}}$ (Readout Masking & Gradient Sparsity)**:
   - Supervising only query positions $4k+3$ creates an exponential gradient vanishing problem for intermediate data cells acting as communication repeaters.
   - *Prediction*: Coordinating input alone will be insufficient unless accompanied by auxiliary supervision.

4. **$H_{\text{OPT}}$ (Optimization Landscape)**:
   - Early convergence on local chunk bits traps SGD in a local minimum where extending the receptive field across chunk boundaries yields zero immediate gradient.

---

## 3. Experimental Arms & Interventions

### Arm 1: Baseline Matched From-Scratch $L=16, \tau=16$ (Already executed control)
- Architecture: Standard 1D NCA ($C=64$, dense1 width 128, local radius 1).
- Training: 1,000 epochs, AdamW ($\text{lr}=0.003$), zero boundary.

### Arm 2: Coordinate-Channel From-Scratch $L=16, \tau=16$ (`--coord-channel`)
- Intervention: Append a static 1D coordinate channel $p_i = \frac{2i}{L-1} - 1 \in [-1, 1]$ into the NCA perception vector at every tick $t$.
- Parameter change: Exactly $+1$ input weight per dense1 neuron ($+128$ weights out of 43,715; $+0.29\%$).
- Training: 1,000 epochs, AdamW ($\text{lr}=0.003$), `--zero-boundary`, checkpoints every 100 epochs.

---

## 4. Coordinate Counterfactual Evaluation Battery

To prevent declaring a false positive from a static lookup shortcut, every checkpoint will be evaluated under the full suite of **Coordinate Counterfactuals**:

1. **`coord_intact`**: Correct linear coordinate $p_i = \frac{2i}{L-1} - 1$.
2. **`coord_zeroed`**: $p_i = 0.0$ across all cells (ablation).
3. **`coord_shuffled`**: Spatially permuted coordinates $p_{\pi(i)}$ (destroys ordering while preserving marginal values).
4. **`coord_reversed`**: Direction reversed $p_i \to -p_i = \frac{2(L-1-i)}{L-1} - 1$ (tests left-to-right directional dependence).
5. **`coord_constant`**: Homogeneous coordinate $p_i = 0.5$ (tests whether variation across cells is required).

### Standard Causal Controls (Retained at Every Checkpoint)
- **Zero-Tick ($\tau=0$)**: Tests whether the model became a feedforward shortcut.
- **Batch-State Shuffle**: Tests instance-specific recurrent identity dependence ($G_{\text{identity}}$).
- **State Lesion**: Complete ablation of recurrence ($x \to 0$).
- **Latent-Tick Sweep ($\tau \in [0, 1, 2, 4, 8, 12, 16]$)**: Evaluates compute-monotonicity.
- **Per-Slot Accuracy ($s_0, s_1, s_2, s_3$)**: Tracks spatial resolution.

---

## 5. Statistical Protocol & Sample Sizing

- **Independent Seeds**: $N=5$ (`42, 101, 202, 303, 404`).
- **Batch Size**: 64 (large enough to avoid small-sample binomial distortion).
- **Metrics**:
  - $G_{\text{identity}} = \text{Acc}_{\text{intact}} - \text{Acc}_{\text{batch\_shuffle}}$
  - $G_{\text{coord\_ablation}} = \text{Acc}_{\text{coord\_intact}} - \text{Acc}_{\text{coord\_zeroed}}$
  - $G_{\text{coord\_shuffle}} = \text{Acc}_{\text{coord\_intact}} - \text{Acc}_{\text{coord\_shuffled}}$
  - $G_{\text{coord\_reversed}} = \text{Acc}_{\text{coord\_intact}} - \text{Acc}_{\text{coord\_reversed}}$
  - Paired Cohen's $d$, paired standard error of the mean (SEM), and two-tailed paired $t$-test.

---

## 6. Pre-Registered Decision Criteria

### Success Criteria (Confirms $H_{\text{POS}}$)
1. **Interior Rescue**: Both Slot 1 and Slot 2 achieve $\ge 60.0\%$ intact accuracy ($G_{\text{slot}}[1] \ge +10.0\%$ and $G_{\text{slot}}[2] \ge +10.0\%$) across at least 2 consecutive checkpoints.
2. **Coordinate Sensitivity**:
   - $G_{\text{coord\_ablation}} \ge +8.0\%$ with $p < 0.01$.
   - $G_{\text{coord\_shuffle}} \ge +8.0\%$ with $p < 0.01$.
3. **Recurrent Necessity**:
   - Zero-tick accuracy remains near chance ($\le 55.0\%$).
   - State lesion collapses accuracy to $0.0\%$.
   - Batch-state shuffle destroys the performance advantage ($G_{\text{identity}} \ge +10.0\%$).

### Failure Criteria (Falsifies $H_{\text{POS}}$ / Confirms $H_{\text{TRANSPORT}}$)
1. **Interior Flatness**: Both Slot 1 and Slot 2 remain pinned below $53.0\%$ ($G_{\text{slot}} \le +3.0\%$) after 500 epochs of coordinate training.
2. **Escalation**: If $H_{\text{POS}}$ is falsified, escalate to macro recursive feedback (`--feedback-mode dual_timescale`) or dense auxiliary supervision ($H_{\text{READOUT}}$).

### Shortcut / Artifact Criteria (Falsifies Recurrent Mechanism)
1. If accuracy increases but **Zero-Tick ($\tau=0$) accuracy also rises to $\ge 65.0\%$**, the coordinate channel created a feedforward positional shortcut rather than recurrent computation.
2. If accuracy increases but **Batch-State Shuffle is indistinguishable from intact ($G_{\text{identity}} < 3.0\%$)**, the coordinate channel enabled generic static conditioning without instance-specific recurrence.

# Canonical Research-State Packet: Titan Text
**Laboratory for Recurrent Neural-Cellular Latent Dynamics & Formal Pushdown Computation**  
*Document Version: 3.2 · Date: 2026-09-26 · Repository: `titan_text`*  
*Governed by: Evidence-First Protocol & Jev System-1 Cognitive Governor*

---

## 1. ESTABLISHED (Supported by Definite Quantitative Evidence)

1. **Causal Necessity of Latent Recurrent Updates ($d=15.56$)**:
   - On `TaskKind::IteratedParity` ($L=16, B=32$), ablating recurrent latent ticks (Zero-Tick $\tau=0$ or State Lesion `--lesion-state`) collapses task accuracy from $53.28\% \pm 3.42\%$ to $0.00\%$ and explodes loss by $+3.35$ nats ($t(4)=34.80, p=4.07 \times 10^{-6}$, paired Cohen's $d=15.56$). Reversing the update vector ($\Delta x \to -\Delta x$) causes catastrophic divergence ($14.13$ nats, $d=-75.44$).
   - *Source*: [`reports/eval_n5_lesion_state.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/eval_n5_lesion_state.json).

2. **Instance-Specific Recurrent Parity on Compact Horizons ($L=8, \tau=8, d=1.90$)**:
   - 1D NCA learns sample-specific sequential parity on $L=8, \tau=8$ with an identity gap $G_{\text{identity}} = +18.8\% \pm 4.4\%$ across both slots (Slot 0: $68.8\%$, Slot 1: $68.8\%$ intact vs $50.0\%$ shuffled, $d=1.90$).
   - *Source*: [`reports/campaign_arm2_curriculum.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/campaign_arm2_curriculum.json).

3. **Empirical Correlation Length Limit of Continuous 1D NCA ($\xi \approx 4-6$ cells)**:
   - On $L=16, \tau=16..48$, Slot 0 ($k=0$, cell 3) is robustly solved ($71-86\%$), but interior slots 1 and 2 remain pinned at chance ($44-50\%$) across 1000 epochs, curriculum, coordinate channels, and single-slot supervision.
   - Carry information reliably spans 1 chunk (4 cells, $91.4\%$ MLP probe acc), but collapses to $50.8\%$ chance across 2 chunks (8 cells), invariant to unroll depth $\tau \in [16..48]$.
   - *Source*: [`reports/campaign_horizon_48.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/campaign_horizon_48.json), [`reports/escalation_b_final_report.md`](file:///data/data/com.termux/files/home/projects/titan_text/reports/escalation_b_final_report.md).

4. **Explicit Carry Register (ECR) Decouples Local Perception from Carry Propagation**:
   - Titan NCA with ECR ($T=16, C_c=16$, causal DAG) achieves $100.0\%$ train and $76.6\%$ validation accuracy on iterated parity ($L=16$), surpassing Transformer ($55.5\%$), GRU ($55.5\%$), and baseline NCA ($71.9\%$).
   - On iterated-sum-dense ($L=16$), ECR lifts validation accuracy from $27.3\%$ to $70.3\%$ (matching Simple RNN $70.3\%$, beating Transformer $43.8\%$).
   - *Source*: [`reports/ecr_campaign_final_report.md`](file:///data/data/com.termux/files/home/projects/titan_text/reports/ecr_campaign_final_report.md).

5. **Discrete STE Quantization Eliminates Continuous Dissipation Drift**:
   - Discrete carry quantization (`ste_sign`) achieves higher in-distribution accuracy at $L=16$ ($58.0\% \pm 1.4\%$ vs $56.1\% \pm 2.3\%$) and eliminates sub-random drift at $L=64$, holding rock-solid stability at $50.5\% \pm 0.1\%$ where continuous baselines decay to $49.1\% \pm 2.8\%$ and bistable potential collapses to $45.8\% \pm 7.3\%$.
   - *Source*: [`reports/drift_mitigation_campaign.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/drift_mitigation_campaign.json).

6. **Chomsky Type-2 Pushdown Emulation on Dyck-4 ($43.90\%$ vs $0.098\%$ Markov Ceiling)**:
   - CD-DV-NCA achieves $43.90\% \pm 1.61\%$ exact bracket prediction on Dyck-4 at $L=64$, beating the Markov-4 ceiling ($0.098\%$) by $448\times$.
   - Carry lesion stratified depth evaluation confirms discrete carry channels mediate non-local matching: at depth $D=4$, carry lesion causes a $-12.50\%$ drop ($50.5\% \to 38.0\%$), and carry scramble collapses to chance ($26.0\%$).
   - Continuous channels alone collapse to chance ($26.9\% \approx 25.0\%$) at $D \ge 12$.
   - *Source*: [`reports/dyck_pushdown_benchmark_results.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/dyck_pushdown_benchmark_results.json), [`reports/stratified_depth_benchmark_results.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/stratified_depth_benchmark_results.json).

7. **Ballistic Causal Reach Horizon Law ($T^* \ge 2\lceil L/k \rceil$)**:
   - Information roundtrip transport strictly requires horizon $T^* \ge 2\lceil L/k \rceil$. At $L=64, T=24, k=4$, causal reach is 48 cells ($75\%$). Deep brackets ($D \ge 12$) are causally severed, capping theoretical accuracy at $46.08\%$. Intact and lesion models both collapse to chance ($26-29\%$) at $D \ge 12$.
   - *Source*: [`reports/stratified_depth_benchmark_results.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/stratified_depth_benchmark_results.json).

8. **Bit-Level Translation Invariance Under Transport Conjugation ($\Delta \|H\| \le 0.0023, p < 10^{-6}$)**:
   - Under spatial roll (`roll_spatial_4`), continuous field norm $\|H\|$ exhibits bit-level translation invariance ($\Delta \le 0.0023 < 0.02$, TOST $p < 10^{-6}$) across all $t^* \in \{8, 12, 16\}$ and depths $D \in \{4, 8\}$.
   - Channel permutations induce coordinate perturbation ($\Delta = 0.35$ at $t^*=8$) that monotonically contracts to $\Delta \le 0.05$ at $t^*=16$ as recurrence approaches the fixed-point basin.
   - *Source*: [`reports/transport_conjugation_benchmark_results.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/transport_conjugation_benchmark_results.json).

9. **Orthogonal Contractive-Pushdown Decomposition (OCPD) Subspace Normalization**:
   - Restricting bounded normalization to continuous subspace $H$ (`bounded_h_only`, channels $0..31$) bounds continuous field drift, reducing loss by $47.3\%$ at $L=64$ ($5.13 \to 2.70$) and $71.1\%$ at $L=128$ ($12.45 \to 3.59$) while maintaining $30.9\%$ bracket accuracy.
   - Uniform full-channel norm (Arm C) squashes discrete carry amplitudes, collapsing accuracy to uniform chance ($25.8\% \approx 25.0\%$).
   - *Source*: [`reports/subspace_normalization_ocpd_results.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/subspace_normalization_ocpd_results.json).

10. **Double-Dissociated Distance-Resolved Non-Local Pushdown Computation on Adversarial Balanced Dyck-4**:
    - Under strict uniform bigram balancing and non-local transport gaps $G \in \{0, 4, 8\}$, CD-DV-NCA maintains $36.2\% - 58.3\%$ bracket prediction across depths $D \in \{2, 4, 8\}$.
    - *Double Dissociation across Physical Distance $d = G + 2 + 2s$*:
      - At local distance $d \le 4$ cells: continuous channels alone achieve $85.4\% - 93.8\%$ accuracy, with minimal carry delta ($\Delta_{\text{carry}} = +4.1\%$).
      - At non-local distance $d \in [6..24]$ cells: continuous channels collapse to flat chance floor ($24.7\% \approx 25.0\%$), while discrete carry channels sustain retrieval with causal gains of $+14.6\%$ to $+25.0\%$ (peaking at $50.0\%$ at distance 20 cells).
      - Under $G=8$ (all queries at $d \ge 10$ cells), Untied FF ($25.8\%$), Scramble Derangement ($24.2\%$), and Carry Lesion ($24.7\%$) all collapse to chance, while intact CD-DV-NCA sustains $36.2\%$ ($\Delta = +11.5\%$).
    - *Source*: [`reports/adversarial_balanced_dyck_benchmark_results.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/adversarial_balanced_dyck_benchmark_results.json).

11. **Autoregressive Structured ASCII Generation via Causal Recurrence ($\tau=4, 8$)**:
    - A tiny recurrent NCA text model (~43,844 parameters) operating autoregressively with causal 1D stencil perception generates multiline ASCII boxes and motifs.
    - *Causal Necessity of Recurrence*: Ablating recurrent latent ticks ($\tau=0$ or `--lesion-state`) collapses token emission immediately to `<eos>`, $\tau=2$ produces crude repetitions (`#####`), while $\tau=4$ and $\tau=8$ reliably generate multiline boxes with borders (`+====+`), walls (`:`), and clean whitespace ($63.5\%$ horizontal symmetry).
    - *Generalization vs Memorization*: Across 96 post-training evaluations on fixed seeds (`42, 101, 202, 303, 404`), exact training match count is 0/96 ($0.0\%$ memorization), with mean nearest edit similarity shifting from $0.081$ (untrained noise) to $0.391$ (structured grammar).
    - *Source*: [`reports/ascii_generative_campaign_report.md`](file:///data/data/com.termux/files/home/projects/titan_text/reports/ascii_generative_campaign_report.md), [`runs/ascii/ascii_v1_trained_1790462724/`](file:///data/data/com.termux/files/home/projects/titan_text/runs/ascii/ascii_v1_trained_1790462724/).

12. **Adaptive Recurrent Compute / Per-Token Halting Dynamics ($\tau_{eff} \approx 2.35$, +10.4% Pareto Gain vs +0.0001 Lower Bound)**:
    - Dynamic relative velocity halting ($\|x_t - x_{t-1}\|_2 / (\|x_t\|_2 + \epsilon) \le 0.35$, patience 2) converges to a mean compute budget of $\bar{\tau} = 2.35$ ticks per token across 135 runs.
    - *Pareto Efficiency*: At $\tau = 2.35$, adaptive halting achieves nearest edit similarity $0.4679 \pm 0.0381$, a $+10.39\%$ gain over the continuous fixed recurrence interpolation ($0.4238$).
    - *Causal Alignment vs Shuffled Control*: Comparing Arm A (Adaptive) against Arm C (identical tick multiset shuffled across positions) yields a paired difference $\Delta = +0.0768 \pm 0.0422$ with a 95% bootstrap CI of `[+0.0001, +0.1605]` (4 wins, 10 ties, 1 loss, one-tailed $p \approx 0.045$). The lower bound touching zero and high tie rate indicate suggestive, but not decisive, causal superiority.
    - *Orthogonality of State Velocity and Predictive Entropy*: Pearson correlation $r(H(P_t), \tau_t) = -0.0827$ ($N=485$, 95% CI crosses zero). Recurrent halting is governed by hidden state kinetic velocity, not output categorical uncertainty.
    - *Fixed Recurrence Sweet Spot and Over-Smoothing Collapse*: Fixed recurrence peaks at $\tau = 8$ ($0.6315 \pm 0.0221$) before collapsing at $\tau = 16$ ($0.4264 \pm 0.0584$) due to contractive over-smoothing onto a low-rank manifold.
    - *Source*: [`reports/ascii_adaptive_halting_campaign.md`](file:///data/data/com.termux/files/home/projects/titan_text/reports/ascii_adaptive_halting_campaign.md), [`reports/raw/adaptive_halting/campaign_analysis.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/raw/adaptive_halting/campaign_analysis.json).

---

## 2. SUPPORTED BUT NOT ESTABLISHED (Plausible Interpretations With Competing Alternatives)

1. **Spectral Damping Capacity Tradeoff vs Asymptotic Fixed Point**:
   - Arm B (`bounded_h_only`) bounds loss growth at extreme horizons ($L=128$), but accuracy drops slightly ($30.9\%$ vs $36.7\%$ in unconstrained Arm A) and loss still creeps slowly ($2.20 \to 3.59$).
   - *Interpretation A*: OCPD is a spectral-damping capacity tradeoff rather than a true zero-drift fixed point.
   - *Alternative B*: A contractive fixed-point projector with strictly negative real-part eigenvalues could achieve zero loss drift without expressivity sacrifice.

2. **Carrier-Modulation Decoupling in Direct-Sum State $x = [H; C_c]$**:
   - Continuous state $H$ acts as local syntactic carrier wave and clock trigger; discrete channels $C_c$ act as non-local LIFO payload.
   - *Alternative*: Non-linear dense layers couple $H$ and $C_c$ during early steps ($t^* \le 8$), meaning pushdown dynamics are entangled rather than purely orthogonal.

---

## 3. MECHANISTIC HYPOTHESES (Awaiting Decisive Empirical Separation)

1. **Hypothesis H-UNITARY (Continuous Dissipation Bypass via Unitary/Orthogonal Gating)**:
   - A continuous NCA parameterized with orthogonal or skew-symmetric recurrence weights can achieve dissipation-free ballistic transport without discrete STE sign quantization.
   - *Decisive Test*: Train an orthogonal continuous NCA on Dyck-4 ($L=64$) and measure correlation length $\xi$ and stratified accuracy at $D=4, 8$.

2. **Hypothesis H-BOUNDED-DPDA (Simulation Lemma for Finite Physical Grids)**:
   - CD-DV-NCA implements a Bounded Deterministic Pushdown Automaton with effective stack capacity $C_{\text{stack}} = \min(L, C_c \cdot B)$.
   - *Decisive Test*: Measure pushdown retrieval fidelity as a function of stack depth $D$ on adversarial balanced n-gram sequences where local context provides 0 information.

3. **Hypothesis H-FIXED-POINT-OCPD (Strictly Contractive Field Tail Equilibrium)**:
   - Introducing an adaptive step damping factor $\gamma_t = \gamma_0 / (1 + \alpha t)$ on the continuous update $\Delta H$ will extinguish loss creep over $T \ge 64$ while preserving ballistic pushdown carry in $C_c$.

---

## 4. FALSIFIED / SUPERSEDED (Historical Artifacts & Disproven Hypotheses)
> [!WARNING]
> AGENTS MUST NEVER REASON FROM SUPERSEDED FINDINGS. The following historical claims have been invalidated by subsequent forensic tests and experiments.

1. **[ARTIFACT DEBUNKED] "Canonical Sequence Models Collapse to 0.0% on Dyck-4"**:
   - *Debunking*: Forensic audit in `src/main.rs:1725` proved `cmd_benchmark` evaluated randomly initialized checkpoints because checkpoint weights were not loaded into the model wrapper.
   - *Corrected Finding*: When properly initialized with Pre-LN and gate retention, Simple RNN achieves $68.4\%$, Transformer $60.6\%$, GRU $52.8\%$ at $L=16$. Baselines degrade under length extrapolation ($L=64: 46-52\%$). Untied feedforward collapses to chance ($24.2\%$).

2. **[ARTIFACT RESOLVED] "FC-4 Carry Lesion Shows No Causal Delta on Dyck-4"**:
   - *Debunking*: Initial evaluation computed unweighted aggregate accuracy across all tokens, where $50\%$ of brackets have depth $D=1$ and are resolved by local bigrams.
   - *Corrected Finding*: Depth-stratified benchmark revealed a massive $+12.5\%$ causal carry gain at $D=4$ ($50.5\%$ intact vs $38.0\%$ lesion, and $26.0\%$ scramble).

3. **[FALSIFIED] "H1: Settling Latency Explains $L=16$ Parity Interior Failure"**:
   - *Falsification*: Causal horizon sweep with $\tau=48$ across 5 seeds yielded flat $46.9\%$ accuracy ($G_{\text{identity}} = -2.5\%$). Settling time was not the bottleneck; spatial diffusion length $\xi \le 6$ was.

4. **[FALSIFIED] "H2: 1D Continuous NCA Cannot Learn Sequential Parity"**:
   - *Falsification*: Disproved on $L=8, \tau=8$ where $G_{\text{identity}} = +18.8\% \pm 4.4\%$ ($d=1.90$). Continuous NCAs can compute instance-specific parity on compact chains.

5. **[SUPERSEDED] "Theorem 12: CD-DV-NCA is Formally Equivalent to a Chomsky Type-2 DPDA"**:
   - *Supercession*: Formal methods audit proved that any physical machine on a finite 1D grid with finite channels is strictly a Chomsky Type-3 Finite State Automaton. Narrowed to Bounded DPDA Emulation.

6. **[SUPERSEDED] "The $v_0$ Text Checkpoint Proves Instance-Specific Latent Computation"**:
   - *Supercession*: Shuffled batch controls revealed $p=0.388, d=0.43$, failing statistical significance.

7. **[FALSIFIED] "Dyck-4 Pushdown Accuracy is Explained by Shallow Bigrams or Local Convolution"**:
   - *Falsification*: Evaluated on the Adversarial Balanced-N-Gram Benchmark across transport gaps $G \in \{4, 8\}$ where all queries are at physical distances $d \in [6..24]$ cells (exceeding continuous correlation length $\xi \le 6$). Continuous-only models collapse to flat uniform chance ($24.7\% \approx 25.0\%$), while intact CD-DV-NCA maintains $36.2\% - 37.0\%$ with $+11.5\%$ to $+12.3\%$ carry lesion delta and peak retrieval of $50.0\%$ at $d=20$ cells. Untied feedforward collapses to $25.8\%$ chance.

8. **[FALSIFIED] "Predictive Entropy Governs Recurrent Compute Halting"**:
   - *Falsification*: Measured Pearson correlation between next-token predictive entropy $H(P_t)$ and allocated compute $\tau_t$ is null ($r = -0.0827$, $N=485$, 95% CI includes zero). Output categorical uncertainty and latent continuous state velocity $\|x_t - x_{t-1}\|$ are decoupled in tiny recurrent NCAs; halting is driven by internal velocity, not next-token entropy.

---

## 5. OPEN QUESTIONS (Active Theoretical & Empirical Frontiers)

1. **Infinite-Horizon Stability**: Can we construct a continuous update operator that attains an exact contractive fixed point ($\|H_{t+1} - H_t\| \to 0$ and loss $\le 2.30$) over $T \in [64..256]$ without damping expressive capacity?
2. **Adversarial Stack Depth Scaling**: What is the exact stack capacity breakdown curve for $D \in [8, 16]$ under $G=8$ non-local gaps?
3. **Continuous vs Discrete Ballistic Transport**: Is discrete quantization mathematically necessary for dissipation-free non-local transport on a lattice, or can continuous unitary operators achieve identical reach?
4. **Topological Shear Mechanism**: Is the sub-chance collapse ($18.1\% < 25.0\%$) under batch carry shuffling caused by spatial phase mismatch or out-of-distribution logit saturation?

---

## 6. KNOWN CONFOUNDS & METHODOLOGICAL WEAKNESSES

1. **[RESOLVED] Shallow Bracket Density**: Standard stratified Dyck-4 sampling generates $50\%$ of brackets at depth $D \le 2$ within continuous receptive field $\xi \le 6$.
   - *Resolution*: Resolved by `scripts/run_adversarial_balanced_dyck.py` with explicit transport gap $G \in \{4, 8\}$ and per-slot distance tracking $d(s) = G + 2 + 2s$.
2. **Unconstrained Continuous Drift**: Models trained at fixed unroll $T=20$ develop Lyapunov drift when tested at $T \ge 32$ without multi-step curriculum or subspace norm, confounding kinematic reach tests with loss explosion.
3. **Static Input Memorization**: Adding persistent input channels (`--persistent-input`) allows the model to memorize static inputs rather than accumulating intermediate state, dropping val accuracy from $70.3\%$ to $49.2\%$.
4. **Readout Logit Temperature Distortion**: Extreme OOD activations inflate logit norms, which can saturate softmax and disguise random predictions as systematic biases. Temperature calibration is required.

---

## 7. NEXT DISCRIMINATING EXPERIMENTS

1. **[COMPLETED] Adversarial Balanced-N-Gram Pushdown Test ($D \in [2, 8], G \in [0, 8]$)**:
   - Proved non-local stack retrieval sustains $+11.5\%$ to $+25.0\%$ carry gains across distances $d \in [6..24]$ cells while continuous and feedforward baselines collapse to flat chance ($24.7\% \approx 25.0\%$).
2. **Continuous Unitary/Skew-Symmetric Operator Ablation**:
   - Implement skew-symmetric continuous recurrence $\Delta H = (W - W^T) H$ and compare correlation length $\xi$ and deep bracket accuracy directly against discrete STE sign quantization.
3. **Adaptive Step Damping Curriculum ($T \in [24..128]$)**:
   - Evaluate whether step damping $\gamma_t = \gamma_0 / (1 + 0.05 t)$ halts loss explosion at $L=128$ while maintaining pushdown bracket matching at $D=4, 8$.

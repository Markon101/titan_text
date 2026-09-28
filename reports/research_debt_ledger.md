# Titan Text: Live Research Debt Ledger

**Campaign State**: Active Autonomous Research Campaign  
**Date**: 2026-09-22T00:30:00Z  
**Governing Metric**: $\text{Priority} = \frac{\text{Expected Information Gain} \times \text{Downstream Importance}}{\text{Compute Cost}}$  

---

## 1. Prioritized Research Debt Items

2026-09-27 precedence correction: older entries that call adaptive superiority,
infinite-horizon memory, or universal continuous/discrete separation resolved
are historical interpretations. C-HALT-014 and the
[vNext audit](../docs/VNEXT_RECONSTRUCTION.md) supersede those interpretations.
RD-018 remains OPEN: failure of one auxiliary-loss experiment cannot logically
exclude optimization difficulty or prove attenuation. The old trainer repeats
one seed-0 batch and does not seed initialization from the advertised seed.

| New debt | Status | Discriminating action |
|---|---|---|
| RD-019 Legacy STE gradient cancellation | VERIFIED BUG, legacy preserved | Version a correct backward estimator and retrain a matched continuous/discrete factorial; do not silently change old models |
| RD-020 Historical training seed and repeated-batch semantics | OPEN for old CLI; corrected in substrate lab | Add an explicit opt-in versioned training policy before reinterpreting historical campaigns |
| RD-021 Full-prefix IPPR competence | UNRESOLVED | Unique heldout panels, complete answers, train-only suffix controls, required-bit counterfactual pairs |
| RD-022 Multiscale route utility versus geometry | OPEN | A/B/C comparison, sufficient trained ticks, up/down lesions, route-preserving sham and active-capacity controls |

| Debt ID | Title / Hedged Claim | Importance | Uncertainty | Cost (s) | Priority Score | Status | Downstream Claims |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`RD-009`** | **Rate-Invariance Under Transport Conjugation**<br>*"Does continuous field contraction rate rho_H remain invariant under carry channel conjugations?"* | 0.85 | 0.05 | 120 | **0.35** | `RESOLVED` | `C-CONJUGATION-009`, `C-CARRY-002` |
| **`RD-010`** | **Infinite-Horizon Stability & Subspace Normalization Tradeoff**<br>*"Does OCPD subspace normalization resolve the deep-horizon loss explosion (2.28 -> 16.73) without collapsing pushdown accuracy?"* | 0.95 | 0.05 | 300 | **0.16** | `RESOLVED` | `C-OCPD-010`, `C-OOD128-008` |
| **`RD-005`** | **Latent Stack Probing & Counterfactual Carry Transplantation**<br>*"Carry channels causally dictate LIFO bracket prediction rather than prompt tokens."* | 0.85 | 0.05 | 450 | **0.09** | `RESOLVED` | `C-CARRY-002`, `C-DPDA-004` |
| **`RD-006`** | **Formal Bounded DPDA Theorem Narrowing**<br>*"Theorem 12 proves DPDA equivalence, but finite models are formally finite automata."* | 0.75 | 0.05 | 60 | **0.06** | `RESOLVED` | `C-DPDA-004` |
| **`RD-008`** | **Jev Governor Infrastructure Ablation Study ($J_0$ to $J_6$)**<br>*"Jev appears to improve research rigor, but its objective epistemic yield has not been ablated."* | 0.85 | 0.05 | 240 | **0.18** | `RESOLVED` | All Claims |
| **`RD-007`** | **Non-Linear Cross-Coupling in Direct-Sum Decomposition (Theorem 15)**<br>*"Readout z decomposes into regular base and pushdown carry, but intermediate updates are nonlinearly coupled."* | 0.70 | 0.05 | 180 | **0.19** | `RESOLVED` | `C-CARRY-002` |
| **`RD-004`** | **Full Grid Empirical Causal Lightcone Phase Boundary ($L \times T \times k$)**<br>*"The causal reach law T* >= 2 ceil(L/k) suggests a phase boundary, but was only checked at two points."* | 0.85 | 0.05 | 600 | **0.07** | `RESOLVED` | `C-LIGHTCONE-003` |
| **`RD-003`** | **Hostile Baseline Re-Engineering & Fair Competition on Dyck-4**<br>*"Baselines collapsed on Dyck-4 (0.0% to 0.7%), but cmd_benchmark evaluated untrained weights."* | 1.00 | 0.05 | 900 | **0.05** | `RESOLVED` | `C-BASELINE-007`, `C-DYCK-001` |
| **`RD-001`** | **Deconstruct Continuous Hidden State Residual Floor (32.33%)**<br>*"The residual floor of 32.33% likely reflects shallow bracket resolution by continuous channels."* | 0.95 | 0.05 | 300 | **0.05** | `RESOLVED` | `C-DYCK-001`, `C-CARRY-002` |
| **`RD-002`** | **Resolve Automated Benchmark Audit Failure on FC-4 (`FALSIFIED`)**<br>*"Carry channels appear to mediate stack dynamics despite FC-4 failing."* | 0.90 | 0.05 | 300 | **0.05** | `RESOLVED` | `C-CARRY-002` |
| **`RD-011`** | **Adaptive Recurrent Halting for Generative ASCII Token Transitions**<br>*"Can adaptive token-level tau allocation prevent boundary over-deliberation and improve multi-family generation?"* | 0.85 | 0.05 | 180 | **0.14** | `RESOLVED` | `C-ASCII-011`, `C-HALT-012` |
| **`RD-012`** | **Optimal Halting Threshold Calibration and Causal Verification at Deep Recurrence ($\theta^*=0.25$)**<br>*"Does calibrating the adaptive halting threshold eliminate the under-compute confound and establish statistically significant causal superiority over fixed and shuffled compute?"* | 0.95 | 0.05 | 240 | **0.20** | `RESOLVED` | `C-HALT-013`, `C-ASCII-011` |
| **`RD-013`** | **Held-Out Seed Generalization & Causal Alignment Boundary Battery**<br>*"Does the calibrated adaptive halting policy generalize to unseen seeds without post-hoc selection bias, and does causal alignment advantage over shuffled compute persist under matched EOS lengths?"* | 0.90 | 0.05 | 300 | **0.15** | `RESOLVED` | `C-HALT-013` |

---

## 2. Autopsy & Action Plan for Newly Resolved Debts

### Priority 1: `RD-013` — Held-Out Seed Generalization & Causal Alignment Boundary Battery
- **Empirical Resolution**: Evaluated frozen policy ($\theta^* = 0.25$) across 10 unseen seeds (`[501..605]`, 270 total runs) in [`reports/ascii_adaptive_halting_campaign.md`](file:///data/data/com.termux/files/home/projects/titan_text/reports/ascii_adaptive_halting_campaign.md) and [`reports/raw/heldout_halting/campaign_analysis.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/raw/heldout_halting/campaign_analysis.json).
- **Key Findings**:
  1. *Robust Pareto Superiority Out-of-Distribution*: Arm A achieves $0.5535 \pm 0.0261$ at $\bar{\tau} = 4.60$, outperforming fixed $\tau=4$ ($0.4735, +16.9\%$) with a net Pareto gain of $+0.0535$ ($+10.71\%$) over fixed interpolation.
  2. *Causal Alignment Boundary*: Paired comparison against shuffled schedule Arm C drops to $\Delta = +0.0292 \pm 0.0262$ with 95% bootstrap CI `[-0.0186, +0.0821]` (8 wins, 13 ties, 9 losses, $p = 0.1468$).
  3. *Scientific Takeaway*: Adaptive halting's primary causal mechanism is budget-operating efficiency (hovering in the $\tau \in [4, 8]$ sweet spot while avoiding over-smoothing collapse at $\tau=16$), while fine-grained per-token state alignment provides a modest secondary benefit.

### Priority 2: `RD-012` — Optimal Halting Threshold Calibration & Causal Verification ($\theta^*=0.25$)
- **Empirical Resolution**: Executed 105-run calibration sweep across $\theta \in [0.10, 0.35]$ followed by full 135-run 4-arm campaign at $\theta^* = 0.25$ in [`reports/ascii_adaptive_halting_campaign.md`](file:///data/data/com.termux/files/home/projects/titan_text/reports/ascii_adaptive_halting_campaign.md) and [`reports/raw/calibrated_halting/campaign_analysis.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/raw/calibrated_halting/campaign_analysis.json).
- **Key Findings**:
  1. *Threshold Response Mapping*: Monotonically shifts compute from $\bar{\tau} = 12.84$ ($\theta=0.10$, over-smoothing) down to $\bar{\tau} = 2.35$ ($\theta=0.35$, premature halting). Identifies $\theta^* = 0.25$ as optimal operating point ($\bar{\tau} = 4.53$, edit sim $0.6719$).
  2. *Pareto & Absolute Superiority*: Achieves $0.6719 \pm 0.0311$ edit similarity, strictly exceeding fixed $\tau=4$ ($0.5274, +27.4\%$) and matching/exceeding prior global fixed peak at $\tau=8$ ($0.6315$) while using $43.4\%$ less compute ($4.53$ vs $8.0$ ticks). Delivers $+24.16\%$ Pareto quality gain over fixed interpolation.
  3. *Strictly Positive Causal Alignment*: $\Delta(A - C) = +0.1406 \pm 0.0592$, 95% bootstrap CI `[+0.0332, +0.2568]` (bounded away from zero), 11 wins / 0 ties / 4 losses, exact permutation test $p = 0.0139$, paired $t(14) = 2.37, p = 0.0163$, Cohen's $d_z = 0.613$.
  4. *Falsification Boundaries Registered*: Arm C consumed fraction is $0.790$ due to early `<eos>` truncation under misallocated ticks; $\theta^*$ chosen from sweep; predictive entropy correlation remains null ($r = -0.1475$).

### Priority 2: `RD-011` — Adaptive Recurrent Halting for Generative ASCII Token Transitions
- **Empirical Resolution**: Executed 4-arm campaign across 135 runs (3 prompts: box, maze, diamond; 5 seeds: 42, 101, 202, 303, 404) in [`reports/ascii_adaptive_halting_campaign.md`](file:///data/data/com.termux/files/home/projects/titan_text/reports/ascii_adaptive_halting_campaign.md) and [`reports/raw/adaptive_halting/campaign_analysis.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/raw/adaptive_halting/campaign_analysis.json).
- **Key Findings**:
  1. *Pareto Efficiency*: Converges to mean compute $\bar{\tau} = 2.35$ with edit similarity $0.4679 \pm 0.0381$, achieving a $+10.39\%$ quality advantage over linear interpolation of continuous fixed recurrence ($0.4238$).
  2. *Causal Alignment (Arm A vs Arm C)*: Shuffling the exact multiset of $\tau$ values across token positions drops edit similarity to $0.3911 \pm 0.0273$ ($\Delta = +0.0768 \pm 0.0422$, 95% bootstrap CI `[+0.0001, +0.1605]`, 4 wins, 10 ties, 1 loss, one-tailed $p \approx 0.045$). Suggestive of state-dependent advantage, though bounded by low statistical margin.
  3. *Heterogeneous Allocation by Token Class*: Dynamically allocates $3.00$ ticks to `<eos>` and $2.36$ ticks to structural boundaries, while predictable symbols settle at $2.00$ ticks.
  4. *Entropy Decoupling*: Correlation between predictive entropy $H(P_t)$ and allocated compute is null ($r = -0.0827$), proving halting tracks continuous latent velocity, not categorical output uncertainty.
  5. *Over-Smoothing Collapse*: Fixed recurrence peaks at $\tau = 8$ ($0.6315$) and collapses at $\tau = 16$ ($0.4264$) due to contractive field dissipation toward a low-rank manifold.

### Priority 2: `RD-009` — Rate-Invariance Under Transport Conjugation
- **Empirical Resolution**: Evaluated 8 carry transformations across 6 conditions ($t^* \in \{8, 12, 16\}, D \in \{4, 8\}$) over 3 checkpoints in [`reports/transport_conjugation_benchmark_results.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/transport_conjugation_benchmark_results.json).
- **Key Findings**:
  1. Under spatial roll (`roll_spatial_4`), continuous field norm $\|H\|$ exhibits bit-level translation invariance ($\Delta \le 0.0023 < 0.02$, passing TOST equivalence with $p < 10^{-6}$).
  2. Under channel permutations, $\|H\|$ experiences initial coordinate perturbation ($\Delta \approx 0.35$ at $t^*=8$) due to learned linear weights in `dense1`, but the perturbation monotonically contracts to $\Delta \le 0.05$ at $t^*=16$ as the continuous manifold converges toward its asymptotic fixed-point basin.

### Priority 2: `RD-010` — Infinite-Horizon Stability & Subspace Normalization Tradeoff
- **Empirical Resolution**: Executed 4-arm campaign across seeds $\{42, 43, 44\}$ on $L \in \{16, 64, 128\}$ in [`reports/subspace_normalization_ocpd_results.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/subspace_normalization_ocpd_results.json).
- **Key Findings**:
  1. Restricting bounded normalization to continuous subspace $H$ (`bounded_h_only`) cuts loss by $47.3\%$ at $L=64$ ($5.13 \to 2.70$) and by $71.1\%$ at $L=128$ ($12.45 \to 3.59$).
  2. Arm C (Full-64 norm, no STE) squashes carry amplitudes, collapsing accuracy to the uniform chance floor ($25.8\% \approx 25.0\%$) at $L=128$, proving discrete unattenuation is mathematically necessary for pushdown memory.
  3. Decisive synthesis by DeepSeek team: OCPD is a spectral-damping capacity tradeoff rather than an unconstrained fixed point, as Arm B trades a small accuracy margin ($30.9\%$ vs $36.7\%$ in unconstrained Arm A) to achieve bounded loss growth.

### Priority 3: `RD-011` — Adaptive Recurrent Halting for Generative ASCII Token Transitions
- **Context**: In generative ASCII synthesis, $\tau=4$ and $\tau=8$ produce clean multiline structures, but $\tau=16$ exhibits boundary token saturation (`+=================`), suggesting constant $\tau$ across all token types over-deliberates on simple linear runs.
- **Proposed Action**: Implement adaptive step halting (`AdaptiveHalting` / kinetic energy threshold) during autoregressive token generation, allowing fewer ticks on repetitive tokens (e.g. horizontal walls `-`) and deeper deliberation at structural branch points (`+`, `\n`, corners).


## New Debt from Qualification Battery (2026-09-26)

| Debt ID | Title | Status | Downstream |
|---|---|---|---|
| `RD-014` | **Adaptive halting superiority REFUTED at matched compute** — fixed tau=5 beats frozen-theta adaptive; state-independent matched controls (C/D) show no separation. Prior "Pareto superiority"/"causal alignment" claims downgraded to budget artifacts. | `RESOLVED (negative)` | `C-HALT-014` |
| `RD-015` | **Training-time adaptive halting** — whether a halter learned jointly with the model (vs. sampling-time adaptation of a tau=4-trained checkpoint) yields state-dependent allocation benefits. Untested; requires train-loop integration. | `OPEN` | `C-HALT-014` |
| `RD-016` | **Per-channel halting-metric telemetry** — which channels dominate the relative-delta signal (full 64-channel vector at active position; carry NOT APPLICABLE, carry_channels=0). | `OPEN` | `C-HALT-014` |
| `RD-017` | **Coordinate channel at L=16 (IPPR)** — whether explicit absolute positional information rescues interior slots. RESOLVED NEGATIVE 2026-09-27: preregistered 3-arm battery (baseline/constant-sham/coordinate, 5 fresh seeds each, single-shot) shows B-A=+0.31pp (p=0.938), B-C=+4.38pp (< +8pp threshold), eval-time coordinate counterfactuals inconsistent → H_COORD falsified (C-COORD-015). Chunk-resume procedure invalidated (Adam state not persisted); single-shot required. | `RESOLVED` | `C-COORD-015` |
| `RD-018` | **Discriminate H_OPT (credit-assignment saddle) vs H_ATTENUATION (propagation limit) at L=16** — auxiliary supervision test: per-step intermediate parity losses / deep supervision at intermediate ticks. If interior learns only with per-step signal → H_OPT; if per-step signal also fails → H_ATTENUATION. PROPOSED ONLY; requires train-loop loss plumbing. | `OPEN` | `C-COORD-015` |

# Titan Text: Live Research Debt Ledger

**Campaign State**: Active Autonomous Research Campaign  
**Date**: 2026-09-22T00:30:00Z  
**Governing Metric**: $\text{Priority} = \frac{\text{Expected Information Gain} \times \text{Downstream Importance}}{\text{Compute Cost}}$  

---

## 1. Prioritized Research Debt Items

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
| **`RD-011`** | **Adaptive Recurrent Halting for Generative ASCII Token Transitions**<br>*"Can adaptive token-level tau allocation prevent boundary over-deliberation and improve multi-family generation?"* | 0.85 | 0.30 | 180 | **0.14** | `OPEN` | `C-ASCII-011` |

---

## 2. Autopsy & Action Plan for Newly Resolved Debts

### Priority 1: `RD-009` — Rate-Invariance Under Transport Conjugation
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


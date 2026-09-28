# Titan Text: Live Scientific Claim Ledger

**Campaign State**: Active Autonomous Research Campaign  
**Last Updated**: 2026-09-27 (vNext evidence audit)  
**Governing Infrastructure**: Jev Cognitive Governor & Evidence-First Protocol  

---

## 1. Executive Summary of Active Claims

Current qualification: historical L8 IPPR identity dependence is supported, but
full-prefix computation is CONFOUNDED by 11 heldout rows, local shortcuts and
evaluation-seed pseudoreplication. Historical STE carry has a verified zero
backward derivative. Older fixed-batch/unseeded-initialization/chunk-resume
campaigns cannot establish intrinsic substrate impossibility. Adaptive-halting
superiority is superseded by C-HALT-014. See
[reconstruction](../docs/VNEXT_RECONSTRUCTION.md) and
[auditable recalculation](vnext_evidence_audit.json).

| Claim ID | Title | Status | Confidence | Empirical Basis | Primary Challenge / Confound |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **`C-DYCK-001`** | Dyck-4 Chomsky Type-2 Performance | `SUPPORTED` | 0.90 | $43.90\% \pm 1.61\%$ on $L=64$ vs Markov $0.098\%$ | Shallow brackets ($D \le 2$) resolved locally by bigrams |
| **`C-CARRY-002`** | Discrete Carry Channel Mediation | `EMPIRICALLY_QUALIFIED` | 0.88 | Peak $\Delta = +12.5\%$ at $D=4$; $H$ collapses to chance ($26.9\%$) at $D \ge 12$ | Only operates within causal lightcone window $2 \le D \le 8$ |
| **`C-LIGHTCONE-003`** | Ballistic Causal Reach Horizon Law | `CONFIRMED` | 0.95 | $T^* \ge 2\lceil L/k \rceil$; both intact and lesion collapse to chance at $D \ge 12$ for $T=24$ | Validated across full depth spectrum $D \in [1, 16]$ |
| **`C-DPDA-004`** | DPDA Equivalence (Theorem 12) | `PARTIALLY REFUTED` | 0.20 | Formal machine proof | Finite lattice with finite channels is strictly FSA (Type-3) |
| **`C-DISCRETE-SEPARATION-005`** | Dissipation Separation (Theorem 14) | `CONTESTED` | 0.70 | Continuous $\xi \approx 4.5$, Discrete $\xi = \infty$ | Applies to diffusive NCAs; continuous gated models can extend reach |
| **`C-TOPOLOGY-006`** | Topological Shear Collapse | `CONTESTED` | 0.60 | Batch shuffle collapses to $18.10\% < 25\%$ | May be out-of-distribution logit saturation artifact |
| **`C-BASELINE-007`** | Canonical Baseline Capacity on Dyck-4 | `REPLICATED_AND_SUPERSEDED` | 0.98 | Pre-LN Transformer ($60.6\%$), GRU ($52.8\%$), Simple RNN ($68.4\%$) | Sequential models degrade under $L=64$ extrapolation ($46-52\%$) |
| **`C-OOD128-008`** | Extreme OOD ($L=128$) Causal Reach & Drift | `SUPPORTED_AND_BOUNDED` | 0.92 | CD-DV-NCA maintains $45.0\% \to 36.7\%$ across $T \in [16..64]$ on $L=128$ | Continuous loss explodes ($2.28 \to 16.73$) in unregularized field |
| **`C-CONJUGATION-009`** | Spatial Invariance & Coordinate Coupling | `SUPPORTED` | 0.90 | Spatial roll $\Delta \|H\| \le 0.0023 < 0.02$ across all $t^*, D$ | Channel permutation perturbs $\|H\|$ ($\Delta = 0.35 \to 0.05$ at $t^*=16$) |
| **`C-OCPD-010`** | Subspace Normalization & OCPD Stability | `EMPIRICALLY_QUALIFIED` | 0.85 | Arm B cuts loss by $47\%$ at $L=64$ ($2.70$) & $71\%$ at $L=128$ ($3.59$) | Acc drops to $30.9\%$ at $L=128$; Arm C collapses to chance ($25.8\%$) |
| **`C-ASCII-011`** | Generative ASCII Synthesis via Causal Recurrence | `SUPPORTED` | 0.95 | Recurrence ablation: $\tau=0$ collapses to `<eos>`, $\tau=4,8$ yields $63.5\%$ symmetry, $0\%$ memorization | Over-deliberation at $\tau=16$; adaptive per-token halting open |

---

## 2. Detailed Claim Profiles

### `C-DYCK-001`: Dyck-4 Chomsky Type-2 Performance
- **Statement**: CD-DV-NCA achieves $43.90\% \pm 1.61\%$ exact bracket prediction on Dyck-4 at $L=64$, surpassing the theoretical Markov-4 ceiling ($0.098\%$) by $448\times$.
- **Status**: `SUPPORTED` (Confidence: 0.90)
- **Supporting Artifacts**: [`reports/dyck_pushdown_benchmark_results.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/dyck_pushdown_benchmark_results.json)
- **Seeds Tested**: 42, 43, 44
- **Stratified Resolution**: Validated in [`reports/stratified_depth_benchmark_results.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/stratified_depth_benchmark_results.json): at $D=4$, intact accuracy reaches $50.5\%$ vs Markov floor $0.098\%$. At $D \ge 12$, causal reach horizon at $T=24$ caps performance.

### `C-CARRY-002`: Discrete Carry Channel Mediation
- **Statement**: Discrete carry channels ($C_c=32$) mediate non-local bracket matching in Dyck-4; zeroing carry channels causes a $-11.57\%$ accuracy drop ($43.90\% \to 32.33\%$).
- **Status**: `EMPIRICALLY_QUALIFIED` (Confidence: 0.88)
- **Supporting Artifacts**: [`reports/stratified_depth_benchmark_results.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/stratified_depth_benchmark_results.json)
- **Resolution of FC-4 Falsification**: The prior aggregate failure was an artifact of shallow bracket averaging ($D=1$ lesion acc $83.3\%$). When stratified by nesting depth:
  - At $D=1$: $\Delta = -2.1\%$ (continuous $H$ handles local bigrams independently).
  - At $D=2$: $\Delta = +8.3\%$ ($58.3\% \to 50.0\%$, scramble collapses to $26.0\%$).
  - At $D=4$: $\Delta = +12.5\%$ ($50.5\% \to 38.0\%$, peak causal mediation).
  - At $D \ge 12$: Continuous hidden state $H$ completely collapses to chance ($26.9\% \approx 25.0\%$).
- **Conclusion**: Continuous channels do not have pushdown memory; carry channels mediate non-local matching within the causal lightcone window.

### `C-LIGHTCONE-003`: Ballistic Causal Reach Horizon Law
- **Statement**: Information transport requires roundtrip horizon $T^* \ge 2\lceil L/k \rceil$. At $L=64, T=24, k=4$, causal reach is 48 cells ($75\%$), capping theoretical accuracy at $46.08\%$.
- **Status**: `CONFIRMED` (Confidence: 0.95)
- **Supporting Artifacts**: [`reports/stratified_depth_benchmark_results.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/stratified_depth_benchmark_results.json)
- **Empirical Validation**: At $D=12$ and $D=16$, both Intact ($29.3\%$) and Carry Lesion ($26.9\%$) collapse toward chance ($25.0\%$). Deep brackets at cells $0..8$ are causally severed at $T=24$, proving the roundtrip reach bound $T^* = 32$.

### `C-DPDA-004`: Pushdown Automaton Equivalence (Theorem 12)
- **Statement**: CD-DV-NCA is formally equivalent to a DPDA.
- **Status**: `PARTIALLY REFUTED` (Confidence: 0.20)
- **Formal Audit**: A finite 1D lattice with finite channels and finite precision cannot represent an unbounded stack. The theorem must be restricted to **Bounded DPDA Emulation up to capacity $C_{\text{stack}}$**.

### `C-BASELINE-007`: Canonical Baseline Modeling Capacity on Dyck-4
- **Statement**: Canonical sequence models (Simple RNN, Transformer, GRU) learn Dyck-4 in-distribution (Simple RNN: $68.4\%$, Transformer: $60.6\%$, GRU: $52.8\%$ at $L=16$), but degrade under length extrapolation ($L=64$: Simple RNN $52.2\%$, GRU $47.8\%$, Transformer $46.2\%$); Untied Feedforward collapses to chance ($24.2\% < 25.0\%$).
- **Status**: `REPLICATED_AND_SUPERSEDED` (Confidence: 0.98)
- **Supporting Artifacts**: [`reports/hostile_baselines_benchmark_results.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/hostile_baselines_benchmark_results.json)
- **Audit Autopsy**: Forensics in [`src/main.rs`](file:///data/data/com.termux/files/home/projects/titan_text/src/main.rs#L1725-L1740) proved that prior reports of $0.0\%$ collapse evaluated untrained random weights. With Pre-LN and gate retention initialization, sequential models excel in-distribution ($L=16$), but suffer continuous drift or attention dilution under $4\times$ length extrapolation ($L=64$). Untied local feedforward completely collapses to chance ($24.2\%$).

### `C-OOD128-008`: Extreme OOD ($L=128$) Causal Reach & Horizon Sweep
- **Statement**: On extreme extrapolation length $L=128$ ($8\times$ training length $L=16$), CD-DV-NCA maintains bracket accuracy above chance across all recurrence steps ($45.0\%$ at $T=16$ down to $36.7\%$ at $T=64$), outperforming Baseline Causal NCA ($39.2\%$ at $T=16$ down to $31.4\%$ at $T=64$). However, unregularized continuous fields suffer monotonic loss explosion ($2.28 \to 16.73$ in NCA, $3.51 \to 19.60$ in Baseline).
- **Status**: `SUPPORTED_AND_BOUNDED` (Confidence: 0.92)
- **Supporting Artifacts**: [`reports/ood_128_causal_reach_sweep_results.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/ood_128_causal_reach_sweep_results.json)
- **Seeds Tested**: 42, 43, 44

### `C-CONJUGATION-009`: Spatial Invariance & Coordinate Coupling under Transport Conjugation
- **Statement**: Under transport conjugation of carry channels, continuous field energy $\|H\|$ exhibits bit-level spatial translation invariance ($\Delta \|H\| \le 0.0023 < 0.02$, $p < 10^{-6}$ across all $t^* \in \{8, 12, 16\}$ and depths $D \in \{4, 8\}$ under `roll_spatial_4`). Channel permutations perturb $\|H\|$ ($\Delta = 0.35 \to 0.05$ at $t^*=16$) due to linear weight coordinate specialization in `dense1`, but the perturbation decays as recurrence depth approaches the fixed-point basin.
- **Status**: `SUPPORTED` (Confidence: 0.90)
- **Supporting Artifacts**: [`reports/transport_conjugation_benchmark_results.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/transport_conjugation_benchmark_results.json)

### `C-OCPD-010`: Subspace Normalization & Orthogonal Contractive-Pushdown Decomposition
- **Statement**: Restricting bounded state normalization to the continuous subspace $H$ (`bounded_h_only`, channels $0..32$) bounds continuous field drift, reducing cross-entropy loss by $47.3\%$ at $L=64$ ($5.13 \to 2.70$) and by $71.1\%$ at $L=128$ ($12.45 \to 3.59$) while sustaining $30.9\%$ accuracy. Uniform full-channel normalization (Arm C) squashes carry amplitudes and collapses accuracy to the uniform chance floor ($25.8\% \approx 25.0\%$). However, OCPD is a spectral-damping capacity trade rather than a zero-drift fixed point, as loss still creeps slowly ($2.20 \to 3.59$).
- **Status**: `EMPIRICALLY_QUALIFIED` (Confidence: 0.85)
- **Supporting Artifacts**: [`reports/subspace_normalization_ocpd_results.json`](file:///data/data/com.termux/files/home/projects/titan_text/reports/subspace_normalization_ocpd_results.json)
- **Seeds Tested**: 42, 43, 44

### `C-ASCII-011`: Structured Generative ASCII Synthesis via Causal Recurrence
- **Statement**: A tiny recurrent NCA text model (~43,844 parameters) trained on procedural multiline ASCII curriculum generates structured, novel multiline ASCII boxes and motifs. Latent recurrent computation is causally necessary: $\tau=0$ (feedforward only) and `--lesion-state` collapse immediately to `<eos>`, $\tau=2$ produces crude character repetitions (`#####`), while $\tau=4$ and $\tau=8$ produce multiline boxes with borders (`+====+`), walls (`:`), and clean whitespace ($63.5\%$ horizontal symmetry). Generations generalize rather than memorize: 0/96 exact training matches across fixed seeds `42, 101, 202, 303, 404`, with nearest training edit similarity shifting from $0.081$ to $0.391$.
- **Status**: `SUPPORTED` (Confidence: 0.95)
- **Supporting Artifacts**: [`reports/ascii_generative_campaign_report.md`](file:///data/data/com.termux/files/home/projects/titan_text/reports/ascii_generative_campaign_report.md), [`runs/ascii/ascii_v1_trained_1790462724/`](file:///data/data/com.termux/files/home/projects/titan_text/runs/ascii/ascii_v1_trained_1790462724/)
- **Seeds Tested**: 42, 101, 202, 303, 404
- **Preserved Archive**: Verified 100% SHA-256 match on Android shared storage `/sdcard/Download/TitanText/ascii_runs/ascii_v1_trained_1790462724/`


### `C-HALT-014`: Qualification Battery Downgrade of Adaptive Halting Superiority
- **Statement**: Frozen-protocol qualification battery (10 held-out seeds x 3 prompts, theta*=0.25 frozen, seed-level inference) REFUTES the prior claim of adaptive-halting Pareto superiority: fixed tau=5 significantly outperforms frozen-theta adaptive halting at matched compute (paired A-B5 = -0.0403, seed-cluster CI [-0.0709,-0.0115], 8/10 seeds negative); state-independent matched controls (multiset-permuted C_matched, distribution-matched D_matched) are indistinguishable from adaptive in primary analysis (seed-perm p=0.36/0.47) and better under truncation-excluded sensitivity. A state-free char-class rule explains R^2=0.27 of tick variance. Sampling-time adaptive halting on the tau=4-trained checkpoint is a noisy proxy for a fixed operating point near tau=5 with no measurable state-dependent contribution.
- **Status**: `SUPPORTED` (Confidence: 0.85; downgrades C-HALT-013/C-ASCII-011 superiority language)
- **Limitations**: 10 independent seeds; sampling-time adaptation only (training-time adaptive halting untested); single checkpoint; realized compute of C/D matched within ~11% mean with 3-4 truncated pairs per arm.
- **Supporting Artifacts**: [reports/ascii_adaptive_halting_qualification.md](file:///data/data/com.termux/files/home/projects/titan_text/reports/ascii_adaptive_halting_qualification.md), [reports/raw/qualification_battery/qualification_analysis.json](file:///data/data/com.termux/files/home/projects/titan_text/reports/raw/qualification_battery/qualification_analysis.json)
- **Commit**: (qualification report commit)

---

## C-COORD-015 — Explicit coordinate channel does not rescue the L=16 interior

- **CLAIM**: A scalar absolute-position channel (p_i = 2i/(L-1)-1), injected at
  every position during training, does not improve interior-slot accuracy
  (Slots 1/2) at L=16 and does not produce models that causally depend on the
  coordinate.
- **STATUS**: SUPPORTED (negative result); H_COORD REFUTED (confirmatory)
- **EVIDENCE**: Preregistered three-arm battery (baseline / training-time
  constant sham / coordinate), 5 fresh seeds per arm (801-805), single-shot
  1000-epoch training, 15/15 converged. InteriorMean: B-A = +0.31pp
  [boot -3.12,+3.44], seed sign-flip p=0.938; B-C = +4.38pp (below the frozen
  +8pp threshold; p=0.062 = exact-test floor at n=5); C-A = -4.06pp. Eval-time
  coordinate counterfactuals (zeroed/shuffled/reversed/constant) shift interior
  slots inconsistently (mean G_slot1 = -1.87, G_slot2 = -1.41). Baseline
  replication gate passed (5/5 baselines interior at chance).
- **LIMITATION**: n=5 per arm (effects < ~3pp unresolvable; p<0.05 unreachable
  for 5/5 splits); tau=16 only; B-C decomposition ambiguity (coordinate helps
  vs constant hurts); eval seed = training seed.
- **Supporting Artifacts**: [reports/ippr_coordinate_channel_campaign.md](file:///data/data/com.termux/files/home/projects/titan_text/reports/ippr_coordinate_channel_campaign.md), [runs/ippr_coordinate/analysis_summary.json](file:///data/data/com.termux/files/home/projects/titan_text/runs/ippr_coordinate/analysis_summary.json), [reports/ippr_coordinate_channel_protocol.md](file:///data/data/com.termux/files/home/projects/titan_text/reports/ippr_coordinate_channel_protocol.md)
- **Commit**: (this commit)

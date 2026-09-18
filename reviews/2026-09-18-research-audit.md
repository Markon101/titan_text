# Phase 1: Comprehensive System Audit & Research Plan

**Date**: September 18, 2026  
**Auditor**: Builder (Gemini) & DeepSeek 4.1 Flash Subagent  
**Scope**: Titan Text 1D NCA, 2D Blackboard, Falsification Battery, Stability Diagnostics, and Experimental Validity.

---

## 1. System Inventory & Audit of Requested Capabilities

| # | Capability | Current Implementation Status | What Already Exists | Gaps / Identified Deficiencies | Intended Modifications |
|---|---|---|---|---|---|
| **1** | **Recurrence / write / read ablations** | Implemented in 1D & 2D | 1D: `InterventionConfig` (`disable_recurrent`, `disable_residual`, `recurrence_gain`). 2D: `ReadIntervention` (`Active`, `Zeroed`), `WriteIntervention` (`Active`, `Disabled`, `FreezeAfter(k)`), and `DynamicsMode::StaticBuffer`. | Static buffer control in 2D suffered non-finite gradients under global normalization (fixed with `normalize_written_cell_only`). 1D lacked clean per-step read/write ablation separation. | Ensure identical ablation APIs across 1D and 2D with paired effect-size reporting on complete-answer metrics. |
| **2** | **State reset / freeze interventions** | Implemented | 1D: `reset_step`, `freeze_step` in `InterventionConfig`. 2D: `TemporalIntervention::ZeroAtStep(k)`, `ZeroAtFraction(frac)`. | Interventions were historically evaluated as qualitative strings or aggregate digit accuracy rather than measuring causal degradation $\Delta \text{Acc}$ on held-out tasks. | Expose step-resolved causal probes reporting effect sizes on task success. |
| **3** | **Channel ablation** | Implemented in 1D | `ablate_channel_indices: Vec<usize>` and `ablate_channel_pct: f32` in `InterventionConfig`. | Not plumbed into 2D blackboard rollout; no systematic comparison against random-channel controls. | Expose channel ablation across both 1D and 2D execution pathways. |
| **4** | **Topology scrambling & retraining controls** | Implemented in 2D | `SpatialCoordMode`: `Hilbert`, `FixedShuffled(seed)`, `PerSampleShuffled(seed)`, `InferenceOnlyShuffled(seed)`. | Historical experiments ran with `feedback_mode="global_pool"`, broadcasting global state and masking any spatial routing effect. | Evaluate topological scrambling strictly with `feedback_mode="none"` to isolate local 2D diffusion dynamics. |
| **5** | **Latent tick / budget sweeps** | Implemented | `LatentExecutor::run_latent_budget_sweep_seeded` (budgets 0, 1, 2, 4, 8, 16, 32); `DynamicsMode::FixedTicks(ticks)`. | Previous sweeps were tested on next-token text prediction (memorizing 4 sentences) or degenerate arithmetic where accuracy did not reflect true iterative computation. | Run budget sweeps on tasks whose ground truth genuinely requires multi-step state accumulation (e.g. Iterated Parity and Carry Chains). |
| **6** | **Randomized recurrence-horizon training** | Implemented | `train.rs` supports `horizon_mode`: `"fixed"`, `"randomized"`, `"jitter"`, `"multi_tick"`, `"stability_tail"`. Uses seeded LCG hash. | Unseen horizon evaluation $\tau \notin [h_{\min}, h_{\max}]$ showed severe drift and divergence. | Evaluate horizon robustness on out-of-training steps and test stabilization regimes. |
| **7** | **Stability losses / state-energy control** | Partially implemented | 1D: `state_norm` (`"rms"`, `"layer_norm"`), Navier-Stokes viscous dissipation. `train.rs`: `tail_equilibrium_weight`, `stability_tail`. 2D: `damping_alpha = 0.8`, RMS normalization. | 1D defaulted to `state_norm = "none"`, causing 1024-step autonomous rollouts to explode (state norm ~342, energy ~58,600). | Investigate divergence mechanism; test conservative stabilization (RMS unit sphere projection, leaky integration, damped updates) without freezing transient dynamics. |
| **8** | **Delayed-recall and algorithmic tasks** | Partially implemented / Audited | `tasks.rs` (`DelayedRecall`, `BracketDepth`, `Parity`, `Reverse`, `HiddenRule`, `Associative`, `AmbiguousBasin`, `ColumnArithmetic`). `arithmetic_corpus.rs` (exact carry-depth conditioned corpus). | Legacy tasks suffered severe shortcuts: BracketDepth constant-0 dummy (100%), Parity constant-1 dummy (100%), DelayedRecall 2-sequence collapse, MaxRipple 80%+ constant-0 baseline. | Connect `arithmetic_corpus.rs` to training/evaluation; implement Iterated Parity with Positional Readout (IPPR) where recurrent state is causally required. |
| **9** | **Extrapolation evaluation** | Flawed in legacy | `falsification.rs` had `ExtrapolationEvaluation` (`K=2, 3, 4, 5`). | `extrap_plus2_acc` was previously a manufactured average; nominal OOD ripple carry was dominated by zeros (`100..0`), making constant-0 predictor score 80%+ and creating spurious "OOD is easier than ID" illusion. | Enforce strict split isolation with independent oracles, reporting complete answers and separating difficulty levels. |
| **10** | **Matched parameter/compute baselines** | Implemented | `baselines.rs` (`TransformerBaseline`, `GruBaseline`, `RnnBaseline`, `FeedForwardBaseline`, `UntiedFeedforwardBaseline`). | `UselessCompute1D` in `falsification.rs` was tokenwise MLP without sequence memory; CLI `benchmark --train` ran only 20 gradient steps (non-converged). | Ensure baselines are properly matched in sequence memory, parameter count, and compute steps. |
| **11** | **Multiple-seed statistical evaluation** | Implemented | `exact_mcnemar_binomial`, `combine_fisher_pvalues`, `bootstrap_difference_ci` in `falsification.rs`. Multi-seed training loops. | Previous report used digit-accuracy differences for effect sizes while testing complete-answer booleans where all runs had 0 complete answers. | Align statistical tests with measured metrics and report paired bootstrap confidence intervals. |
| **12** | **Causal interventions at selected latent timesteps** | Implemented | `TemporalIntervention` (`ZeroAtStep`, `NoiseAtStep`), `WriteIntervention::FreezeAfter`. | Tested at coarse fractions rather than probing step-by-step causal curves $t \in [0, T]$. | Profile causal necessity curves across latent timesteps. |
| **13** | **Activation / state recording for offline analysis** | Partially implemented | `InstrumentationTrace` records scalar `TickMetrics` (norm, entropy, gate stats, effective dimension). | No compact per-cell state snapshots or channel trajectories were saved for offline PCA/SVD or decodability probes. | Add opt-in compact per-step activation trace export (`--trace-output <FILE>`) recording channel trajectories and state snapshots. |

---

## 2. Root Cause of Suspicious Generalization Behaviors (Phase 2 Diagnostic)

1. **The "OOD Is Easier Than ID" Illusion**:
   - In legacy `ColumnArithmetic`, the worst-case carry stratum `MaxRipple` ($99\dots9 + 1 = 100\dots0$) produced answers containing a single `1` and $d$ zeros.
   - A dummy predictor emitting constant `0` scored **80.63% digit accuracy** on MaxRipple, versus only ~10% on random in-distribution addition.
   - When metrics reported digit accuracy rather than complete-answer accuracy, the harder ripple-carry test appeared dramatically "easier" than in-distribution examples.
   - In truth, the model scored **0% complete answers** across all seeds and strata.

2. **Task Generator Shortcuts**:
   - `BracketDepth`: Net negative drift and even sequence lengths locked final depth to 0 for 100% of samples. Constant `'0'` scored 100%.
   - `Parity`: The bit generation formula was equivalent to $(s + i) \bmod 2$, making cumulative parity strictly odd for 100% of even sequence lengths. Constant `'1'` scored 100%.
   - `DelayedRecall`: Because $\gcd(3, 6) = 3$, the training universe collapsed to only 2 distinct sequences (`AFED` and `DCBA`). Disjoint validation symbols tested zero-shot transfer rather than sequence recall.

3. **Repairs Required**:
   - Disallow digit-accuracy averaging over unbalanced digit distributions.
   - Score tasks on **complete-answer accuracy** and benchmark against train-fitted digit priors and constant-zero baselines.
   - Deploy `arithmetic_corpus.rs` (which conditions on exact carry depth with canonical operand deduplication) and Iterated Parity with Positional Readout (IPPR).

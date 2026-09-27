# Adaptive Recurrent Compute & Per-Token Halting Campaign Report

**Repository**: `titan_text`  
**Date**: September 27, 2026  
**Branch**: `exp/ascii-adaptive-halting`  
**Checkpoint**: `checkpoints/ascii_v1` (Git commit `ed8ef27`, 43,844 parameters, 64 channels, hidden dim 96, causal DAG stencil $N(i) = \{i-1, i\}$, trained at fixed $\tau=4$)  
**Target Hardware**: Termux / ARM64 Linux on Android (Qualcomm Snapdragon)  
**External Artifact Archive**: `/sdcard/Download/TitanText/adaptive_halting/`  

---

## 1. Executive Summary & Deflationary Epistemic Assessment

The core research question of this campaign:
> «Can Titan allocate different amounts of recurrent latent computation to different generated tokens, and does adaptive compute improve structured ASCII generation more efficiently or more coherently than using one fixed $\tau$ everywhere?»

To answer this question without confounding compute variance with token-state alignment, we executed a **4-arm experimental design** across 135 full autoregressive generation runs (3 structural pattern prompts: `<BOX>`, `<MAZE>`, `<DIAMOND>`; 5 fixed canonical seeds: `42, 101, 202, 303, 404`; 48 tokens rollout per sample).

### Summary Table of Experimental Arms
| Arm | Protocol Description | Mean $\tau$ | Nearest Edit Sim (mean ± se) | Mean H-Symmetry | Exact Matches |
|---|---|---|---|---|---|
| **A: Adaptive** | Dynamic relative delta halting ($\le 0.35$, patience 2) | **2.35** | **0.4679 ± 0.0381** | 0.9651 | 0/15 |
| **B: Fixed $\tau=0$** | State lesion baseline (`--lesion-state`) | 0.00 | 0.0731 ± 0.0162 | 0.0000 | 0/15 |
| **B: Fixed $\tau=1$** | Fixed 1 tick per token | 1.00 | 0.2296 ± 0.0538 | 0.1503 | 0/15 |
| **B: Fixed $\tau=2$** | Fixed 2 ticks per token | 2.00 | 0.4022 ± 0.0302 | 1.0000 | 0/15 |
| **B: Fixed $\tau=4$** | Fixed 4 ticks per token (training regime) | 4.00 | 0.5274 ± 0.0516 | 0.8761 | 0/15 |
| **B: Fixed $\tau=8$** | Fixed 8 ticks per token | 8.00 | **0.6315 ± 0.0221** | 0.7217 | 0/15 |
| **B: Fixed $\tau=16$**| Fixed 16 ticks per token (over-smoothing) | 16.00 | 0.4264 ± 0.0584 | 0.7306 | 0/15 |
| **C: Shuffled** | Shuffled multiset of Arm A ticks (`--tau-schedule`) | 2.32 | 0.3911 ± 0.0273 | 0.9944 | 0/15 |
| **D: Random Sham**| Uniform random $\tau \sim \text{Uniform}(1, 16)$ | 9.00 | **0.5889 ± 0.0428** | 0.6697 | 0/15 |

### Deflationary Bottom Line
1. **The Adaptive Policy Under-Computes**: The adaptive halting mechanism with `patience=2` and threshold `0.35` collapses into a near-constant $\tau \approx 2$ regime ($\bar{\tau} = 2.35$). Every token halts at $\ge 2$ ticks, and 90/90 symbol tokens halt at exactly 2.0 ticks.
2. **Fixed Recurrence at $\tau=8$ and High Random Compute Win in Absolute Quality**: Fixed $\tau=8$ achieves the highest nearest-edit similarity ($0.6315$), followed by the random sham Arm D ($0.5889$ at mean $\tau = 9.0$). Both heavily outperform Adaptive Arm A ($0.4679$) because the model fundamentally benefits from deeper recurrence ($\tau \in [4, 8]$) that the adaptive halting rule prematurely aborts.
3. **Causal Alignment is Marginally Positive but Fragile**: Adaptive Arm A marginally outperforms Shuffled Arm C ($\Delta = +0.0768 \pm 0.0422$, 95% bootstrap CI `[+0.0001, +0.1605]`), but with **10 of 15 runs tying**, the lower bound of the CI touches zero, yielding suggestive, not conclusive, evidence for state-dependent advantage.
4. **Predictive Entropy is Decoupled from Halting**: The correlation between token predictive entropy and halting ticks is statistically null ($r = -0.0827$, 95% CI crosses zero). Halting tracks internal latent velocity, not output categorical uncertainty.

---

## 2. Detailed Empirical Analysis

### 2.1 Fixed Recurrence Sweep & The $\tau=8$ Peak
- **The Recurrent Peak**: Model performance scales monotonically from $\tau=1$ ($0.230$) to $\tau=2$ ($0.402$), $\tau=4$ ($0.527$), and peaks at $\tau=8$ ($0.632$).
- **The $\tau=16$ Over-Smoothing Collapse**: At $\tau=16$, performance drops sharply to $0.426$. This reflects contractive dissipation where excessive recurrence drives the 48-cell latent field toward a smooth, low-rank invariant manifold, erasing discrete character boundaries.
- **Lesion Baseline**: State lesion (`--lesion-state`, bypassing recurrent updates) collapses output to immediate `<eos>` or gibberish ($0.0731$ edit similarity, $0.00$ symmetry), confirming recurrence is causally necessary for text generation.

### 2.2 Arm A vs Arm C (Shuffled Schedule Causal Control)
- **Hypothesis Tested**: Does allocating computation to specific token positions improve output quality beyond merely having that aggregate compute budget?
- **Procedure**: For every prompt and seed, Arm C was supplied the exact multiset of ticks computed by Arm A, permuted deterministically via `seed + 9999` using `--tau-schedule`.
- **Result**:
  - Arm A (Adaptive): $0.4679 \pm 0.0381$
  - Arm C (Shuffled): $0.3911 \pm 0.0273$
  - $\Delta(A - C) = +0.0768 \pm 0.0422$
  - 95% Bootstrap CI: `[+0.0001, +0.1605]`
  - Paired t-statistic: $t = 1.82$ ($p \approx 0.045$ one-tailed, $p \approx 0.09$ two-tailed)
  - Wins: 4, Ties: 10, Losses: 1.
- **Scientific Interpretation**: In 4 runs, Arm A achieved a distinct advantage (e.g. `<BOX>` seed 303: 0.677 vs 0.300 in Arm C where early EOS truncation occurred). However, in 10 runs, the output was identical or near-identical because both arms operated close to $\tau=2$. The effect is real in direction but statistically marginal at $N=15$.

### 2.3 Arm A vs Arm D (Unmatched Random Sham)
- **Result**: Arm D achieves $0.5889 \pm 0.0428$, beating Arm A by $\Delta = -0.1210 \pm 0.0750$.
- **Explanation**: Arm D samples $\tau_t \sim \text{Uniform}(1, 16)$, giving an average compute budget of $\bar{\tau} = 9.0$ ticks. Because the model's highest-quality regime is around $\tau=8$, Arm D benefits from the broad compute elevation. This confirms that adaptive halting under-computed: it stopped too early ($\bar{\tau} = 2.35$) relative to the model's optimal processing depth.

### 2.4 Character-Class Recurrence Allocation
- **Special Tokens (`<eos>`)**: **3.00 ticks** (100% of special tokens halted at 3 ticks).
- **Structural Boundaries (`+`, `-`, `=`, `|`)**: **2.36 ticks** (range 2 to 4 ticks).
- **Newlines (`\n`)**: **2.14 ticks** (range 2 to 3 ticks).
- **Whitespace (` `)**: **2.12 ticks** (range 2 to 3 ticks).
- **Symbols**: **2.00 ticks** (100% of symbols halted at 2 ticks).

While the distribution is narrow (spanning 2 to 4 ticks), the direction matches representational novelty: structural termination (`<eos>`) and corners require more latent velocity settling than repetitive interior symbols.

### 2.5 Predictive Entropy Correlation (Null Result)
- $r(H(P_t), \tau_t) = -0.0827$ ($N=485$, 95% CI `[-0.171, +0.006]`).
- The hypothesis that "adaptive compute halts early on low-entropy tokens and allocates more ticks to high-entropy tokens" is **falsified**. Halting is governed strictly by the $L_2$ relative velocity of the continuous latent state $\|x_t - x_{t-1}\|_2$, which is decoupled from output categorical entropy.

---

## 3. Dataset, Memorization, and Split Hygiene

1. **Model Checkpoint**: `checkpoints/ascii_v1` was trained in a prior session (commit `ed8ef27`, 120 epochs, batch size 8, $\tau=4$) on synthetic multiline patterns.
2. **Evaluation Invariance**: During this campaign, model weights were completely frozen (`checkpoint_frozen: true`). Zero gradient updates or fine-tuning occurred.
3. **Memorization vs Novelty**:
   - Across all 135 runs, exact training match count is **0 / 135 (0.0% memorization)**.
   - The model is not performing table-lookup or verbatim regurgitation; it generates novel ASCII geometries guided by prompt prefixes.
4. **Causal Perception**: The sampler uses a rolling 48-cell context window and causal DAG perception $N(i) = \{i-1, i\}$. No future or ground-truth tokens are accessible during autoregressive rollout.

---

## 4. Software Architecture & Verification

All capabilities were implemented in pure Rust with candle-core and candle-nn:
- `src/ascii_sampler.rs`:
  - `HaltingMode`: `Fixed`, `Adaptive`, `Random`, `Schedule`.
  - `HaltingMetric`: `RelativeDelta`, `StateDelta`, `LogitDelta`, `EntropyDelta`, `Cosine`.
  - Forward-pass logit caching across recurrent iterations.
  - Character classification separating `"newline"`, `"whitespace"`, `"boundary"`, `"alphanumeric"`, `"symbol"`, and `"special"`.
  - Exact softmax probability computation under greedy decoding.
  - Option-wrapped `final_metric_val` (serializing as `null` for non-adaptive modes to eliminate fake zero values).
- `src/cli.rs`: CLI options `--halting`, `--tau-min`, `--tau-max`, `--halting-metric`, `--halting-threshold`, `--halting-patience`, `--tau-schedule`.
- `src/main.rs`: Execution dispatcher, strict validation of halting/metric arguments, structured telemetry logging.

---

## 5. Artifact Manifest & Verification Hashes

All code, campaign scripts, and data artifacts are cryptographically hashed and mirrored to external storage:

| Component | Path | SHA-256 Checksum |
|---|---|---|
| Campaign Script | `scripts/run_adaptive_halting_campaign.py` | `ecfe01a6aa2ca28d36655c992ed24ef7770beafbe032b6766c244492b0baeb6c` |
| Analysis Script | `scripts/analyze_adaptive_halting.py` | `016bc8c89226465ada7590b273431f646348f9bb3cea89c8891166e522134c6a` |
| Campaign Manifest | `reports/raw/adaptive_halting/campaign_manifest.json` | `49976d7b100c2552161d09ed135af1c41ebb4d9169836c8706a16df59ec10905` |
| Analysis Results | `reports/raw/adaptive_halting/campaign_analysis.json` | `f3048bebf3bd8fbd46c15b44fb809fd12ee88ac99567db296234bab50b374f47` |
| Raw Generation Samples | `reports/raw/adaptive_halting/campaign_samples.json` | `3acad2e44328496f66e14accee5151ab9ffdb926aee91e92ac5644637d7a3730` |

Mirror on Android shared storage verified bit-for-bit: `/sdcard/Download/TitanText/adaptive_halting/`.

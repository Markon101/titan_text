# INVALIDATED RESULTS & OBSOLETE RUNS PROVENANCE
**Repository**: Titan Text  
**Date**: September 14, 2026  
**Status**: Formally Invalidated Under Scientific Hygiene Protocol  

---

## 1. Rationale for Invalidation

All checkpoints, traces, and benchmark comparisons generated prior to Git commit `HEAD` on September 14, 2026, are formally invalidated and segregated into this directory due to four verified structural flaws in the legacy experimental pipeline:

1. **Target Exposure via Symmetric Spatial Stencil**:
   - In legacy `src/nca.rs:75-89`, the perception stencil linearly combined central gradient and Laplacian differences ($x_i + \text{grad} + 0.5 \times \text{laplacian} \equiv x_{i+1}$).
   - Because `targets[i] == inputs[i+1]` in natural text next-token prediction, cell $i$ directly perceived the unmasked embedding of its target token from its right neighbor at step 1.
   - All high-accuracy claims on natural text under symmetric stencils were driven by this spatial copy shortcut rather than linguistic modeling.

2. **Direct Answer Leakage in Synthetic Tasks**:
   - **Sequence Reversal** (`src/tasks.rs`): Legacy generator appended the reversed target sequence `rev` directly into `in_chars` at the exact positions where `tgt_chars` were evaluated, reducing the task to an identity mapping $\mathbf{x} \mapsto \mathbf{x}$.
   - **Parity Tracking** (`src/tasks.rs`): Legacy generator placed the answer bit into `in_chars` at cell `num_bits + 1`, immediately adjacent to the evaluation position `num_bits`, allowing the stencil to read the answer directly.

3. **Silent Loss Mask Discard Across Synthetic Tasks**:
   - `src/dataset.rs` discarded `batch.loss_mask` when returning `(batch.inputs, batch.targets)`.
   - `src/vocab.rs` computed unmasked cross-entropy across all tokens.
   - In Delayed Recall, dummy padding '.' tokens accounted for 84.4% of the sequence; in Parity, dummy spaces accounted for 93.8%. Constant dummy predictions scored 84–94% accuracy without performing task computation.

4. **Softmax Temperature Artifact in Dynamic Association**:
   - Unconstrained residual norm inflation ($\mathbf{x}_{t+1} = \mathbf{x}_t + \alpha \delta$) drove state norms from 0.16 to 1.10+, multiplying logits by up to $7\times$ and driving softmax temperature toward zero.
   - Negative controls (stimuli 'z', 'q', or zero-input) also ramped to $>95\%$ confidence on arbitrary tokens.

5. **Participation Ratio Metric Defect**:
   - Legacy `compute_effective_dimension` ignored off-diagonal covariance, returning 64.0 on rank-1 collinear 1D lines.

---

## 2. Catalog of Invalidated Artifacts

The following checkpoints and reports are archived for historical audit only and must NOT be cited as evidence of latent computational emergence:

- `checkpoints/v0_text` (Trained at static $T=8$; right-copy shortcut present).
- `checkpoints/v0_ns_16step` (Trained at static $T=16$; right-copy shortcut present).
- `checkpoints/v0_ns_32step` (Trained at static $T=32$).
- `checkpoints/v0_nvs_16step` (Trained at static $T=16$).
- `checkpoints/v0_viscous` (Trained with viscous damping under symmetric stencil).
- `checkpoints/falsify_dev1`, `checkpoints/falsify_dev2`, `checkpoints/falsify_dev4` (Historical exploration of fixed-horizon BPTT).
- `reports/diagnostics*.json` (Pre-audit diagnostic dumps).

---

## 3. Fresh Replacement Standards

All fresh runs must adhere to:
1. Validated tasks with zero answer leakage.
2. Properly applied `loss_mask` for synthetic tasks.
3. Explicit manifests containing Git commit hash, random seed, task name, training horizon $T_{\text{train}}$, evaluation horizon $\tau_{\text{eval}}$, thread count, and ISO 8601 timestamp.
4. True covariance Participation Ratio metric.
5. Normalized or temperature-standardized association probes.

---
name: titan-research
description: Run controlled Titan Text training and causal evaluation experiments while preserving seeds, checkpoints, ablations, claims, research debt, and reproducible artifacts.
---

# Titan Text Research Protocol

This skill governs scientific inquiry, causal experimentation, and falsification in the Titan Text repository.

## 0. Chunked-resume training artifact (2026-09-26, coordinate-channel campaign)

When training runs exceed ~10 min, do NOT split them into chained 100-epoch increments (`train --load-dir` resume loops). Adam optimizer state (moment estimates) is NOT persisted in checkpoints — only weights. Every resume resets Adam and can destabilize training: in the IPPR L=16 battery ALL 5 fresh-seed baseline runs diverged (NaN loss) under chunking while an identical single-shot 1000-epoch run converged cleanly. If chunking must be used, it applies equally to every arm and must be validated against a single-shot control run before any arm contrast is interpreted. Also: serde_json serializes NaN/Inf as `null` — NaN diagnostics silently corrupt manifests; keep manifest diagnostic fields `Option<f32>` with finite-filter writers, and detect divergence by reading the manifest (null/nonfinite diag or train_accuracy==0), never by relying on nonzero exit codes.

## 1. Scientific Standards & Research Norms

1. **Mechanistic Claims Require Causal Interventions**:
   - Never claim an architecture feature (e.g. latent recurrence, carry channels, causal stencils) performs a specific computation without running matched counterfactual ablations.
   - For recurrent claims, test latent tick budgets ($\tau \in \{0, 1, 2, 4, 8, 16\}$), state lesions (`--lesion-state`), and gate clamping (`--lesion-gates`).
   - Distinguish capability demonstrations from mechanistic claims.

2. **Matched Baselines & Fixed Seeds**:
   - Compare NCA against canonical sequence baselines: Simple RNN, GRU, Transformer, Pointwise (`titan_text benchmark`).
   - Always run evaluations over the fixed seed battery (`42, 101, 202, 303, 404`).
   - Retain full hyperparameters and seeds in checkpoint manifests and report JSON files.

3. **Honesty & Negative Results**:
   - Record negative results, dead ends, and unexpected failures.
   - Do not silently re-seed or cherry-pick until a nice curve appears.
   - Distinguish exploratory searches from preregistered confirmatory benchmarks.

4. **Claim and Debt Ledgers**:
   - Established and falsified claims are recorded in [docs/RESEARCH_STATE_PACKET.md](../../../docs/RESEARCH_STATE_PACKET.md) and [reports/live_claim_ledger.md](../../../reports/live_claim_ledger.md).
   - Unresolved technical debt and confounds are tracked in [reports/research_debt_ledger.md](../../../reports/research_debt_ledger.md).
   - The IPPR bulk-interior stagnation ($L=16$) and coordinate-channel investigation is an open mechanistic thread and must not be misrepresented or discarded.

## 2. Common Research CLI Workflows

```bash
# 1. Latent recurrence sweep across tick budgets:
target/release/titan_text sweep --load-dir checkpoints/<CHECKPOINT> --task <TASK> --budgets 0,1,2,4,8,16

# 2. State lesion causal evaluation:
target/release/titan_text rollout --load-dir checkpoints/<CHECKPOINT> --task <TASK> --lesion-state --horizon 32

# 3. Baseline architectural comparison:
target/release/titan_text benchmark --task dyck-pushdown --seq-len 64 --batch-size 32
```

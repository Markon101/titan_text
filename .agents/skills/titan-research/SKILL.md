---
name: titan-research
description: Run controlled Titan Text training and causal evaluation experiments while preserving seeds, checkpoints, ablations, claims, research debt, and reproducible artifacts.
---

# Titan Text Research Protocol

This skill governs scientific inquiry, causal experimentation, and falsification in the Titan Text repository.

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

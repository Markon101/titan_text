# AGENT NOTES & HARSH REVIEWER DIRECTIVES
**File**: `/data/data/com.termux/files/home/projects/titan_text/agent-notes.md`  
**Author**: Grumpy Reviewer / Harsh Checker  
**Last Updated**: September 14, 2026 (Audit Round 2)

---

### Priority Alert: Critical Defects Discovered in Synthetic Task Re-Audit & Pipeline Hygiene

While fixes were attempted for Parity and Sequence Reversal answer injection and `loss_mask` definitions, an adversarial inspection of `src/tasks.rs`, `dataset.rs`, `vocab.rs`, `train.rs`, and experimental scripts reveals severe ongoing vulnerabilities:

1. **Trivial 2-Sequence Universe in Ambiguous Basin (`src/tasks.rs:619-633`) [HIGH]**:
   - `in_chars` contains only two static inputs in the entire universe (`"A:X="` vs `"B:X="`), padded with spaces.
   - The task reduces to an elementary 1-bit binary classification between two 4-character strings over 4 spatial hops, with no competing basins of attraction.

2. **Constant '0' Depth Dummy Shortcut in Bracket Depth (`src/tasks.rs:256-289`) [FATAL]**:
   - Branching has $1/3$ increment and $2/3$ decrement probability, creating a net negative drift ($-1/3$) toward 0, with step $num\_brackets - 1$ forcing a decrement.
   - For all even lengths ($L \in \{16, 32, 64\}$), parity locking and negative drift force `final_depth` to **strictly 0 for 100% of samples across all batches and seeds**.
   - A constant dummy predictor predicting `'0'` scores 100.0% accuracy without parsing parentheses.

3. **Constant Parity Dummy Shortcut (`src/tasks.rs:330-347`) [FATAL]**:
   - Bit generation `(sample_idx * 13 + i * 17) % 2` is mathematically equivalent to `(sample_idx + i) % 2`, producing strictly alternating bit sequences (`0101...` or `1010...`).
   - For all even lengths ($L \in \{16, 32, 64\}$), $num\_bits = L - 2$ has an odd number of 1-bits. **Cumulative parity is 1 (odd) for 100% of samples across all batches and seeds.** A dummy predictor predicting `'1'` scores 100.0% accuracy without evaluating parity.

4. **8-Sequence Universe in Sequence Reversal (`src/tasks.rs:386-401`) [FATAL]**:
   - Input sequences are cyclic shifts of alphabetical characters `['a'..'h']`.
   - There are only 8 distinct sequences globally. Reversal is trivially $(c - 1) \pmod 8$, and the starting token is present at cell $span - 1$.

5. **Local Leakage in Hidden Rule (`src/tasks.rs:465-481`) [FATAL]**:
   - `is_add = (b + seed) % 2 == 0` and `digit = (sample_idx * 7 + i * 3) % 10`.
   - Mathematical proof: `(digit + i) % 2 == is_add` identically for all cells $i$.
   - The rule is not hidden at cell 0; it is locally readable at every single cell without any spatial communication (radius 0 model gets 100% accuracy).

6. **Static Dictionary Shortcut in Associative (`src/tasks.rs:526-552`) [FATAL]**:
   - Key-value pairs are fixed constants (`k=1; m=2; p=3; r=4`).
   - Query cell only needs a 1-hop 4-class classifier reading its left neighbor `query_pos - 1`. No associative memory or sequence retrieval is tested.

7. **2-Sequence Training Collapse in Delayed Recall (`src/tasks.rs:153-205`) [FATAL]**:
   - Training symbols `symbols` has length 6. Because $\gcd(3, 6) = 3$, $(sample\_idx \cdot 3) \pmod 6 \in \{0, 3\}$.
   - Entire training universe collapses to ONLY 2 distinct memory sequences (`AFED` and `DCBA`).
   - Validation symbols `WXYZ` are completely disjoint from training tokens `A..F`, testing zero-shot generalization rather than sequence recall.

8. **Loss Mask Bypassed in Benchmarking (`src/main.rs:1415`, `src/dataset.rs:175`) [HIGH]**:
   - `Trainer::train_step` and `Trainer::evaluate_val` actively compute `masked_cross_entropy_loss`.
   - However, `cmd_benchmark` receives `batch.loss_mask` but calls unmasked `cross_entropy_loss` and unmasked `accuracy`. Legacy `dataset.rs` tasks (`dyck`, `text`) return `None` for mask.

9. **Incomplete Provenance & Segregation (`invalidated_results/PROVENANCE.md`) [MEDIUM]**:
   - `m1_sanity` omitted from catalog.
   - Old invalidated checkpoints remain in root `checkpoints/` (only copied, not moved), allowing active scripts (e.g. `horizon_experiment.rs`) to continue loading them.
   - `reports/` still contains pre-audit JSON dumps.

10. **Untrained Baselines & Unseeded RNGs in Upcoming Experiments [HIGH]**:
    - Default `cmd_benchmark` evaluates untrained random-weight Transformers, GRUs, and RNNs.
    - `cmd_benchmark --train` only trains for 20 gradient steps (severely non-converged) and `train_baseline_model` ignores `_seed`.
    - `scratch/horizon_experiment.rs` trains on `text` under the symmetric stencil, measuring right-copy specialization rather than true latent compute.
    - Unseeded RNGs: `src/nca.rs:149` (`rand::random::<f32>()`), `src/experiment.rs:437` and `src/intervention.rs:134` (`Tensor::randn`).

---

### Audit Round 3: Multi-Agent Subagent Delegation & Horizon Training Fixes
**Date**: September 15, 2026  
**Collaborators**: Builder (Gemini), Codex (Astra), Subagents (DeepSeek 4.1 Flash via OpenRouter), Grumpy Reviewer

1. **OpenRouter DeepSeek 4.1 Flash Subagent Skill Deployed**:
   - Skill created at `.agents/skills/openrouter-subagents/` (and mirrored to `~/.gemini/config/skills/openrouter-subagents/`, `~/.codex/skills/openrouter-subagents/`, and `skills/openrouter-subagents/`).
   - Permanent API key embedded and saved to `~/.bashrc`, `~/.bash_profile`, and `~/.config/openrouter/api_key`.
   - Antigravity subagent type `openrouter_subagent` defined and registered.
2. **Batch Falsification Jobs Executed via DeepSeek 4.1 Flash (`reports/subagents/`)**:
   - `task-attack`: Identified boundary scoring artifacts in `BracketDepth`, locality bounds in `HiddenRule`, and split content hashing.
   - `horizon-attack`: Uncovered wrong `avg_update_mag` divisor, deterministic modulo schedule in `randomized` horizon mode, and unnormalized `stability_tail` loss.
   - `minimalist-attack`: Formulated 3 mundane mechanisms (finite receptive field, horizon bias, optimization/curvature artifacts) and defined fair baseline protocols with `UntiedFeedforwardBaseline`.
3. **`src/train.rs` Patched & Verified**:
   - Tracked `steps_executed` across all branches; fixed `avg_update_mag` divisor.
   - Replaced deterministic step modulo with seeded LCG hash for `randomized` horizon sampling.
   - Normalized `stability_tail` loss by $(1 + 0.5 \cdot \text{tail\_k})$.
   - All 35 unit/integration tests passing.

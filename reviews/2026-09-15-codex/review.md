# Gemini + DeepSeek work review and research continuation

Date: September 15, 2026. Scope: the dirty `main` checkout at `0b59e11`, with emphasis on the newest 2D blackboard, arithmetic generator, and falsification results. Existing work and checkpoints were preserved. This is an assessment of the artifacts, not of the agents' intentions.

## Assessment

The new 2D model has a working causal read/write path and can receive useful gradients. The saved five-arm experiment does **not** establish arithmetic generalization, useful adaptive computation, or an advantage of 2D topology. Several failures are in the experiment and reporting, so they also do not disprove the architecture.

### Findings, in priority order

1. **Critical: the static-buffer control is numerically invalid.** All three historical static runs contain `null` training losses and BUR values, consistent with non-finite floats serialized to JSON. A fresh, deterministically initialized two-example backward pass reproduces non-finite gradients in the embedding and write projection before the first optimizer update. `normalize_state` uses epsilon `1e-5`; at zero its Jacobian is approximately `316 I`. Normalizing the entire mostly empty field on every write repeatedly amplifies derivatives at untouched cells. Evidence: `audit-before.json`, `audit-after.json`; implementation: `src/nca2d.rs`, `normalize_after_write`.

2. **Critical: arithmetic scores are dominated by simple output frequencies.** On the exact seed/chunk schedule used by the historical validation, a constant `0` predictor beats every fixed-blackboard seed:

   | Seed | Saved fixed blackboard, digit accuracy | Constant `0` | Complete answers from saved model |
   |---|---:|---:|---:|
   | 42 | 34.26% | 44.26% | 0/256 |
   | 101 | 32.38% | 45.39% | 0/256 |
   | 202 | 22.97% | 48.75% | 0/256 |

   The intervention panel has 126 unique observations among 256 rows. Constant `0` scores **50.23%**, exactly the saved write-disabled model's accuracy, versus **38.95%** intact. The MaxRipple panel contains only **nine** distinct observations among 128 rows; constant `0` scores **80.63%**, versus **59.61%** for the saved model. LongPartial is similarly zero-heavy: constant `0` **81.17%**, model **73.44%**. These are descriptive comparisons; no new paired significance claim is made from missing historical predictions. Regeneration found no train/validation overlap for the audited schedules, so the demonstrated problem is evaluation composition and low diversity, not observed cross-split leakage.

3. **Critical: the report mixes statistical outcomes and manufactures an extrapolation score.** Its displayed effects are digit-accuracy differences, while McNemar and bootstrap receive complete-answer booleans. All 15 saved runs have zero correct complete answers. Failure to reject on those outcomes cannot justify the reported bypass/topology conclusions. `extrap_plus2_acc` was `(s2_acc+s1_acc)/2`, with no K=5 evaluation; LongPartial spans K=2 through K=8 and is not K=4. `scratch/test_stats.py` additionally assumes 2,304 query tokens where the actual panel has 2,560, and constructs invented paired outcomes from aggregate counts. That script cannot recover valid paired uncertainty.

4. **High: the purported 1D recurrent control has no sequence memory.** `UselessCompute1D::forward` repeatedly applies an MLP separately to each token embedding. Every query is `?`, so answers cannot depend on operands or query position. The new operand intervention gives exactly zero answer-logit change for this arm, versus a nonzero change for the dynamic blackboard. Its listed MACs are also about 28 times smaller than the fixed blackboard's, despite the CLI claiming matched compute. It is useful only as a tokenwise negative control.

5. **High: halting telemetry does not support adaptive efficiency.** Batch-wide step counts were divided by batch-times-sequence length, shrinking the mean by 32; the field named median stored this mean and p95 was hard-coded to 6. The saved histogram actually implies mean **1.09375**, median **1**, p95 **1**. Stopping below the cap does not show an accuracy/compute advantage. Halting is decided using a mean over the entire microbatch, so another example can change an example's stopping time. Depth-stratified BUR also assigns each batch's scalar to every example, and cannot measure individual utilization.

6. **High: provenance and review coverage were overstated.** `train_arm` did not call the existing seeded initializer. Report timestamp and commit were hard-coded; trained arm weights were not retained by this runner. The latest `nca2d-code-review.json` has `files: []`, `finish_reason: "length"`, and explicitly says it did not receive the file. Its hypothetical code suggestions are not a source audit. Several other latest DeepSeek outputs were likewise truncated, prompt-only proposals. Earlier task/masking improvements are real, but the separate stratified arithmetic generator bypasses the generic content-split safeguards and formerly fell back silently to `0+0` after impossible rejection constraints.

7. **Interpretation constraint: global pooling breaks strict local communication.** The experiment enables `feedback_mode="global_pool"`, broadcasting information throughout the grid. Its behavior cannot isolate an advantage from local Hilbert routing. The current residual path is a raw token embedding, not a trained external language model. Earlier fluid, basin, and monotonic-compute narratives in the long research documents remain hypotheses; finite measurements or architecture terminology do not establish them. The legacy bidirectional text objective also remains unsuitable as causal-language evidence.

## Repairs implemented

- Added `BlackboardConfig.normalize_written_cell_only`, an opt-in write rule that preserves the identity path at untouched cells. The default and missing-field deserialization preserve the legacy write rule. Tests compare both rules across seeds 42, 101, and 202: legacy static gradients fail; the opt-in gradients are finite and the write projection remains connected.
- Seeded fresh falsification-arm initialization through the existing initializer; reject non-finite loss/parameter gradients before AdamW updates.
- Reject incompatible or exhausted arithmetic sampling requests rather than silently replacing them with `0+0`.
- Align statistical effects with complete-answer tests, qualify first-seed scope, apply a conservative three-comparison threshold, and report inconclusive results without treating them as proof of a null. These remain exploratory statistics on a dataset whose composition requires repair.
- Compute halting mean/quantiles from observed counts with correct batch weighting. Remove the automatic adaptive-advantage verdict.
- Emit null for unmeasured K=4/K=5 panels, retain the measured mixed-depth panel separately, replace hard-coded provenance, and include interpretation limitations in new reports. Report fields changed; historical JSON files were not rewritten.
- Relabel the tokenwise control honestly. Its implementation is preserved.

The original five-arm `falsify` command still selects the legacy write rule. It now stops on invalid gradients; it is **not** a validated full benchmark. Use the isolated research runner below for the opt-in experiment. Further corpus and protocol work is required before restarting the full battery.

## New research

The CPU audit uses actual Rust task generation and model operations, deterministic fresh initialization, two threads, and exclusive creation of each output file. It verifies future-prefix invariance and operand sensitivity, audits simple predictors on the historical schedules, and runs a small optimization gate. No checkpoints are loaded or saved.

First gate: fixed and static blackboards with the opt-in write rule, plus a genuine sequence GRU with 47 hidden units. Parameter counts are 22,979 versus 22,941 (0.17% difference); active parameter counts and compute are not equal. Each fits the same four length-16 arithmetic examples, across three initialization seeds. This is deliberately a memorization check, not an arithmetic benchmark.

At 32 updates, the dynamic blackboard fits all four answers in all three seeds. The static buffer remains finite but fits 45–50% of digits and no complete answers; the GRU reaches 55–60% of digits and no complete answers. The dynamic model takes roughly 7–8 seconds versus around 0.1 seconds for the GRU, so equal optimizer steps are not equal compute. A follow-up gives the GRU 256 updates and evaluates all models on 128 distinct, content-disjoint held-out examples. The follow-up completed:

| Model | Updates | Training complete-answer accuracy, seeds 42/101/202 | Held-out digit accuracy, seeds 42/101/202 | Held-out complete answers |
|---|---:|---|---|---|
| Static buffer, local write normalization | 32 | 0% / 0% / 0% | 18.13% / 19.38% / 16.09% | 0/128 in every seed |
| Dynamic blackboard, local write normalization | 32 | 100% / 100% / 100% | 17.03% / 21.09% / 19.53% | 0/128 in every seed |
| GRU, 47 hidden units | 256 | 100% / 100% / 100% | 17.66% / 16.88% / 18.28% | 0/128 in every seed |

The held-out panel contains 128 unique observations, no overlap with the four training examples, and a constant-zero digit baseline of 17.34%. All gradients remained finite throughout all nine runs. The GRU also passes the memorization gate, so the blackboard's 32-update result is not evidence of a unique computational capability. Training on four examples is intentionally insufficient for a meaningful generalization comparison; zero held-out complete answers should not be interpreted as proof that either architecture cannot learn addition.

Recorded follow-up training times were about 23–24 seconds for the blackboard and 2.6 seconds for the GRU despite its eightfold update budget. Times vary substantially between runs on this phone, so these are workload observations rather than a controlled speed benchmark. The follow-up used more GRU updates to test basic trainability, not to claim a fair quality or efficiency win.

## Next research gates

1. Build a new arithmetic schema with diverse operands conditioned on **exact** carry depth, balanced scoring by query position/depth, and explicit content-disjoint train/validation/test sets. Keep the nine-template ripple panel as a named stress fixture, not a generalization test. Use an independent integer-addition oracle and count duplicates before training.
2. Require held-out complete-answer performance above position-only and digit-prior controls. Preserve per-example IDs, predictions, masks, depths, and checkpoint/source hashes. Separate same-seed paired effects from training-seed variability; resample whole examples for digit metrics, never fabricate pairs from totals. SciPy's [paired-bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) provides the relevant resampling contract.
3. Compare a real GRU, a 1D spatial model, and the 2D blackboard with parameter and measured compute budgets reported separately. Retain static and write-disabled controls. Test Hilbert versus retrained shuffled routing with global pooling both enabled and disabled.
4. Only after task competence, compare fixed tick counts with adaptive policies at measured accuracy/compute tradeoffs. This is motivated by [Fojo et al.'s fixed-versus-adaptive comparison](https://arxiv.org/abs/1803.08165), which found fixed repetition competitive on their selected tasks; it is not a prediction of the result here.
5. Test persistence and recovery separately from arithmetic accuracy. [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) explicitly distinguishes growth, persistence, and regeneration training. Those experiments motivate specific tests; they do not imply that this text blackboard has those properties.

## Reproduction and evidence

From the repository root, choose a fresh output filename:

```sh
cargo test --locked --offline
cargo run --release --locked --offline --example blackboard_audit -- reviews/2026-09-15-codex/research-rerun.json
python reviews/2026-09-15-codex/summarize.py
```

- `research-heldout.json`: final raw audit and follow-up results.
- `summary.json`: tables recomputed by `summarize.py`.
- `audit-before.json` / `audit-after.json`: original failure and opt-in gradient comparison.
- `research.json`: first optimization gate, before extending GRU training and adding held-out evaluation.
- `source-at-research/`, `research-build-hashes.json`: source and executable identity for the first optimization run.
- `source-at-followup/`, `followup-build-hashes.json`: final follow-up source and executable identity.
- `provenance.json`: starting dirty status and hashes, followed by preservation checks.
- `tests.log`: 58 unit tests and 3 CLI integration tests passed (61 total).
- `release-build.log`: release binary rebuilt; command help and invalid-seed smoke checks are captured separately.

Existing whitespace warnings in `README.md` and `src/vocab.rs` predate these changes. They were left intact. No commit or push was performed.

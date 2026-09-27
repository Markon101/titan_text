# PREREGISTRATION: Adaptive-Halting Qualification Control Battery

**Frozen before execution.** Written 2026-09-26, branch `exp/ascii-adaptive-halting`,
HEAD at protocol commit (recorded in run manifest). Changes after freezing are
permitted only for demonstrated bugs / invalid runs, and must be documented in
the changelog section at the bottom.

## Research question

Does STATE-DEPENDENT allocation of recurrent computation improve structured
ASCII generation beyond what is explained by total compute budget, random
variation in tau, the nearest fixed operating point, or simple
positional/structural schedules?

## Fixed conditions (unchanged from prior campaigns)

- Checkpoint: `checkpoints/ascii_v1` (sha256 model.safetensors
  `97d00bd0085052e0ddd82799ffc0fb9451e523d7bd6243e1db7898a62ec13e3e`,
  manifest `2d3bdedbcb1b4cdc300b264719a5b2f78c14e034c95baf72403b75ded8b0c86d`,
  43,844 params, 64 ch, hidden 96, causal stencil {i-1,i}, carry_channels=0)
- Held-out seeds: [501,502,503,504,505,601,602,603,604,605] (never tuned on)
- Prompts: `<BOX>\n`, `<MAZE>\n`, `<DIAMOND>\n`
- max_len 48, temperature 0.7, top_k 0
- halting: metric `relative_delta`, theta 0.25, patience 2, tau_min 1, tau_max 16
- Primary endpoint: `nearest_edit_similarity` (existing metric, unchanged)
- Seed 42 calibration battery NOT reused for inference; prior artifacts untouched.

## Arms (30 runs each; 10 seeds x 3 prompts; 180 total)

| Arm | Construction |
|---|---|
| A_adaptive | `--halting adaptive` (frozen theta*) |
| B_fixed_tau4 | `--tau 4` |
| B_fixed_tau5 | `--tau 5` (mandatory compute-matched fixed comparator; adaptive realized mean 4.49) |
| B_fixed_tau8 | `--tau 8` (high-compute reference) |
| C_matched | Schedule mode; per-(prompt,seed) exact multiset of SAME-BATTERY Arm A ticks, permuted with `random.Random(seed+9999)` (state-independent within-sequence reallocation) |
| D_matched | Schedule mode; per-(prompt,seed) i.i.d. draws from the battery-global adaptive tick empirical distribution (4:w641, 5:w529, 6:w30 over held-out Arm A), length = Arm A sequence length for that pair, rng `random.Random(seed+7777)` recorded in the sample record (state-independent random compute at matched distribution+mean) |

Schedule length for C/D = the Arm A emitted-token count L for the SAME
(prompt, seed), constructed after Arm A completes in the same battery run.

## Compute matching rules (define before seeing outcomes)

- Every sample records: total_ticks, steps_generated (emitted tokens),
  mean/median tau, and for C/D: requested vs consumed schedule entries.
- Matching tolerance: D_matched per-pair realized total ticks within +/-10% of
  Arm A in expectation (sampling noise allowed; report realized distribution).
- Truncation: if a C/D run emits `<eos>` early, its realized compute and length
  differ. This is retained in the PRIMARY analysis (misallocation causing
  truncation is part of the allocation strategy's outcome); a pre-registered
  SENSITIVITY analysis repeats comparisons excluding pairs where the control
  run's emitted length < 0.8 x Arm A length. Both are reported; neither is hidden.

## Statistical unit and tests (frozen)

- True independent unit: the 10 held-out SEEDS (sequences nested in seeds; 3
  prompt-sequences per seed). N=30 paired sequence comparisons, 10 independent
  clusters. This limit is acknowledged in all claims.
- Primary comparisons: A - X for X in {B4, B5, B8, C_matched, D_matched},
  paired per (prompt, seed):
  1. mean/median paired difference + per-pair differences listed;
  2. seed-cluster bootstrap 95% CI (10 clusters, 10,000 resamples of seeds,
     preserving within-seed pairs);
  3. exact sign-flip permutation test over the 30 paired differences.
- Token-level statistics (entropy/tau, positional) are DESCRIPTIVE ONLY unless
  cluster-aware (sequence-level summaries + cluster bootstrap) are attached.
- Decision rule (frozen): adaptive state-dependence is SUPPORTED only if
  A > B5 AND A > D_matched AND A > C_matched in paired mean with cluster CI
  excluding zero for C_matched and D_matched. If A <= B5, the budget hypothesis
  (H_BUDGET) explains the headline and claims are downgraded.

## Hypotheses under test

H_ADAPTIVE, H_BUDGET (nearest fixed operating point), H_VARIANCE (variable tau
itself helps), H_POSITION (positional/structural prior), H_EOS (truncation
artifacts), H_THRESHOLD (post-hoc theta choice), H_METRIC (halting metric is a
broad channel-scale artifact; carry-channel contamination classified
NOT APPLICABLE: ascii_v1 carry_channels=0).

## Failure / stop conditions

- checkpoint hash mismatch; binary/config mismatch vs manifest; nondeterministic
  reruns of identical (arm, prompt, seed); crashed runs without logged reason;
  any need to alter theta. Stop, log, repair before interpreting.

## Changelog after freezing

- [pre-run, documented] Clarifications from independent protocol review, all
  verified against source (no behavioral changes):
  1. relative_delta pinned: `||x_t - x_{t-1}||_2 / (||x_t||_2 + 1e-6)`,
     denominator = post-step state norm (src/ascii_sampler.rs:373-379).
  2. patience pinned: consecutive sub-threshold ticks required; counter resets
     on any supra-threshold tick; gating active only for tick >= tau_min
     (src/ascii_sampler.rs:410-419).
  3. D_matched interpretation pinned: tests marginal-compute equivalence only,
     not conditional/state-dependence equivalence (distribution is a plug-in
     of Arm A's marginal by design).
  4. Added null-lesion sanity run (--lesion-state on one cell) to confirm
     zero-tick bypass behaves as recorded (tau=0 baseline consistency).
  5. Determinism check: one identical rerun of an Arm A cell; outputs must
     match bit-for-bit (RNG: per-run StdRng::seed_from_u64(cfg.seed),
     src/ascii_sampler.rs:279; C/D schedule RNGs are separate Python streams,
     re-instantiated per (prompt, seed)).
  6. Truncation asymmetry acknowledged: A's early-EOS length propagates into
     C/D schedule length by design (paired); primary analysis keeps all pairs,
     sensitivity excludes control-truncated pairs (< 0.8 x A length) only.

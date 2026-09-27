# IPPR Coordinate-Channel Campaign — L=16 Interior Failure (Confirmatory)

Branch: `exp/ippr-coordinate-channel`
Preregistration commit: **6670875** (committed and pushed BEFORE any battery run)
Battery/infra commits: 57f2a88 (single-shot fix), plus final results commit (below)

## RESEARCH QUESTION

Why did Titan learn genuine instance-specific sequential parity computation at
L=8 but fail in the bulk interior at L=16?

- **H_COORD**: missing explicit positional information blocks interior binding;
  a scalar coordinate channel should rescue Slots 1/2.
- **H_ATTENUATION / H_OPT**: state propagation / attenuation / an optimization
  (credit-assignment) barrier limits the interior; explicit coordinates should
  not rescue it.

## CANONICAL PRIOR STATE (verified against raw artifacts)

- L=8: intact slots [68.75, 68.75]% vs shuffle 50.0% → G_identity ≈ +18.8pp
  (campaign_arm2_curriculum.json, epochs 200–500). REPLICATED.
- L=16: interior slots 40.0–54.4% (chance ≈ 50%) across
  campaign_arm1_from_scratch.json, campaign_arm2_curriculum.json (epochs
  600–1000), campaign_arm2_coordinate.json. REPLICATED.
- Prior coordinate test (2026-09-18): coord-trained model intact 57.34%,
  interior 48.75/54.38; counterfactuals within 1–3% (G_coord_zeroed +1.41% <<
  +8%) → H_POS falsified, but on ONE trained model per arm (eval seeds only)
  and with NO training-time sham arm.
- Probes: prefix-parity physically absent downstream; causal flip prob 0.0%
  (prefix_parity_probes.json, causal_influence_analysis.json,
  state_inversion_probes.json).
- NEW this campaign: converged-baseline state lesion at L=16 ≈ 48–52% →
  recurrence contributes only ~5–9pp overall; report Table 1 of the prior
  campaign ("lesion 0.00%") reflects early-epoch records, not convergence.

## HYPOTHESES

As preregistered (H_COORD / H_ATTENUATION / H_OPT); decision rules frozen in
`reports/ippr_coordinate_channel_protocol.md` §5.

## ARCHITECTURE CHANGE

One scalar coordinate channel p_i = 2i/(L−1) − 1 ∈ [−1,1] appended to the
perception vector of every position (`NCA::generate_coordinates`,
src/nca.rs:139; concat in the single shared perception path
`perceive_with_feedback_and_seed`, src/nca.rs:320–327 — identical for training
and eval). dense1 gains +1 input (3C → 3C+1) = +128 weights ≈ +0.29%.
Training-time modes via `NcaConfig::coord_train_mode` (`--coord-train-mode`):
`constant` (sham, 0.5 everywhere), `zeroed`, `shuffled`, `reversed`.

## SHAM DESIGN (frozen before results)

Arm C trains the identical added channel with CONSTANT value 0.5 — zero
positional information, exact parameter match with Arm B, input support inside
the intact range. Known limitation (acknowledged pre-run): a constant input is
absorbable into dense1 bias, so Arm C ≈ Arm A is expected even if extra
capacity would help; the informative contrast for POSITION is B vs C (and B vs
A). C vs A acts as a weak capacity control only.

## LABEL-LEAKAGE AUDIT

`generate_coordinates(b, l, mode, device)` is a pure function of position and
L (src/nca.rs:139–172); no targets, inputs, or batch statistics enter. Sham
constant cannot encode labels. Eval batching mirrors training batching
(batch 32, sweep). Code-auditor review of the pre-run packet raised no
leakage; the only unverifiable-at-the-time item (training-loop construction)
was confirmed by direct inspection of `Trainer::train_step` (src/train.rs) —
perception is built from inputs only.

## PARAMETER DELTA

A: dense1 [128, 192] (3C=192); B/C: [128, 193] → +128 weights (+bias equal).
B and C match exactly; A differs minimally as preregistered.

## TRAINING CONFIG (frozen)

iterated-parity, L=16, `--dev-steps 16 --zero-boundary`, 1000 epochs, batch 32,
default lr/viscosity/state-norm. Seeds 801–805 (verified fresh: full-repo
standalone-integer search found only step numbers/hash substrings/weight bytes;
prior family used 42/101/202/303/404). Evaluation: `sweep --budgets 16
--batch-size 32`, paired eval seed = training seed; controls: lesion-state,
batch-state shuffle, coordinate counterfactuals (Arm B).

## PROCEDURAL CORRECTION (documented per prereg §7)

The first battery used chained 100-epoch resumes and ALL 5 fresh-seed baseline
runs diverged (NaN loss) — Adam optimizer state is not persisted across resume.
An identical single-shot baseline run (seed 801) converged cleanly
(loss 0.094, acc 93.75%). The chunked battery was archived UNMODIFIED at
`runs/ippr_coordinate_chunked_invalid/` and the full battery rerun single-shot
from scratch. Chunk-resume is therefore barred from this experiment family.
(Manifest NaN→null serialization also fixed: diagnostics now Option<f32>.)

## RESULTS

### Baseline replication (prereg §5 gate) — PASS

15/15 runs converged single-shot (no divergences; final train_loss 0.0002–0.09).
All 5 converged baselines show the interior failure at tau=16:

| Arm/seed | Slot0 | Slot1 | Slot2 | Slot3 | InteriorMean |
|---|---|---|---|---|---|
| A/801 | 71.88 | 37.50 | 40.62 | 56.25 | 39.06 |
| A/802 | 75.00 | 50.00 | 46.88 | 46.88 | 48.44 |
| A/803 | 71.88 | 37.50 | 40.62 | 46.88 | 39.06 |
| A/804 | 59.38 | 43.75 | 46.88 | 46.88 | 45.31 |
| A/805 | 65.62 | 53.12 | 56.25 | 50.00 | 54.69 |

Zero baselines exceed chance+8pp → gate PASS; the coordinate question is
well-posed.

### L=16 primary endpoints (seed-level, paired)

InteriorMean per arm (seeds 801–805):
- A baseline: 39.06, 48.44, 39.06, 45.31, 54.69 → mean 45.31
- C sham:     34.38, 43.75, 35.94, 45.31, 46.88 → mean 41.25
- B coord:    39.06, 48.44, 42.19, 50.00, 48.44 → mean 45.63

Paired contrasts (exact sign-flip over 2^5, seed-level bootstrap CI):
- **B − A: +0.31pp [−3.12, +3.44], p = 0.938 (+3/5 seeds)** — exact null.
- **B − C: +4.38pp [+2.81, +5.63], p = 0.062 (+5/5 seeds)** — below the
  preregistered +8pp effect threshold; p = 0.062 is the exact-test FLOOR at
  n=5 (5/5 split), so α=0.05 is unreachable for any perfectly consistent
  direction at this n — a preregistration power limitation, not evidence.
- C − A: −4.06pp [−6.25, −1.87], p = 0.125 (0/5 seeds positive).

Decomposition: B − C = (B − A) − (C − A) ≈ 0.31 − (−4.06). The B>C pattern is
indistinguishable between "coordinate helps slightly" and "constant channel
hurts slightly" (constant input is linearly degenerate and absorbable into
bias — acknowledged pre-run). Slot1/Slot2 divergences are within ±1–4 eval
examples (batch 32 ⇒ 3.125pp/example granularity).

### FROZEN DECISION RULE OUTCOME

H_COORD required B to beat BOTH A and C by ≥ +8pp with seed-level
significance. **B − A = +0.31pp fails by ~8pp; B − C = +4.38pp fails by ~4pp.**
**H_COORD IS NOT SUPPORTED — confirmatory falsification** with proper
replication (5 independent models/arm vs the prior campaign's 1), a
training-time sham arm, and fresh seeds.

### Mechanism checks: coordinate counterfactuals (Arm B, eval-time)

Per-seed G_coord = intact − mean(zeroed, shuffled, reversed, constant),
interior slots:

| seed | G_slot1 | G_slot2 |
|---|---|---|
| 801 | +4.70 | 0.00 |
| 802 | −5.47 | −0.78 |
| 803 | +3.13 | −2.34 |
| 804 | −8.59 | +1.56 |
| 805 | −3.12 | −5.47 |

Mean G_slot1 = −1.87, G_slot2 = −1.41, signs inconsistent across seeds.
**Corrupting the coordinate does not systematically hurt the coordinate-trained
models** — replicating the 2026-09-18 counterfactual null with 5 independent
models. Even the weak B>C signal cannot be attributed to positional usage.

### Causal controls

- State lesion (tau=16): 0.0% on all slots for 14/15 runs (exception: A/801
  71.9/46.9/53.1/53.1 — one baseline retains a partial feedforward path).
  NOTE: this differs from the prior campaign artifacts (~48–52% at
  convergence); the sweep lesion semantics zero the state each step rather
  than exposing a feedforward path. Descriptive only; identical across arms.
- Identity gap (intact − shuffle) at interior slots: inconsistent sign across
  seeds/arms (−18.8 to +12.5), mean ≈ 0 — consistent with the interior not
  performing instance-specific sequential computation in ANY arm at L=16.
- Training loss: A/801 reached 0.00017 (train acc 100%); B/C final
  train_loss 0.042–0.086 (train acc 96.9–100%) — all arms fit training data;
  the interior failure is a generalization/mechanism failure, not
  underfitting.

### Parameter counts (VERIFIED from manifests)

A: 43,715; B: 43,811; C: 43,811 — B and C match EXACTLY; A delta = +96
parameters (~0.22%; the prereg estimated +128 from dense1 input width — the
measured delta is smaller; recorded as a prereg estimation correction).

## ADVERSARIAL REVIEW (3 workers, full JSON persisted)

`~/.hermes/cache/scratch/ippr/postrun_review.json` (6,371–6,376 prompt tokens).

Worker-artifact note: the PRE-RUN review (code-auditor, falsification-arbiter,
experiment-designer) was invoked with the same packet but its output was only
partially captured (piped through a terminal tail); the captured
falsification-arbiter output raised four verification gaps, all of which were
independently verified against source before the run (coord block in shared
perceive path, sweep config reconstruction from checkpoint manifest,
label-leakage via Trainer::train_step inspection, eval --coord-mode
intervention path). The post-run review persisted full JSON to disk.
Verified findings incorporated above; disposition of blocking claims:

1. "Single-shot correction is a prereg violation" — REJECTED. Prereg §7
   explicitly covers post-preregistration fixes: document in the changelog and
   rerun affected arms from scratch. The chunk-resume divergence was an
   infra/procedure bug (Adam state not persisted), arm semantics unchanged
   (identical CLI flags per arm), documented in commit 57f2a88 and this
   report's PROCEDURAL CORRECTION section. The invalid battery is preserved.
2. "Arm C input-support claim false" — ACCEPTED as a prereg WORDING error
   (C's support is the singleton {0.5}, not [−1,1]). The sham choice itself
   was correctly frozen before results; the justification sentence overstated
   equivalence. Noted in the changelog.
3. "B−C p=0.062 is the n=5 test floor; decision rule can't adjudicate 5/5" —
   ACCEPTED as a power limitation of the frozen design. The B−C pattern is
   reported as SUGGESTIVE and uninterpretable between the two candidate
   explanations (see decomposition above). No claim rests on it.
4. "Coordinate counterfactuals not reported" — addressed: extracted from
   per-seed artifacts and reported above (mechanism-check null).
5. "Parameter counts / loss curves not reported" — addressed (manifests).
6. "Eval seed = training seed inflates noise floor" — ACCEPTED as a design
   note; does not affect the paired null.

## MECHANISTIC INTERPRETATION

- **H_COORD: NOT SUPPORTED (falsified, confirmatory).** Explicit absolute
  position (a) does not improve the interior over baseline, (b) does not
  produce models that causally depend on the coordinate, and (c) does not
  restore the identity gap. Combined with the prior probes (prefix parity
  physically absent downstream; causal flip probability 0.0), the interior
  failure is not a missing-positional-information problem.
- **H_ATTENUATION / H_OPT: STRENGTHENED but not uniquely confirmed.** A null
  coordinate result rejects only the simple missing-position explanation. The
  surviving candidates — state propagation/attenuation limits, an
  optimization/credit-assignment saddle (interior query slots get no usable
  training signal through 16 recurrent steps), and readout limitations — are
  not discriminated by this experiment.

## NEXT DECISIVE EXPERIMENT (proposed, NOT executed)

Discriminate H_OPT (credit-assignment/optimization saddle) from H_ATTENUATION
(propagation limit): auxiliary supervision — train with per-step intermediate
parity losses at increasing horizons (curriculum over step count) or with a
deep-supervision readout at intermediate ticks at L=16, matched seeds/arms.
If interior slots learn only with per-step signal, the barrier is
credit-assignment through the unrolled recurrence (H_OPT); if per-step signal
also fails to propagate, the barrier is transport/attenuation (H_ATTENUATION).
Alternative (weaker): longer training / larger tau on the coordinate arm —
already bounded by prior tau sweeps.

## CLAIMS UPDATED

- live_claim_ledger.md: new claim C-COORD-015 — "Explicit scalar position does
  not causally contribute to the L=16 interior failure; H_COORD falsified
  (confirmatory, N=15 converged runs, sham-controlled, fresh seeds)." STATUS:
  REFUTED for H_COORD / SUPPORTED for the null.
- research_debt_ledger.md: RD-IPPR-017 (coordinate channel) resolved —
  negative; new RD-IPPR-018 open: discriminate H_OPT vs H_ATTENUATION via
  auxiliary supervision at L=16.

## LIMITATIONS

- n=5 seeds/arm: exact sign-flip floor p=0.0625 for perfectly consistent
  directions; effects < ~3pp (one eval example) are unresolvable.
- tau=16 only (frozen per prereg §9); transient effects at other tau are
  untested (bounded by prior tau sweeps showing no interior rescue).
- B−C decomposition ambiguity (coordinate helps vs constant hurts) is
  unresolvable without a permuted-coordinate training arm (declined in prereg
  to keep the design minimal).
- The chunk-resume instability finding means prior incremental-training
  artifacts (including the 2026-09-18 coordinate campaign) carry an
  unquantified resume-perturbation risk; their conclusions are directionally
  consistent with this battery but were produced under the weaker procedure.

## RAW ARTIFACT PATHS

- runs/ippr_coordinate/ — 15 per-run JSONs, battery_manifest.json,
  analysis_summary.json, checkpoints per arm/seed
- runs/ippr_coordinate_chunked_invalid/ — preserved invalid chunked battery
- reports/ippr_coordinate_channel_protocol.md — preregistration (commit 6670875)
- scripts/run_ippr_coord_battery.py, scripts/analyze_ippr_coord_battery.py

## ANDROID EXPORT

/sdcard/Download/TitanText/ippr_coordinate/ (copied with SHA-256 verification —
see below).

## COST

DeepSeek delegation (verified from telemetry): pre-run review (falsification-
arbiter captured) $0.00163 @5,528 prompt tokens; post-run review
(skeptical-reviewer + experiment-critic + counter-hypothesis-generator)
$0.01311 total (3 × ~6,375 prompt tokens). Pre-run review for all 3 roles was
invoked; only the falsification-arbiter output was captured in-conversation
(see Worker-artifact note above). Local compute: 15 single-shot runs (~45 min)
+ invalid chunked battery (~50 min, preserved as negative evidence) +
diagnostic runs.


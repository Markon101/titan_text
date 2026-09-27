# Adaptive Halting Qualification Report

**Date**: 2026-09-26 | **Branch**: `exp/ascii-adaptive-halting`
**Battery commit**: recorded in `reports/raw/qualification_battery/qualification_manifest.json` (`git_head`)
**Checkpoint**: `checkpoints/ascii_v1`, model.safetensors sha256 `97d00bd0...3e3e`, 43,844 params, 64ch, hidden 96, `carry_channels=0`
**Protocol**: [ascii_halting_qualification_protocol.md](ascii_halting_qualification_protocol.md) (frozen; changelog documents 6 pre-run clarifications, no behavioral changes)

## RESEARCH QUESTION

Does state-dependent allocation of recurrent latent computation improve structured
ASCII generation beyond what is explained by total compute budget, random tau
variation, the nearest fixed operating point, or positional/structural schedules?

## IMPLEMENTATION VERIFIED (from source, not reports)

All verified in `src/ascii_sampler.rs::generate`:
- (A/B) Field re-initialized from token embeddings per emitted token
  (line 303) — recurrence is within-token only; no cross-token/cross-sample state.
- (C) tau_min enforced via the stability gate `tick >= min_t` (line 410); tau_max
  via loop bound `1..=max_t` (line 357). No accumulator clamping exists.
- (D) `used = tick` (line 408) equals ticks actually executed.
- (E) Emitted logits read from the post-halt state (line 407 assignment, line 428 read).
- (F) RelativeDelta = `||x_t - x_{t-1}||_2 / (||x_t||_2 + 1e-6)` (lines 373-379),
  denominator = post-step norm. All channels of the active position (lines 358-363;
  `field.x` is [batch, seq_len, channels], position-dim narrowed only).
- (H) `--lesion-state` is a clean bypass returning 0 ticks (lines 307-309); sanity
  run in this battery confirmed zero ticks.
- (I) Fixed mode uses the identical per-token state-initialization path (lines 312-316).
- (J/K) Random mode draws ticks from the same StdRng as sampling (lines 279, 321, 505);
  irrelevant under greedy, shared stream under temperature 0.7 (noted; the
  qualification battery avoided Random mode in favor of Schedule mode).
- (L/M/N) Schedule mode indexes by generated-token position (line 328), falls back
  to `--tau` when exhausted (lines 329-331); EOS truncates consumption (lines 554-556).

## PREVIOUS CLAIMS RECHECKED

| Prior claim | Status after qualification |
|---|---|
| "Calibrated adaptive achieves Pareto superiority (+27.4% over tau=4, 43.4% less compute than tau=8)" | **DOWNGRADED/REFUTED as a superiority claim**: +27.4% over tau=4 is a compute-budget artifact (sensitivity: +0.005, p=0.41); adaptive is significantly *worse* than fixed tau=5 at matched budget. "43.4% less compute" remains true only as a mean-tau ratio (4.53/8.0). |
| "Causal alignment vs shuffled strictly positive (p=0.0139, 5 seeds)" | **UNRESOLVED→NOT REPLICATED**: on the same-battery paired design at the seed unit, A−C_matched = +0.029, seed-perm p=0.36; sensitivity −0.022. |
| "theta*=0.25 sweet spot" | **SUPPORTED descriptively** (response curve stable), but the operating point it selects does not beat simply fixing tau=5. |
| "Budget-operating efficiency is the primary mechanism; per-token alignment modest secondary" | **SUPPORTED** — and sharpened: the per-token alignment component is statistically indistinguishable from zero. |
| "Entropy decoupled from tau (r=-0.1475, 'uncorrelated')" | **CORRECTED**: token-level r=-0.242 (N=1200, descriptive); sequence-cluster bootstrap CI [-0.373,-0.098] — a weak but significant *anti*-correlation; "uncorrelated" was overstated. |
| "Per-token adaptivity causally established" | **REFUTED** at this checkpoint/protocol. See CLAIM LANGUAGE. |

## FROZEN PROTOCOL & CONTROL ARMS

Arms (30 runs each, 10 held-out seeds x 3 prompts, temp 0.7, theta 0.25 frozen):
A_adaptive; B_fixed_tau4; **B_fixed_tau5**; B_fixed_tau8; C_matched (per-(prompt,seed)
multiset permutation of same-battery Arm A ticks, rng seed+9999); D_matched (per-(prompt,seed)
i.i.d. draws from the frozen adaptive tick distribution {4:641,5:529,6:30}, rng seed+7777,
length = Arm A length). Plus lesion sanity (zero ticks OK) and bit-identical
determinism rerun (OK). 182 samples total.

## COMPUTE MATCHING

Realized tick ratio (control/A): C 0.893, D 0.893 mean; 24/30 and 25/30 pairs
within ±10%; 4 and 3 pairs control-truncated (early EOS), retained in primary,
excluded in sensitivity. Matching quality is good but not perfect; the
truncation-excluded sensitivity analysis is the cleaner read and it *hurts* the
adaptive arm (see RESULTS).

## STATISTICAL UNIT

Independent unit = **10 seeds** (3 prompt-sequences nested per seed). Token-level
statistics are descriptive. Two inference procedures reported: seed-cluster
bootstrap CI (10 clusters, 10k resamples) and **seed-level exact sign-flip
permutation (2^10)** — the latter adopted as primary after worker audit showed
the 30-pair token permutation is anti-conservative (nested pairs treated as
exchangeable). All three are reported side by side.

## RESULTS (primary endpoint: nearest_edit_similarity)

| Arm | edit sim (mean ± se) | ticks/run | mean tau |
|---|---|---|---|
| A_adaptive | 0.5535 ± 0.0261 | 179.6 | 4.491 |
| B_fixed_tau4 | 0.4735 ± 0.0317 | 127.2 | 4.000 |
| **B_fixed_tau5** | **0.5937 ± 0.0351** | 183.3 | 5.000 |
| B_fixed_tau8 | 0.6506 ± 0.0109 | 384.0 | 8.000 |
| C_matched | 0.5242 ± 0.0329 | 155.2 | 4.486 |
| D_matched | 0.5436 ± 0.0348 | 160.6 | 4.474 |

Adaptive tick histogram: {4: 641, 5: 529, 6: 30} — the controller operates on
only 3 values; it is a near-binary 4/5 switch with rare 6.

Paired A−X (primary, all 30 pairs; seed-perm p; cluster CI):

| Contrast | mean diff | cluster CI | seed-perm p | seeds neg |
|---|---|---|---|---|
| A−B4 | +0.0800 | [-0.0007, +0.1899] | 0.094 | 3/10 |
| **A−B5** | **−0.0403** | **[-0.0709, −0.0115]** | 0.985 | **8/10** |
| A−B8 | −0.0972 | [-0.1551, −0.0436] | 0.995 | 9/10 |
| A−C_matched | +0.0292 | [-0.0276, +0.1155] | 0.359 | 3/10 |
| A−D_matched | +0.0099 | [-0.0581, +0.1044] | 0.468 | 6/10 |

Sensitivity (exclude control-truncated pairs): A−B4 +0.005 (p=0.38); A−C −0.022;
A−D −0.032. The A>B4 advantage disappears; A>C/D reverses sign.

**Frozen decision rule outcome**: requires A>B5 AND A>D_matched AND A>C_matched.
A<B5 with CI excluding zero → the rule fires H_BUDGET. No adaptive advantage
survives at the correct independence unit.

## NEGATIVE RESULTS

1. Fixed tau=5 beats adaptive at matched compute — the "Pareto superiority"
   headline was an artifact of comparing against tau=4 and interpolating the
   fixed curve instead of running tau=5.
2. State-independent matched controls (C/D) are statistically indistinguishable
   from adaptive in primary analysis and *better* in the truncation-cleaned
   sensitivity analysis.
3. A cheap char-class-mean rule (no state) explains R²=0.27 of tick variance
   (symbols always 4, boundaries ~4.75); per-sequence r(position, tau) mean
   −0.25 (25/30 negative). The schedule is largely a positional/structural rule.
4. Entropy–tau association is weak, negative, and heterogeneous (per-seq r from
   −0.71 to +0.55; 17/30 negative): no useful uncertainty signal.

## ADVERSARIAL REVIEW (5 independent DeepSeek workers, conclusion-blind)

All five independently converged on **H_BUDGET (+H_POSITION, +H_EOS)**. Key
verified worker contributions:
- Sign-flip permutation over 30 nested pairs is anti-conservative (statistics-auditor,
  dynamics-agent) — VERIFIED: seed-level permutation lifts A−B4 from p=0.0068 to
  p=0.094; the token-level p-value was the only thing making A>B4 look real. Adopted.
- C_matched is the load-bearing state-dependence control and it failed to
  separate (p=0.14) — direct evidence against the state-dependence mechanism.
- "43.4% less compute than tau=8" flagged (dynamics-agent) — resolved: true as a
  mean-tau ratio, misleading as a total-ticks statement; wording corrected.
- Zero hallucinated symbols this run; workers correctly declared EVIDENCE ABSENT
  where the packet lacked artifacts (per-pair lists, sanity outputs are in
  `qualification_samples.json` / `qualification_analysis.json`).

GLM corrections of workers: none needed beyond the tau=8 wording clarification;
all blocking claims verified against `src/ascii_sampler.rs` and raw artifacts.

## CONFIRMED CONFOUNDS

- H_BUDGET: confirmed (A<B5, CI excludes 0).
- H_EOS: confirmed contributor (sensitivity analysis flips A−C, A−D sign).
- H_POSITION: supported (R²=0.27 class rule; positional correlation).

## REFUTED CONFOUNDS / CONCERNS

- Carry-channel contamination of the halting metric: **NOT APPLICABLE** —
  `carry_channels: 0` verified in the checkpoint manifest (the concern was
  imported from a different Titan thread).
- State leakage across tokens/samples: **REFUTED** (per-token field re-init,
  per-run process, per-run RNG seed; bit-identical determinism rerun).
- tau_min/tau_max clamping or patience-ordering bugs: **REFUTED** (verified in source).

## UNRESOLVED QUESTIONS

- Full-channel halting metric concentration (which channels dominate the delta)
  remains unmeasured (needs per-channel telemetry; classified IMPLEMENTATION
  FACT ONLY, no carry subspace exists here).
- Whether a *trained-with-adaptive-halting* model (vs. a fixed-tau-trained
  checkpoint sampled adaptively) benefits from adaptive allocation — the current
  result applies to sampling-time adaptation of a tau=4-trained checkpoint only.
- 10 seeds / 3 prompts bound generalization; no prompt-level inference attempted.

## ARTIFACT PATHS & REPRODUCTION

- Raw samples: `reports/raw/qualification_battery/qualification_samples.json` (182 records incl. sanity + rerun)
- Manifest: `reports/raw/qualification_battery/qualification_manifest.json`
- Analysis: `reports/raw/qualification_battery/qualification_analysis.json`
- Runner: `scripts/run_qualification_halting_battery.py`
- Analysis: `scripts/analyze_qualification_battery.py`

Reproduce:
```
cargo build --release
python3 scripts/run_qualification_halting_battery.py
python3 scripts/analyze_qualification_battery.py
```
(both deterministic; the runner self-verifies checkpoint sha256, bit-identical
rerun, and lesion sanity before/while running)

## COST

- Battery compute: 182 local generation runs on ARM64 (minutes, no API cost).
- Worker reviews: protocol review 4 workers (~$0.011) + post-run review 5 workers
  ($0.0113, 14,280 in / 10,240 out tokens). Total delegation cost this task ≈ $0.031.

## CLAIM LANGUAGE (final status)

- OBSERVED/VERIFIED: adaptive operates on {4,5,6} ticks, mean 4.49.
- VERIFIED: fixed tau=5 outperforms frozen-theta adaptive halting on the held-out
  battery at matched compute (cluster CI excludes 0, 8/10 seeds).
- SUPPORTED: budget + a coarse positional schedule explain the adaptive arm's
  performance; sampling-time adaptive halting adds no measurable benefit over
  fixed tau=5 for this checkpoint.
- REFUTED: "causal Pareto superiority of adaptive halting" (as previously claimed);
  "causally established per-token adaptivity".
- SUGGESTIVE: weak anti-correlation of entropy and tau (cluster CI excludes 0).

## NEXT DECISION

Do not pursue sampling-time adaptive halting further on this checkpoint. The
honest mechanism statement is: "adaptive halting is a noisy proxy for a fixed
operating point near tau=5; state-dependent allocation shows no measurable
causal contribution." Any future adaptive-compute work requires training-time
integration (the halter must be learned with the model), not post-hoc sampling.

---

## CONFIRMATORY BATTERY (Git-anchored, frozen at commit c1a30d4, run after)

Correction of record: the 501-605 battery above was disk-preregistered but NOT
Git-anchored (protocol committed after execution), and seeds 501-605 were
previously observed fixed evaluation seeds, not pristine. A confirmatory
battery was therefore run per the frozen protocol addendum:

- Seeds: [701..710] — verified never used as seed values anywhere in the repo.
- Frozen commit before execution: c1a30d4 (manifest records git_head c1a30d49).
- Arms/theta/metric/matching/decision rule/analysis: unchanged.
- Determinism rerun: OK; lesion sanity: OK.

Results (30 runs/arm):
A_adaptive 0.5904+-0.0140 (ticks 199.9, tau 4.491) | B4 0.5878+-0.0226 |
B5 0.6342+-0.0191 | B8 0.6118+-0.0193 | C_matched 0.6104+-0.0255 |
D_matched 0.6343+-0.0228.

Paired A-X (primary; seed-cluster CI; seed-perm p):
- A-B4: +0.0027, CI [-0.0283,+0.0477], p=0.50, 5/10 seeds neg
- A-B5: -0.0438, CI [-0.1144,+0.0215], p=0.87, 6/10 seeds neg
- A-B8: -0.0213, CI [-0.0900,+0.0434], p=0.72, 5/10 seeds neg
- A-C_matched: -0.0200, CI [-0.1054,+0.0569], p=0.69, 4/10 seeds neg
- A-D_matched: -0.0438, CI [-0.1126,+0.0222], p=0.87, 6/10 seeds neg
Sensitivity (truncation-excluded): A-B5 -0.0548; A-D -0.0672
(CI [-0.1286,-0.0165], excludes zero - matched random control significantly
BETTER than adaptive).

CONFIRMATORY VERDICT (frozen decision rule verbatim): A>B5 FALSE,
A>D_matched FALSE, A>C_matched FALSE -> H_BUDGET stands; adaptive
state-dependent compute allocation NOT supported. Directionally identical to
the first battery (A loses to B5; A at or below matched controls); the prior
+27.4%-over-tau=4 headline collapses to +0.003 on fresh seeds.

CLAIM LANGUAGE (final): "These results support no detectable causal
contribution from state-dependent recurrent compute allocation for
sampling-time adaptive halting on this checkpoint; performance is explained
by total compute budget, with fixed tau=5 the best matched operating point."

Artifacts: reports/raw/qualification_battery_confirmatory/ (samples, manifest
with git_head c1a30d49, analysis).

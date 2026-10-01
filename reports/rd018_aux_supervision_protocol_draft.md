# RD-018 Protocol Draft: Auxiliary Deep Supervision at Interior Slots (L=16 Iterated Parity)

**Status**: DRAFT preregistration — not yet frozen. Commit-freeze before any run.
**Date**: 2026-10-01. Session: agy literature pull + 7-specialist research meeting.

## Question (RD-018, open)
Does interior-slot stagnation at L=16 reflect H_OPT (credit-assignment saddle) or
H_ATTENUATION (carry propagation limit, xi ~ 4-6 cells vs 16-cell grid)?

Discriminator: train-time-only auxiliary per-tick loss at interior slots. Interior
learns only with aux signal -> H_OPT. Aux signal also fails -> H_ATTENUATION
(conditional on the asymmetric-trap controls below).

## Prior evidence anchoring the design
- Slot 0 (cell 3) solved 71-86%; interior slots 1,2 at 44-50% across 1000 epochs,
  curriculum, coordinate channels (H_COORD falsified, C-COORD-015), and dense
  supervision arms (escalation_b).
- C-falsified this session: "legacy STE returns zero gradients through carry" —
  REFUTED at source: training path uses quantize_carry with identity-gradient STE
  (src/nca.rs:406-411, called at src/nca.rs:709); the hard-sign site (src/nca.rs:997)
  is the non-autograd CPU ping-pong eval path. A gradient-norm telemetry gate is
  still included as a cheap positive control.
- Single-slot supervision precedent (exp6_slot3_only, 56.1%) supervised a READOUT,
  not intermediate ticks; per-tick aux is a materially different intervention.

## Literature calibration (agy/Gemini pull, 2026-10-01; saved in session scratch)
Lee et al. 2015 (DSN), Szegedy et al. 2015 (GoogLeNet aux), Pezeshki et al. 2021
(gradient starvation), Kaiser & Sutskever 2016 (Neural GPU: cellular carry chains
needed stepwise supervision), Velickovic et al. 2020 (algorithmic hint losses),
Arjovsky et al. 2016 / Jing et al. 2017 (unitary/EUNN transport), Bengio et al.
2013 (STE). All VERIFIED by the model; none contradicted.

## Arms (3-arm, matched single-shot 1000-epoch training — NO chunk-resume)
- Arm A: terminal-loss baseline (replication gate: interior at chance, >=4/5 seeds).
- Arm B: + auxiliary per-tick loss at interior slots, weight alpha = 0.3 (frozen;
  GoogLeNet precedent), targets = true intermediate slot parity.
- Arm C: sham auxiliary control — identical machinery and compute, targets are
  constant-sham parity (bias-absorbable control, same convention as C-COORD-015).

## Seeds & sample size
- 5 fresh seeds per arm from the UNUSED pool (901-905, 906-910, 911-915).
  Used pools: 42/101/202/303/404 (ascii/general), 801-805 (coordinate campaign).
- Power: n=5/arm resolves only >~3pp with sign-flip exact tests (floor p=0.062).
  Frozen primary threshold: Arm B interior mean exceeds Arm A by >= +8pp with
  >=4/5 seed-level wins. Secondary (exploratory, labeled): alpha sweep 0.1/1.0 on
  2 seeds each ONLY if primary is null and B>A direction is suggestive.
- Eval: seed-level sign-flip permutation + paired bootstrap; InteriorMean primary;
  never token-level pooling.

## Decision rule (frozen)
- H_OPT supported: B-A >= +8pp AND carry-lesion counterfactual on Arm B collapses
  interior back to chance (guards against local-shortcut false positives).
- H_ATTENUATION supported: B ~= A ~= C at chance, AND (a) aux-head gradient norms
  > 1e-4 (positive control), (b) frozen-model linear probe on final-state interior
  carry channels fails (forward information genuinely absent).
- Otherwise: inconclusive; record as such. No interpretive override.

## Required Rust changes (scoped this session)
**IMPLEMENTED 2026-10-01, build clean, 112/112 tests pass.** Smoke-verified at
L=16, 20 epochs, seed 901: Arm A aux=0.0000; Arm B aux CE 1.166->0.776 with
elevated terminal loss (alpha term active); Arm C sham aux CE 1.136->0.837
through identical machinery on constant targets. Epoch trace now carries an
aux= column (mean interior-slot aux CE per step).
1. src/train.rs: extend loss accumulation (multi-tick path, ~L154-181) with
   per-tick interior-slot masked CE terms: L = L_final + alpha * mean_t L_aux(t).
   Reuse apply_target_slot_filter / masked_cross_entropy_loss; add interior-slot
   mask variant.
2. Config/CLI: --aux-supervision <alpha> (0 = off) and --aux-sham flag (constant
   targets through identical machinery).
3. Telemetry: per-arm manifest fields for aux loss trajectory and aux-head grad
   norm (Option<f32> finite-filtered per NaN/JSON-null convention; divergence
   detected from manifest, never exit codes).

## Pitfalls carried from literature review (controls)
- Scratchpad destruction: aux heads use dedicated readout projections, never raw
  H/Cc equality to targets.
- Aux-weight domination: alpha frozen at 0.3; no post-hoc retuning.
- Local-shortcut false positive: mandatory carry-lesion counterfactual on any B win.
- STE variance compounding: monitor discrete sign-transition fraction; grad clip
  1.0 on aux terms.
- Asymmetric negative-result trap: the (a)/(b) positive controls above are
  mandatory before any H_ATTENUATION claim.

## Explicitly out of scope this campaign
- DISC_2 (unitary/skew-symmetric continuous recurrence) — queued next if
  H_ATTENUATION wins, as the mechanistic follow-up.
- RD-015 training-time adaptive halting (ASCII thread) — unchanged priority.

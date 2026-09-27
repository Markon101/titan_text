# PREREGISTRATION — IPPR Coordinate-Channel Confirmatory Experiment (L=16)

Status: FROZEN before execution. This file is committed and pushed to
`exp/ippr-coordinate-channel` BEFORE any training run of this battery. The
preregistration commit hash is recorded in the run manifest produced by
`scripts/run_ippr_coord_battery.py`.

## 1. Research question

Why did Titan learn genuine instance-specific sequential parity computation at
L=8 but fail in the bulk interior at L=16?

- **H_COORD**: the model lacks sufficiently explicit positional information to
  bind computation correctly in the long interior; an explicit scalar
  coordinate channel should rescue interior slots (Slots 1, 2).
- **H_ATTENUATION / H_OPT**: the limiting factor is state propagation /
  information attenuation / an optimization saddle (credit-assignment barrier),
  NOT missing positional information. Explicit coordinates should NOT
  materially rescue the interior.

## 2. Canonical prior state (verified against raw artifacts 2026-09-26)

All numbers re-verified today against the stored JSON artifacts, not report prose:

- **L=8 (epochs 100–500 of campaign_arm2_curriculum.json)**: intact slots
  [68.75, 68.75]% vs batch-state shuffle 50.0% → identity gap G_identity
  ≈ +18.8pp: genuine instance-specific sequential computation. REPLICATED.
- **L=16 (epochs 600–1000, same artifact; also campaign_arm1_from_scratch.json
  and campaign_arm2_coordinate.json)**: Slot0 68.75–77.5%, Slots 1–2
  40.0–54.4% (chance ≈ 50%), Slot3 55.0–60.0%. Interior failure replicated
  across three artifacts and multiple training configurations.
- **Coordinate intervention (campaign_arm2_coordinate.json, 2026-09-18,
  preregistered)**: coord-trained model at epoch 1000: intact 57.34%,
  slots [68.75, 48.75, 54.38, 57.50]. Coordinate counterfactuals at tau=16:
  zeroed 55.94, shuffled 54.69, reversed 54.69, constant 56.25 →
  G_coord_zeroed = +1.41% << preregistered +8% threshold. Interior slots did
  not move. H_POS (= H_COORD) was FALSIFIED on that run.
- **Probes (causal_influence_analysis.json, prefix_parity_probes.json,
  state_inversion_probes.json)**: prefix-parity probes fail downstream
  (physical absence, not readout failure); causal flip probability 0.0%;
  prefix task easier locally but no prefix-based global strategy learned.

**Known weaknesses of the prior coordinate test (this experiment's reason to exist):**
1. ONE trained model per arm (training seed 42 continued across increments);
   the "N=5 seeds" were evaluation-data seeds, so the falsification rests on a
   single model run per arm.
2. No TRAINING-TIME dimension-matched sham arm: the extra coordinate
   dimensionality/parameter opportunity was never trained against a
   non-informative channel from scratch.

## 3. Design: minimal three-arm, five fresh seeds

Arms (identical except the intervention; predetermined order A → C → B):

- **Arm A — BASELINE**: existing architecture, no coordinate channel
  (`--coord-channel` absent).
- **Arm B — COORD**: `--coord-channel` (intact scalar coordinate
  p_i = 2i/(L-1) - 1 ∈ [-1, 1] at every position, implemented in
  `NCA::generate_coordinates`, src/nca.rs:136).
- **Arm C — SHAM**: `--coord-channel --coord-train-mode constant`
  (identical added perception dimension, constant 0.5 at every position —
  carries NO positional information; implemented via new
  `NcaConfig::coord_train_mode`, src/nca.rs perceive path).

Sham choice justification (frozen BEFORE results): CONSTANT CHANNEL best
preserves the exact parameter delta and dimensionality of Arm B while removing
all absolute-position signal (a permuted coordinate would additionally inject
variance and a random-static channel would alter input statistics
distributionally; the constant is the minimal sufficient control). The value
0.5 lies inside the coordinate range so B and C have identical input support.

- **Seeds**: training/eval seeds [801, 802, 803, 804, 805] — verified FRESH:
  full-repository search for standalone integers 801–805 found only training
  step numbers, sha256 hash substrings, and binary weight data; the prior
  seed battery was 42/101/202/303/404. Five seeds = 5 independent model runs
  per arm (the independent statistical unit), 15 runs total.
- **Primary length**: L=16, `--dev-steps 16 --seq-len 16 --zero-boundary`,
  `--task iterated-parity`, 1000 epochs, batch 32, default lr/viscosity/state
  norm (identical to prior campaign configuration).
- **Evaluation**: tau = 16 (`sweep --budgets 16 --batch-size 32`), paired
  eval seeds = the arm's training seed.
- **Parameter delta**: Arm B/C perception is 3C+1 vs A 3C → dense1 gains
  +128 weights (C=64, hidden 128) ≈ +0.29% of parameters. B and C match
  EXACTLY; A differs minimally as preregistered.

## 4. Primary endpoints (frozen)

At L=16, tau=16:
- **Slot 1 accuracy** and **Slot 2 accuracy**, reported individually AND as
  predeclared `InteriorMean = mean(Slot1, Slot2)`.

Secondary: Slot 0, Slot 3, overall intact accuracy, identity gap
(G_identity = intact − batch-state-shuffle), state lesion accuracy
(feedforward-only path), training loss curve (stored per-100-epoch intact
eval), coordinate counterfactuals for Arm B (zeroed/shuffled/reversed/
constant) as mechanism checks.

## 5. Decision rules (frozen)

- **Replication gate (§18)**: if Arm A solves the interior (InteriorMean
  materially above chance, > +8pp over chance at more than one seed), STOP —
  do not interpret coordinates; investigate baseline drift.
- **H_COORD supported**: Arm B improves InteriorMean over BOTH Arm A and Arm C
  by ≥ +8pp (the prior preregistration's effect threshold), with the
  improvement concentrated in interior slots, reproduced at a majority of
  seeds (≥4/5 directionally), and seed-level inference (paired sign-flip
  permutation across the 5 seeds) significant at α=0.05.
- **H_COORD falsified (confirmatory)**: Arm B ≤ Arm C ≈ Arm A on InteriorMean
  (all within noise, seed-level p > 0.05 both directions), interior slots
  remain at chance. This would replicate the 2026-09-18 falsification with
  proper replication and a training-time sham.
- **Ambiguous**: coordinate improves aggregate but not interior, or improves
  only one interior slot inconsistently, or high seed variance prevents any
  stable conclusion. Do not force ambiguous data into either hypothesis.

## 6. Statistical unit and inference (frozen)

The independent unit is the SEED/MODEL RUN. Paired by seed across arms.
Analysis: per-seed paired differences, mean difference, seed-level sign-flip
permutation test (2^5 exact) and seed-level bootstrap CI. Token-level /
per-example counts are DESCRIPTIVE ONLY and never used for inference
(nested-data rule from the adaptive-halting qualification).

## 7. Failure/stop conditions

- Baseline replication gate fails (§5) → stop and investigate.
- Any training run NaNs or diverges → record, exclude that seed from BOTH
  arms it appears in, and report (do not substitute seeds).
- Implementation bug discovered post-preregistration → fix may NOT alter the
  arm semantics already trained; document any fix in the changelog and rerun
  affected arms from scratch.

## 8. Label-leakage audit (required before acceptance)

Coordinate depends only on position and L (src/nca.rs:136-146); parity labels
never enter the perception channel; the sham constant cannot encode labels;
evaluation batching mirrors training batching. A code auditor reviews this
explicitly before runs.

## 9. Changelog

(Frozen at commit time — see git history for the preregistration hash.)
